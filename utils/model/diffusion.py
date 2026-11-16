import torch
import torch.nn as nn
import torch, torch.nn.functional as F
import torch_geometric as tg
import math

from torch_geometric.utils import softmax as global_softmax
from torch_geometric.nn import MessagePassing, inits, global_mean_pool

from diffusers import DPMSolverMultistepScheduler
from torch_scatter import scatter, scatter_max, scatter_add
from utils.model.t2p import Text2Program, LabelEmbedding, make_pseudo_program_from_text
from utils.eval.evaldi import _get_ground_truth_x0
from utils.data.GraphConstructor import GraphConstructor
from torch_geometric.utils.num_nodes import maybe_num_nodes


def gumbel_softmax(src, index, tau=-1, num_nodes=None):
    """modified from torch_geometric.utils import softmax"""
    num_nodes = maybe_num_nodes(index, num_nodes)

    gumbels = -torch.empty_like(src, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (src + gumbels) / tau  # ~Gumbel(logits,tau)

    out = gumbels - scatter_max(gumbels, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    argmax = scatter_max(out, index, dim=0, dim_size=num_nodes)[1]
    out_hard = torch.zeros_like(out, memory_format=torch.legacy_contiguous_format).scatter_(0, argmax, 1.0)
    out_hard = out_hard - out.detach() + out

    return out, out_hard


def softmax_to_hard(y_soft, dim):
    """the function uses the hard version trick as in gumbel softmax"""
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    return y_hard - y_soft.detach() + y_soft


def position_encoder(d_model, max_len=20):
    # PE(l): floor positional encoding -- Eq. 8 (Table A3: "Floor positional encoding PE")
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
 

def get_voxel_floor_level(vfc, vbi):
    # floor ids are serialized in batch, need to subtract the accumulated num of node in each graph
    pool = -tg.nn.max_pool_x(cluster=vbi, x=-vfc, batch=torch.cuda.LongTensor([0] * vfc.shape[0]))[0]
    return vfc-pool.index_select(0, vbi)



def MLP(dims, act=""):
    assert len(dims) >= 2
    nn_list = []
    for i in range(len(dims)-1):
        nn_list.append(nn.Linear(dims[i], dims[i+1], bias=True))
        if "leaky" in act:
            nn_list.append(nn.LeakyReLU(negative_slope=0.01))
        elif "relu" in act:
            nn_list.append(nn.ReLU())
        elif "tanh" in act:
            nn_list.append(nn.Tanh())
        elif "sigmoid" in act:
            nn_list.append(nn.Sigmoid())
    return nn.Sequential(*nn_list)


def extract_pos(feature):
    """ Split position features and the rest"""
    pos = feature[:, 3:6]  # 3, 4, 5 are position coordinates
    non_pos = torch.cat((feature[:, 0:3], feature[:, 6:]), dim=-1)
    return pos, non_pos


# ==== MODULE: Program GNN -- message passing (Table A3: "Program GNN") ====
class ProgramBlock(MessagePassing):
    def __init__(self, hidden_dim):
        super(ProgramBlock, self).__init__()
        self.aggr = 'mean'
        self.messageMLP = MLP([2 * hidden_dim, hidden_dim])   # Program GNN "Message MLP"
        self.updateMLP = MLP([3 * hidden_dim + 1, hidden_dim], act="leaky")   # Program GNN "Update MLP"

    def forward(self, feature, edge_index, class_index, class_feature, batch_index, far):  
        """
        feature=x, edge_index=graph.program_edge, class_index=graph.program_class_cluster,
        class_feature=graph.program_target_ratio, batch_index=graph.program_class_feature_batch
        """
        kwargs = {"feature": feature, "class_index": class_index, "class_feature": class_feature, "batch_index": batch_index, "far": far} 
        return self.propagate(edge_index, size=None, **kwargs)

    def message(self, feature_j, feature_i=None):
        # m_k: pairwise program messages (Section 3.2.3 (1))
        return self.messageMLP(torch.cat((feature_i, feature_j), dim=-1))

    def update(self, aggr_out, feature=None, class_index=None, class_feature=None, batch_index=None, far=None):
        # Eq. 14: program node update  h_k += MLP_upd([h_k, m_k, TPR_c * h_c, FAR])
        agg_class_feature = tg.nn.avg_pool_x(cluster=class_index, x=feature, batch=batch_index, size=GraphConstructor.number_of_class)[0] * class_feature.view(-1, 1)  # TPR_c(k) * h_c(k)
        pool_class_feature_on_node = agg_class_feature.index_select(0, class_index)
        return self.updateMLP(torch.cat((feature, aggr_out, pool_class_feature_on_node, far), dim=-1))


class ProgramGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, noise_dim, layer_num):  # hidden_dim = 128
        super(ProgramGNN, self).__init__()
        self.layer_num = layer_num
        self.enc = MLP([input_dim + noise_dim + 1, hidden_dim])   # Program GNN "Node encoder MLP": h_k^(0) = MLP_enc([1_c(k), l, FAR])
        self.block = ProgramBlock(hidden_dim)

    def forward(self, cf, slf, e, cc, tr, cfb, z=None, far=None): 
        """
        cf=graph.program_class_feature, slf=graph.story_level_feature, e=graph.program_edge, cc=graph.program_class_cluster,
        tr=graph.program_target_ratio, cfb=graph.program_class_feature_batch, z=noise, far=per-node FAR (F from eq1,eq4 in the original paper)
        """
        parts = [cf, slf.view(-1, 1)]
        if z is not None:
            if z.dim() == 1: 
                z = z.view(-1, 1)
            parts.append(z)

        far_node = far.view(-1, 1) if far is not None else torch.zeros(cf.shape[0], 1, device=cf.device)
        parts.append(far_node)
        x = torch.cat(parts, dim=-1)
        x = self.enc(x)
        for i in range(self.layer_num):   # 4 residual message-passing iterations, shared weights (Section 3.2.3 (1))
            x = x + self.block(feature=x, edge_index=e, class_index=cc,
                            class_feature=tr, batch_index=cfb, far=far_node)
        return x


# ==== MODULE: Denoising GNN -- Voxel block (Table A3: "Message passing x12") ====
class VoxelBlock(MessagePassing):
    def __init__(self, hidden_dim, if_agg_story_in_update=True):
        super(VoxelBlock, self).__init__()
        self.aggr = 'mean'
        self.if_agg_story_in_update = if_agg_story_in_update  # True in VoxelGNN_G, False in VoxelGNN_D
        self.messageMLP = MLP([2 * hidden_dim + 3, hidden_dim])   # "Message MLP"
        self.updateMLP = MLP([4 * hidden_dim, hidden_dim], act="leaky") if if_agg_story_in_update else MLP([2 * hidden_dim, hidden_dim])   # "Update MLP"

    def forward(self, voxel_feature, edge_index, batch_index, position, floor_index):
        """
        voxel_feature=v, edge_index=graph.voxel_edge, position=pos, floor_index=graph.voxel_floor_cluster, batch_index=graph.voxel_feature_batch
        cross_edge_voxel_index=graph.cross_edge_program_index_select, cross_edge_program_index=graph.cross_edge_voxel_index_select,
        """
        kwargs = {"voxel_feature": voxel_feature, "batch_index": batch_index, "position": position, "floor_index": floor_index}
        return self.propagate(edge_index, size=None, **kwargs)

    def message(self, voxel_feature_j, voxel_feature_i=None, position_i=None, position_j=None):
        # Eq. 12: m_i = mean_j MLP_msg([h_i, h_j, dp_ij])
        return self.messageMLP(torch.cat((voxel_feature_i, voxel_feature_j, position_i-position_j), dim=-1))

    def update(self, aggr_out, voxel_feature=None, floor_index=None, batch_index=None):
        if self.if_agg_story_in_update:
            # Eq. 13: h_i += MLP_upd([h_i, m_i, h_floor, h_bldg - h_floor])
            agg_story_feature, agg_story_feature_batch = tg.nn.avg_pool_x(cluster=floor_index, x=voxel_feature, batch=batch_index)  # h_floor: pool over same floor
            agg_building_feature = tg.nn.avg_pool_x(cluster=agg_story_feature_batch, x=agg_story_feature, batch=torch.cuda.LongTensor([0] * agg_story_feature.shape[0]))[0]  # h_bldg: pool over all floors
            agg_story_feature = agg_story_feature.index_select(0, floor_index)  # populate the feature to each voxel node
            agg_building_feature = agg_building_feature.index_select(0, batch_index)  # populate the feature to each voxel node
            return self.updateMLP(torch.cat((voxel_feature, aggr_out, agg_story_feature, agg_building_feature-agg_story_feature), dim=-1))
        else:
            return self.updateMLP(torch.cat((voxel_feature, aggr_out), dim=-1))


# ==== MODULE: Denoising GNN -- Cross-modal attention (Table A3: "Cross-modal attention") ====
class Attention(nn.Module):
    """
    Cross-Attention module -- Eq. 9 and Eq. 10
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.dec = MLP([hidden_dim, hidden_dim // 2, 2])   # "Fuse MLP" gamma_i (Eq. 10 gate)
        # Cross-Attention for Q, K projector
        self.NN_v = nn.Linear(hidden_dim, hidden_dim) # W_vox: "Voxel projection" (Eq. 9)
        self.NN_p = nn.Linear(hidden_dim, hidden_dim) # W_prog: "Program projection" (Eq. 9)
        self.theta = nn.Parameter(torch.Tensor(1, hidden_dim))   # w: "Attention vector" (Eq. 9)
        inits.glorot(self.theta)

    def forward(self, voxel_feature, condition, cross_edge_program_index, cross_edge_voxel_index):
        """
        voxel_feature (Nv, hid): from Voxel GNN 
        condition (Np, hid): conditions from ProgramGNN or Text Encoder
        """
        
        # gamma_i = [softmax(MLP_gate(h_i))]_1  -- Eq. 10 gate
        soft_mask = torch.nn.functional.softmax(self.dec(voxel_feature), dim=-1)
        out_mask_soft = soft_mask[:, 0].view(-1, 1)

        # 2. prepare Q & K
        program_feature_j = condition.index_select(0, cross_edge_program_index) # K
        voxel_feature_i = voxel_feature.index_select(0, cross_edge_voxel_index) # Q

        # 3. Eq. 9: u_ik = w^T tanh(W_vox h_i + W_prog h_k)
        att = (self.theta.view(1, -1) * torch.tanh(self.NN_v(voxel_feature_i) + self.NN_p(program_feature_j))).sum(-1)

        # 4. softmax_k over program nodes on the same voxel (Eq. 10)
        att = global_softmax(att, cross_edge_voxel_index)
        out_att_soft = att.view(-1, 1)

        # 5. Eq. 10: h_i <- h_i + gamma_i * sum_k softmax(u_ik) h_k
        weighted_program_feature = out_att_soft * condition[cross_edge_program_index]

        new_voxel_feature = voxel_feature + out_mask_soft * scatter(
            weighted_program_feature, 
            cross_edge_voxel_index, 
            dim=0, 
            dim_size=voxel_feature.shape[0], 
            reduce="sum"
        )
        
        return new_voxel_feature


# ==== MODULE: Denoising GNN backbone -- eps_theta(x_t, t, c) (Table A3: "Denoising GNN"; Section 3.2.2) ====
class VoxelGNN_G(nn.Module):
    def __init__(self, voxel_input_dim, num_classes, hidden_dim, noise_dim, layer_num):
        """
        Backbone for denoising GNN
        voxel_input_dim: same size as v_feat
        num_classes: C
        """
        super(VoxelGNN_G, self).__init__()
        self.layer_num = layer_num
        # ---- Input encoder (Eq. 6 terms) ----
        self.register_buffer('pe', position_encoder(hidden_dim, 20))   # PE(l): floor positional encoding (Eq. 8)
        self.xyz_emb = nn.Linear(3, hidden_dim)          # E_p: coordinate embedding
        self.v_feat_enc = MLP([voxel_input_dim - 3, hidden_dim]) # E_v: voxel attribute encoder (-3 for pos)
        self.v_label_enc = MLP([1 + num_classes, hidden_dim])    # E_x: noisy-label encoder
        self.time_enc = nn.Sequential(                   # E_t: timestep encoder MLP w/ SiLU (Eq. 7)
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.z_enc = nn.Linear(noise_dim, hidden_dim)    # E_eta: aux-noise encoder

        self.blocks = nn.ModuleList([                    # Voxel block x12 (Eq. 12-13)
            VoxelBlock(hidden_dim, if_agg_story_in_update=True)
            for _ in range(layer_num)
        ])

        self.attention = Attention(hidden_dim)   # Cross-modal attention (Eq. 9-10)

        self.output_mlp = nn.Sequential(         # "Output head" / Output MLP -> predicted noise
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1 + num_classes)
        )

        self.global_pool_mixer = nn.Sequential(  # "Global context mixer" / Mixer MLP (Eq. 11)
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, v_feat, v_noisy_label, t_emb, cond, z, 
                  e, cep, cev, vbi, vfc, vfb):

        # Eq. 6: h^(0) = E_v(v) + E_x(x_t) + E_t(t) + E_eta(eta) + E_p(p) + PE(l)
        pos, v_feat = extract_pos(v_feat)
        v_feat_emb = self.v_feat_enc(v_feat)          # E_v(v)
        v_label_emb = self.v_label_enc(v_noisy_label) # E_x(x_t)
        v_time_emb = self.time_enc(t_emb)[vbi]        # E_t(t), computed per graph then broadcast to voxels
        v_z_emb = self.z_enc(z)                       # E_eta(eta)
        v = v_feat_emb + v_label_emb + v_time_emb + v_z_emb + self.pe[get_voxel_floor_level(vfc, vbi), :] + self.xyz_emb(pos)

        #  main loop of GNN blocks: cross-attn -> global mixer -> voxel block, x12 (Section 3.2.2)
        for i, block in enumerate(self.blocks):

            # Cross-modal attention (Eq. 9-10)
            if cond is not None:
                v = self.attention(
                    voxel_feature=v,
                    condition=cond,
                    cross_edge_program_index=cep,
                    cross_edge_voxel_index=cev
                )

            # Eq. 11: global context mixer  h_i <- h_i + MLP_global(mean_j h_j)
            global_feat = global_mean_pool(v, vfb)
            v = v + self.global_pool_mixer(global_feat[vfb])

            # Voxel block (Eq. 12-13) + residual connection
            v_in = v
            v = block(voxel_feature=v, edge_index=e, position=pos, floor_index=vfc, batch_index=vfb)
            v = v + v_in

        # Output head: eps_theta(x_t, t, c)
        predicted_noise = self.output_mlp(v)
        return predicted_noise


# ==== MODULE: Diffusion network -- forward process (Section 3.2.1; Eq. 1-3) ====
class GaussianDiffusion(nn.Module):
    """
    Gaussian diffusion core: precomputes forward (q) and reverse (posterior) coefficients,
    exposes q_sample for training and a single-step ddpm_step for sampling.
    Expects shape (N, D); schedule tensors are broadcast over all dims except the last.

    ----
    T : int 
        timestep 1000
    schedule : str 
        'linear' or 'cosine'. cosine for sable
    beta_start, beta_end : float
        end for 'linear', 'cosine' will ignore these.
    eps : float
        small constant for numerical stability.
    """
    def __init__(
        self,
        T: int = 1000,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        eps: float = 1e-8,
    ):
        super().__init__()
        assert T >= 2
        self.T = int(T)
        self.eps = float(eps)

        # ---------- forward schedule: q(x_t|x_{t-1}) = N(sqrt(1-beta_t) x_{t-1}, beta_t I) (Eq. 2) ----------
        # alpha_t = 1 - beta_t ,  alpha_bar_t = prod_s alpha_s   (Eq. 3 coefficients)
        if schedule.lower() == "linear":
            beta = torch.linspace(beta_start, beta_end, T)         # (T,)
            alpha = 1.0 - beta                                     # (T,)
            alpha_bar = torch.cumprod(alpha, dim=0)                # (T,)
        elif schedule.lower() == "cosine":
            s = 0.008
            steps = torch.arange(T + 1, dtype=torch.float64)
            f = torch.cos(((steps / T + s) / (1 + s)) * math.pi * 0.5) ** 2
            alpha_bar_f64 = f / f[0]
            alpha_bar = alpha_bar_f64[:-1].to(torch.float32)       # (T,)
            alpha_bar_prev = alpha_bar_f64[1:].to(torch.float32)   # (T,)
            alpha = (alpha_bar / torch.clamp(alpha_bar_prev, min=self.eps)).clamp(max=1.0)
            beta = (1.0 - alpha).clamp(min=1e-8, max=0.999)
        else:
            raise ValueError("schedule must be 'linear' or 'cosine'")

        alpha = alpha.to(torch.float32)
        alpha_bar = torch.cumprod(alpha, dim=0) if schedule.lower()=="linear" else alpha_bar

        # ---------- buffers ----------
        self.register_buffer("beta", beta)                       # (T,)
        self.register_buffer("alpha", alpha)                     # (T,)
        self.register_buffer("alpha_bar", alpha_bar)             # (T,)
        self.register_buffer("alphas_cumprod", alpha_bar)        # (T,)
        self.register_buffer("sqrt_alpha", torch.sqrt(alpha))    # (T,)

        # clamp sqrt_alpha_bar to prevent numerical overflow
        ab_clamped = torch.clamp(alpha_bar, min=self.eps, max=1.0 - self.eps)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(ab_clamped))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(torch.clamp(1.0 - ab_clamped, min=0.0)))

        # ᾱ_{t-1}, ᾱ_{-1} ≡ 1
        alpha_bar_prev = torch.cat(
            [torch.ones(1, dtype=alpha_bar.dtype, device=alpha_bar.device), alpha_bar[:-1]], dim=0
        )
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)     # (T,)

        # posterior (q(x_{t-1}|x_t, x_0))
        posterior_var = beta * (1.0 - alpha_bar_prev) / torch.clamp((1.0 - alpha_bar), min=self.eps)
        posterior_var = torch.clamp(posterior_var, min=1e-20)
        self.register_buffer("posterior_var", posterior_var)       # (T,)
        self.register_buffer("posterior_log_var_clipped", torch.log(posterior_var + 1e-20))

        c1 = (torch.sqrt(alpha_bar_prev) * beta) / torch.clamp((1.0 - alpha_bar), min=self.eps)
        c2 = (torch.sqrt(alpha) * (1.0 - alpha_bar_prev)) / torch.clamp((1.0 - alpha_bar), min=self.eps)
        self.register_buffer("posterior_mean_coef1", c1)           # (T,)
        self.register_buffer("posterior_mean_coef2", c2)           # (T,)

    # -------------------- helpers --------------------
    @staticmethod
    def _expand_like(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Expand a time-indexed coefficient v[t] of shape (T,) to be broadcast-compatible with x.
        Supports x of shape (N, D) or higher-dimensional tensors (broadcast over all dims except the last).
        """
        while v.dim() < x.dim():
            v = v.view(*v.shape, *([1] * (x.dim() - v.dim())))
        return v

    # -------------------- training (forward) --------------------
    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        Eq. 3: forward diffusion closed form  x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε
        x0: (N, D)
        t : (N,)  timestep for each sample in the batch
        return: x_t, ε
        """
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self._expand_like(self.alpha_bar.index_select(0, t), x0)          # (N, 1, ...)->(N,D,...)
        ab = torch.clamp(ab, min=self.eps, max=1.0 - self.eps)  # same as buffers
        x_t = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise
        return x_t, noise



class DiffusionGenerator(nn.Module):
    def __init__(self,  
             program_input_dim,
             voxel_input_dim,
             num_classes,
             hid,
             program_layer,
             voxel_layer,
             T=1000,
             beta_start=1e-4,
             beta_end=0.02,
             schedule='linear',
             text_encoder=None,
             cfg_scale=5.0,
             class_weights=None,
             mask_weights=None):
        super().__init__()

        self.hid = int(hid)
        self.num_classes = int(num_classes)
        self.noise_dim = 2 * hid
        self.cfg_scale = cfg_scale
        program_input_dim = int(program_input_dim)
        voxel_input_dim = int(voxel_input_dim)
        program_layer = int(program_layer)
        voxel_layer = int(voxel_layer)
        T = int(T)

        # ------- module -------
        self.programGNN = ProgramGNN(program_input_dim, hid, noise_dim=0, layer_num=program_layer)   # MODULE: Program GNN (Section 3.2.3 (1))
        self.model = VoxelGNN_G(                       # MODULE: Denoising GNN backbone (Section 3.2.2)
            voxel_input_dim=voxel_input_dim,
            num_classes=self.num_classes,
            hidden_dim=hid,
            noise_dim=self.noise_dim,
            layer_num=voxel_layer
        )
        self.text_encoder = text_encoder              # MODULE: Text encoder (frozen T5), see t2p.py
        self.t2p = Text2Program(hid, num_classes)     # MODULE: Text2Program (Eq. 15), see t2p.py
        self.label_emb = LabelEmbedding(num_classes, hid)

        # ------- diffusion -------
        self.diff    = GaussianDiffusion(T=T, schedule=schedule, beta_start=beta_start, beta_end=beta_end)   # MODULE: Diffusion network (Eq. 1-3)
        self.time_embed = nn.Sequential(nn.Linear(1, hid), nn.SiLU(), nn.Linear(hid, hid))   # E_t: timestep encoder (Eq. 7)

        # ------- gating fusion -------
        self.gate_prog_fuse = nn.Sequential(          # MODULE: Hybrid conditioning "Gate MLP" (Eq. 17)
            nn.Linear(hid * 2, hid),
            nn.SiLU(),
            nn.Linear(hid, hid),
            nn.Sigmoid()
        )
        self.cond_norm = nn.LayerNorm(hid)            # condition normalization for CFG (Section 3.2.4)
        self.x0_activation = nn.Tanh()

        # Eq. 16: FAR/TPR embeddings E_FAR, E_TPR added to pseudo-program nodes (text mode)
        self.far_emb_text = nn.Linear(1, hid)
        self.tpr_emb_text = nn.Linear(num_classes, hid)

        # ------- DPM-Solver++ for inference -------
        sampler_config = {
            "beta_start": beta_start,
            "beta_end": beta_end,
            "beta_schedule": schedule,
            "num_train_timesteps": T,
            "prediction_type": "epsilon",
            "thresholding": False, 
        }
        self.sampler_scheduler = DPMSolverMultistepScheduler.from_config(sampler_config)

        if class_weights is None:
            class_weights = torch.ones(num_classes)
        if mask_weights is None:
            mask_weights = torch.tensor([1.0, 1.0])
        self.register_buffer('class_weights', class_weights)
        self.register_buffer('mask_weights', mask_weights)

    def compute_tv_loss(self, pred_prob, batch):
        """
        Eq. 22: TV loss  L_TV = E[ || p_i^class - p_j^class ||_1 ] over neighbor voxels.
        Adjacent voxels in a building (especially horizontally) are likely the same class.
        """
        # 1. use voxel_edge to compute differences between adjacent nodes
        row, col = batch.voxel_edge
        
        # pred_prob: (Nv, Num_Classes) post-softmax probabilities
        # compute L1 distance between adjacent voxel distributions
        diff = (pred_prob[row] - pred_prob[col]).abs().sum(dim=-1)
        
        # 2. penalize all edges (could be restricted to same-story edges only)
        tv_loss = diff.mean()
        return tv_loss

    def forward_train(self, graph, prompts_per_graph=None, force_mode=None, p_uncond=0.1):
        # 1) random modality selection among the 3 conditioning modes (Section 3.2.4)
        if force_mode is not None:
            mode = force_mode
        else:
            mode_idx = torch.randint(0, 3, (1,), device=graph.voxel_feature.device).item()
            mode = ("both", "program", "text")[mode_idx]

        device = graph.voxel_feature.device
        cond_feat, cep_idx, cev_idx, _ = self._build_prog_cond(
            graph, device, prompts_per_graph, mode
        )
        # 2) condition dropout -> zero features for CFG (Section 3.2.4)
        if torch.rand((), device=cond_feat.device) < p_uncond:
            cond_feat = torch.zeros_like(cond_feat)
        cond_feat_norm = self.cond_norm(cond_feat)

        # 3) get ground-truth x_0 (N_voxels, 1+C)
        x_0 = _get_ground_truth_x0(graph, self.num_classes)
        N_voxels = x_0.shape[0]
        num_graphs = graph.num_graphs
        t_int = torch.randint(0, self.diff.T, (num_graphs,), device=device).long()
        t_emb_input = t_int.float().unsqueeze(-1) # (B, 1)

        # 4) sample noise epsilon and build x_t via Eq. 3 (q_sample)
        noise = torch.randn_like(x_0)
        t_int_per_voxel = t_int[graph.voxel_feature_batch]
        q_sample_output = self.diff.q_sample(x_0, t_int_per_voxel, noise)
        if isinstance(q_sample_output, tuple):
            x_t = q_sample_output[0]
        else:
            x_t = q_sample_output

        z = torch.randn((N_voxels, self.noise_dim), device=device)
        t_emb = self.time_embed(t_emb_input) # (B, 1) -> (B, hid)

        predicted_noise = self.model(
            v_feat=graph.voxel_feature,
            v_noisy_label=x_t,
            t_emb=t_emb,
            cond=cond_feat_norm,
            z=z,
            e=graph.voxel_edge,
            cep=cep_idx,
            cev=cev_idx,
            vbi=graph.voxel_feature_batch,
            vfc=graph.voxel_floor_cluster,
            vfb=graph.voxel_feature_batch
        )
        # ---- Eq. 20: DDPM loss (weighted MSE, mask channel + C class channels) ----
        loss_per_element = F.mse_loss(predicted_noise, noise, reduction='none')
        gt_labels = graph.voxel_label
        mask_w_map = self.mask_weights[(gt_labels != -1).long()]

        safe_labels = gt_labels.clone()
        safe_labels[safe_labels == -1] = 0
        class_w_map = self.class_weights[safe_labels.long()]

        loss_mask_val = loss_per_element[:, 0] * mask_w_map        # w_mask term (Eq. 20)
        loss_labels_val = loss_per_element[:, 1:] * class_w_map.unsqueeze(-1)   # w_class term (Eq. 20)
        loss_ddpm = loss_mask_val.mean() + loss_labels_val.mean()

        # 2. Eq. 4: recover x_0 estimate  x0_hat = (x_t - sqrt(1-a_bar) eps_hat) / sqrt(a_bar)
        alpha_bar = self.diff.alphas_cumprod.to(device)[t_int_per_voxel]
        sqrt_alpha_bar = torch.sqrt(alpha_bar).unsqueeze(-1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - alpha_bar).unsqueeze(-1)

        should_apply_aux_loss = (t_int < 400).float().unsqueeze(-1) # (Batch, 1)
        pred_x0 = (x_t - sqrt_one_minus_alpha_bar * predicted_noise) / (sqrt_alpha_bar + 1e-6)
        pred_x0 = torch.clamp(pred_x0, -2.0, 2.0)
        
        
        pred_mask_logits = pred_x0[:, 0:1] # Mask
        pred_class_logits = pred_x0[:, 1:] # Classes
        
        pred_mask_prob = torch.sigmoid(pred_mask_logits) # (Nv, 1)
        pred_class_prob = torch.softmax(pred_class_logits, dim=-1) # (Nv, C)
        area_index = 6  # index of area in voxel_feature
        voxel_area = graph.voxel_feature[:, area_index].view(-1, 1) # (Nv, 1)

        # global FAR_hat / TPR_hat from x0_hat (inputs to Eq. 21)
        weighted_area = pred_mask_prob * pred_class_prob * voxel_area # (Nv, C)
        vfb = graph.voxel_feature_batch
        graph_class_area = scatter(weighted_area, vfb, dim=0, reduce="sum") # (G, C)
        graph_total_area = scatter(pred_mask_prob * voxel_area, vfb, dim=0, reduce="sum") # (G, 1)
        pred_FAR = graph_total_area.squeeze(-1)
        pred_TPR = graph_class_area / (graph_total_area + 1e-6)



        target_TPR = graph.program_target_ratio # (G, C)
        target_FAR = getattr(graph, 'far', getattr(graph, 'FAR', None)) # (G,)
        ptr_batch = graph.program_target_ratio_batch 
        denom = scatter_add(target_TPR, ptr_batch, dim=0)[ptr_batch]
        target_TPR_normalized = target_TPR / (denom + 1e-8)

        if target_TPR_normalized.dim() == 1:
             target_TPR_normalized = target_TPR_normalized.view(graph.num_graphs, self.num_classes)

        normalized_t = t_int.float() / self.diff.T

        loss_tv = self.compute_tv_loss(pred_class_prob, graph)   # Eq. 22: TV loss
        tv_weight = 0.1 * (1.0 - normalized_t).pow(2)

        # Eq. 21: L_TPR = E[SmoothL1(TPR_hat, TPR)]
        tpr_errors = F.smooth_l1_loss(pred_TPR, target_TPR_normalized, reduction='none')
        weighted_tpr_errors = tpr_errors * self.class_weights.view(1, -1)
        loss_tpr_per_graph = weighted_tpr_errors.mean(dim=1)
        loss_tpr = (loss_tpr_per_graph * should_apply_aux_loss.squeeze()).mean()

        # Eq. 21: L_FAR = E[SmoothL1(FAR_hat, FAR)]
        loss_far_val = F.smooth_l1_loss(pred_FAR, target_FAR, reduction='none')
        loss_far = (loss_far_val * should_apply_aux_loss.squeeze()).mean()

        # Eq. 23: L_total = lambda_DDPM L_DDPM + lambda_TPR L_TPR + lambda_FAR L_FAR + lambda_TV L_TV
        loss_tv_weighted = (loss_tv * tv_weight).mean()
        total_loss = loss_ddpm + 0.2 * loss_tpr + 0.1 * loss_far + loss_tv_weighted

        loss_dict = {
            "ddpm":    loss_ddpm.detach(),
            "tpr":     loss_tpr.detach(),
            "far":     loss_far.detach(),
            "tv_weighted": loss_tv_weighted.detach(),
            "total":   total_loss.detach(),
        }
        return total_loss, loss_dict

    def sample(self, 
            graph, 
            prompts_per_graph=None,
            mode="text",
            steps=50, 
            eta=0.0, 
            guidance_decay=0.5):
        
        assert mode in ("program", "text", "both"), f"Unknown mode: {mode}"

        self.sampler_scheduler.set_timesteps(steps)
        device = graph.voxel_feature.device
        N_voxels = graph.voxel_feature.shape[0]

        z = torch.randn((N_voxels, self.noise_dim), device=device)
        x_t = torch.randn((N_voxels, 1 + self.num_classes), device=device)

        with torch.no_grad():
            # 1. precompute conditional and unconditional features
            cond_feat, cep_idx, cev_idx, _ = self._build_prog_cond(graph, device, prompts_per_graph, mode)
            cond_feat = self.cond_norm(cond_feat)
            uncond = torch.zeros_like(cond_feat)
            uncond = self.cond_norm(uncond)

            # 2. reverse process: iterative denoising, M=50 steps (Section 3.2.1; Eq. 5 via DPM-Solver++)
            for t_int in self.sampler_scheduler.timesteps:  # t: T -> 0
                num_graphs = graph.num_graphs
                t_batch = torch.full((num_graphs,), t_int.item(), device=device, dtype=torch.long)
                t_emb_input = t_batch.float().unsqueeze(-1) # (B, 1)
                t_emb = self.time_embed(t_emb_input) # (B, hid)
                
                # --- predict unconditional noise ---
                noise_uncond = self.model(
                    v_feat=graph.voxel_feature, v_noisy_label=x_t, t_emb=t_emb, cond=uncond, z=z,
                    e=graph.voxel_edge, cep=cep_idx,
                    cev=cev_idx, vbi=graph.voxel_feature_batch, 
                    vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch
                )
                
                # --- predict conditional noise ---
                noise_cond = self.model(
                    v_feat=graph.voxel_feature, v_noisy_label=x_t, t_emb=t_emb, cond=cond_feat, z=z,
                    e=graph.voxel_edge, cep=cep_idx,
                    cev=cev_idx, vbi=graph.voxel_feature_batch, 
                    vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch
                )
                
                # --- Eq. 19: classifier-free guidance  eps_tilde = eps_uncond + s (eps_cond - eps_uncond) ---
                predicted_noise = noise_uncond + self.cfg_scale * (noise_cond - noise_uncond)
                # predicted_noise = noise_cond
                # Eq. 4 (x0 estimate) + Eq. 5 (DPM-Solver++ update step)
                step_output = self.sampler_scheduler.step(predicted_noise, t_int, x_t)
                x_t = step_output.prev_sample

        # denoising loop complete; x_t is now x_0
        pred_soft_x0 = x_t
        pred_mask_logits = x_t[:, 0:1]
        pred_class_logits = x_t[:, 1:]
        mask_prob = torch.sigmoid(pred_mask_logits) # (Nv, 1) Range [0, 1]
        class_prob = torch.softmax(pred_class_logits, dim=-1) # (Nv, C) Sum=1
        voxel_is_occupied = (mask_prob.squeeze(-1) > 0.5)
        voxel_class_indices = class_prob.argmax(dim=-1)
        pred_hard_labels_raw = torch.full((N_voxels,), -1, device=device, dtype=torch.long)
        pred_hard_labels_raw[voxel_is_occupied] = voxel_class_indices[voxel_is_occupied]

        # neighbor-voting smoothing of discrete labels (Section 3.2.1, end)
        pred_hard_labels = self.refine_prediction(graph, pred_hard_labels_raw)

        out = {
            "pred_soft_x0": pred_soft_x0,  # (Nv, 1+C) soft output, used for PCA
            "pred_hard_labels": pred_hard_labels,  # (Nv,) hard output, used for TPR/FAR and saving
            "pred_hard_labels_raw": pred_hard_labels_raw,  # (Nv,) hard output BEFORE neighbor-vote smoothing, for the with/without-smoothing ablation
            }
        return out

    def _text_program_kv(self, graph, globals_v):
        """
        Return text vectors aligned one-to-one with real program nodes:
          program_text_kv: (P_real, H)
          txt_global_g   : (G, H)
        """
        vfb = graph.voxel_feature_batch.long()
        G = int(vfb.max().item()) + 1 if vfb.numel() > 0 else 0

        # (Nv,H) -> (G,H)
        glob_sum = scatter_add(globals_v, vfb, dim=0, dim_size=G)
        cnt      = scatter_add(torch.ones_like(vfb, dtype=glob_sum.dtype).unsqueeze(-1), vfb, dim=0, dim_size=G)
        txt_global_g = glob_sum / cnt.clamp_min(1.0)                 # (G,H)

        # text -> per-graph per-class vectors (additive bias over class embeddings)
        text_gc = self.t2p(txt_global_g)                             # (G,C,H)
        # index by real program node order (g,c)
        p_batch = graph.program_class_feature_batch.long()           # (P_real,)
        p_cls   = graph.program_class_feature.argmax(dim=-1).long()  # (P_real,)
        program_text_kv = text_gc[p_batch, p_cls]                    # (P_real,H)
        return program_text_kv, txt_global_g

    @torch.no_grad()
    def _encode_text(self, prompts_per_graph, device, graph):
        # text encoding: returns a global vector per voxel
        if prompts_per_graph is None:
            print("no prompt")
            return None
        vfb = graph.voxel_feature_batch
        G = int(vfb.max().item()) + 1 if vfb.numel() > 0 else 0

        if isinstance(prompts_per_graph, str):
            prompts = [prompts_per_graph] * G
        elif isinstance(prompts_per_graph, (list, tuple)):
            prompts = list(prompts_per_graph)
            if len(prompts) == 1 and G > 1:
                prompts = prompts * G
            elif len(prompts) < G:
                prompts = prompts + [""] * (G - len(prompts))
            elif len(prompts) > G:
                prompts = prompts[:G]
        else:
            return None

        if G == 0:
            return None

        globals_g = self.text_encoder(prompts, device)  # (G,H)
        globals_v = globals_g[vfb] if vfb.numel() > 0 else globals_g.new_zeros(0, globals_g.size(-1))
        return globals_v

    def _build_prog_cond(self, graph, device, prompts_per_graph, mode):
        # MODULE: Multi-modal conditioning -- builds program-node features h_k for cross-attn (Section 3.2.3)
        # broadcast graph-level FAR to each program node
        far_val = getattr(graph, 'far', getattr(graph, 'FAR', None))
        if far_val is not None:
            far_per_node = far_val[graph.program_class_feature_batch.long()]
        else:
            far_per_node = None

        prog_h_real = self.programGNN(
            cf=graph.program_class_feature,
            slf=graph.story_level_feature,
            e=graph.program_edge,
            cc=graph.program_class_cluster,
            tr=graph.program_target_ratio,
            cfb=graph.program_class_feature_batch,
            z=None,
            far=far_per_node
        )
        if not torch.all(torch.isfinite(prog_h_real)):
            print("!!! programGNN output is NOT finite !!!")
            print(prog_h_real)

        if mode == "program":
            # (1) Program-only condition (Section 3.2.3 (1); Eq. 14 inside programGNN)
            prog_h_real_norm = F.layer_norm(prog_h_real, (self.hid,))
            return prog_h_real_norm, graph.cross_edge_program_index_select, \
                   graph.cross_edge_voxel_index_select, None

        elif mode == "text":
            # (2) Text-only condition (Section 3.2.3 (2))
            globals_v = self._encode_text(prompts_per_graph, device, graph)   # global text vector tau
            fake_prog, _, fake_cep, fake_cev, _ = make_pseudo_program_from_text(
                graph, globals_v, self.num_classes, self.t2p, device
            )   # Eq. 15: pseudo-program nodes h_k = e_c(k) + MLP_text(tau)
            G = graph.num_graphs
            C = self.num_classes
            if fake_prog.shape[0] > 0:
                # Eq. 16: h_k += E_FAR(FAR) + E_TPR(TPR)
                far_val = getattr(graph, 'far', getattr(graph, 'FAR', None))
                far_h = self.far_emb_text(
                    far_val.repeat_interleave(C).view(-1, 1).to(device)
                )
                tpr_val = graph.program_target_ratio.view(G, C).to(device)
                tpr_h = self.tpr_emb_text(tpr_val).repeat_interleave(C, dim=0)
                fake_prog = fake_prog + far_h + tpr_h
            fake_prog_norm = F.layer_norm(fake_prog, (self.hid,))
            return fake_prog_norm, fake_cep, fake_cev, None

        elif mode == "both":
            # (3) Program-text hybrid condition (Section 3.2.3 (3))
            globals_v = self._encode_text(prompts_per_graph, device, graph)
            program_text_kv, _ = self._text_program_kv(graph, globals_v)   # h_k^text = h_hat_c(k)
            prog_h_real_norm = F.layer_norm(prog_h_real, (self.hid,))          # LN(h_k^prog)
            program_text_kv_norm = F.layer_norm(program_text_kv, (self.hid,))  # LN(h_k^text)
            # Eq. 17: g_k = sigma(MLP_gate([LN(h_k^prog), LN(h_k^text)]))
            gate = self.gate_prog_fuse(torch.cat([prog_h_real_norm, program_text_kv_norm], dim=-1))
            # Eq. 18: h_k = g_k * LN(h_k^prog) + (1 - g_k) * LN(h_k^text)
            mixed_feat = gate * prog_h_real_norm + (1.0 - gate) * program_text_kv_norm


            return mixed_feat, graph.cross_edge_program_index_select, \
                   graph.cross_edge_voxel_index_select, None
        else:
            raise ValueError(mode)

    def refine_prediction(self, graph, pred_hard_labels):
        # neighbor-voting smoothing to suppress isolated single-voxel errors (Section 3.2.1)
        # 1. convert to one-hot (Nv, C+1)
        # class 0 = Empty (shifted from -1), 1..C = functional classes
        # input range: [-1, 5] -> shifted range: [0, 6]
        # class 0 = Empty (Air)
        shifted_labels = pred_hard_labels + 1
        num_classes_shifted = self.num_classes + 1
        one_hot = F.one_hot(shifted_labels, num_classes=num_classes_shifted).float()
        
        # 2. get edge indices
        row, col = graph.voxel_edge
        
        # 3. aggregate neighbor votes
        neighbor_sum = scatter(one_hot[col], row, dim=0, reduce="sum", dim_size=one_hot.shape[0])
        neighbor_sum[:, 0] *= 0.05  # decrease the weight of "Empty" neighbors!
        
        # 4. add self-vote with higher confidence
        total_vote = neighbor_sum + one_hot * 1.5

        # 5. take the argmax
        new_labels = total_vote.argmax(dim=-1) - 1
        
        return new_labels

