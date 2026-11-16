import torch, torch.nn as nn
from transformers import T5EncoderModel, AutoTokenizer
from torch_scatter import scatter_add


# ==== MODULE: Text encoder (Table A3: "Text conditioning" -> "Text encoder"); Section 3.2.3 (2) ====
class T5TextEncoder(nn.Module):
    """
    output:
      globals: Tensor(G, hid)  -- global text vector tau: mean-pooled + linearly projected T5 output per graph
    """
    def __init__(self, hid, model_name='t5-base', max_len=512, freeze=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder   = T5EncoderModel.from_pretrained(model_name)   # "T5-base encoder (frozen)"
        self.max_len   = max_len
        self.proj_g    = nn.Linear(self.encoder.config.d_model, hid)  # "Global text projection"
        if freeze:
            for p in self.encoder.parameters(): p.requires_grad = False

    @torch.no_grad()
    def encode_one(self, prompt: str, device):
        if prompt is None: prompt = ""
        toks = self.tokenizer(prompt, truncation=True, max_length=self.max_len,
                              padding=False, return_tensors='pt')
        toks = {k: v.to(device) for k, v in toks.items()}
        out = self.encoder(**toks).last_hidden_state.squeeze(0)  # (L, d_model)
        return out

    @torch.no_grad()
    def forward(self, prompts_per_graph, device):
        globals_list = []
        for p in prompts_per_graph:
            h = self.encode_one(p, device)                  # (L, d_model)
            globals_list.append(self.proj_g(h.mean(0)))     # tau = proj(mean-pool last hidden states)
        globals = torch.stack(globals_list, dim=0)          # (G, hid)
        return globals

class LabelEmbedding(nn.Module):
    def __init__(self, num_classes, d_embed):
        super().__init__()
        self.proj = nn.Linear(num_classes, d_embed, bias=False)
    def forward(self, y_onehot):
        return self.proj(y_onehot)

# ==== MODULE: Text2Program (Table A3: "Text conditioning" -> "Text2Program"); Eq. 15 ====
class Text2Program(nn.Module):
    """
    Eq. 15: h_hat_k = e_c(k) + MLP_text(tau)   (additive bias over C learnable class embeddings)
    """
    def __init__(self, hid, num_classes):
        super().__init__()
        self.class_emb = nn.Parameter(torch.randn(num_classes, hid) * 0.02)   # e_c: "Class embedding" Parameter[C, H]
        self.gate = nn.Sequential(                                            # MLP_text: "Text-bias MLP" (SiLU)
            nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, hid)
        )

    def forward(self, txt_global_g):
        # txt_global_g : (G, H) = tau
        G = txt_global_g.size(0)
        if G == 0:
            C, H = self.class_emb.shape
            return torch.zeros(0, C, H, device=txt_global_g.device,
                               dtype=txt_global_g.dtype)

        bias = self.gate(txt_global_g)                        # MLP_text(tau), (G, H)
        return self.class_emb.unsqueeze(0) + bias[:, None, :] # Eq. 15: e_c + MLP_text(tau) -> (G, C, H)


# Builds the C pseudo-program nodes + fully-connected voxel<->pseudo cross-edges (Section 3.2.3 (2))
def make_pseudo_program_from_text(graph, globals_v, num_classes, t2p, device):
    """
    return:
      - prog_hat_flat: (G*C, H)
      - prog_cls_hat:  (G*C, C)
      - cep_hat, cev_hat: (Nv*C,)  LongTensor
      - p_batch_hat: (G*C,)
    """
    vfb = graph.voxel_feature_batch.to(device).long()          # (Nv,)
    Nv  = vfb.numel()
    G   = int(vfb.max().item()) + 1 if Nv > 0 else 0

    # ----- batch is None -----
    if G == 0:
        if globals_v is not None:
            H = globals_v.size(-1)
            dev = globals_v.device
        else:
            H = getattr(t2p, "hid", None) or getattr(t2p, "class_emb").size(-1)
            dev = device
        zH = torch.zeros(0, H, device=dev)
        zL = torch.zeros(0, dtype=torch.long, device=dev)
        zC = torch.zeros(0, num_classes, device=dev)
        return zH, zC, zL, zL, zL

    # ----- graph-level text global vector (aggregated from per-voxel globals_v) -----
    assert globals_v is not None, "text-only mode requires globals_v=(Nv,H)"
    globals_v = globals_v.to(device)
    H = globals_v.size(-1)

    # sum & count (count must be float / same dtype as sum for division)
    glob_sum = scatter_add(globals_v, vfb, dim=0, dim_size=G)          # (G,H)
    ones_nv  = torch.ones(Nv, 1, device=device, dtype=glob_sum.dtype)  # (Nv,1)
    cnt      = scatter_add(ones_nv, vfb, dim=0, dim_size=G)            # (G,1)
    txt_global_g = glob_sum / cnt.clamp_min(1.0)                       # (G,H)

    # ----- Eq. 15: Text -> Program pseudo nodes h_hat, (G,C,H) -----
    prog_hat = t2p(txt_global_g)                                       # expected shape (G,C,H)
    assert prog_hat.dim() == 3 and prog_hat.size(0) == G and prog_hat.size(1) == num_classes
    prog_hat_flat = prog_hat.reshape(G * num_classes, H)               # (G*C, H)

    # ----- batch index vector for pseudo programs (G*C,) -----
    p_batch_hat = torch.arange(G, device=device).repeat_interleave(num_classes)  # (G*C,)

    # ----- class one-hot: serves as program_class_feature directly -----
    prog_cls_hat = torch.eye(num_classes, device=device).repeat(G, 1)  # (G*C, C)

    # ----- pseudo cross-edges (fully connected within graph: each voxel -> C pseudo programs) -----
    # for voxel i in graph g=vfb[i], the global program offset for that graph is g*C.
    base = vfb * num_classes                                           # (Nv,)
    cep_hat = base.view(-1, 1) + torch.arange(num_classes, device=device).view(1, -1)
    cep_hat = cep_hat.reshape(-1).long()                                # (Nv*C,)

    # cev_hat: (Nv*C,) = [0..Nv-1] each repeated C times
    cev_hat = torch.arange(Nv, device=device).view(-1, 1).repeat(1, num_classes).reshape(-1).long()

    return prog_hat_flat, prog_cls_hat, cep_hat, cev_hat, p_batch_hat
