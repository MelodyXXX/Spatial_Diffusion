import torch
import os
import json
from utils.data.GraphConstructor import GraphConstructor
from torch_geometric.data import Batch
from utils.util_graph import get_program_ratio, data_parallel, rebatch_for_multi_gpu
import torch.nn.functional as F
from contextlib import contextmanager
import math
import numpy as np
from torch_geometric.nn import global_mean_pool
from utils.eval.metrics import calculate_2d_metrics, calculate_fid, calculate_physical_metrics



def save_output(batch_size, batch, class_weights, program_weights, FAR, max_out_program_index, out, follow_batch, output_dir, new_data_id_strs=None):
    """
    Save the evaluation results
    """

    if not os.path.exists(os.path.join(output_dir, "global_graph_data")):
        os.mkdir(os.path.join(output_dir, "global_graph_data"))
        os.mkdir(os.path.join(output_dir, "local_graph_data"))
        os.mkdir(os.path.join(output_dir, "voxel_data"))

    num_of_program_node_accum = 0
    batch_all_g = []
    data = rebatch_for_multi_gpu(batch, list(range(batch_size)), follow_batch, out, class_weights, program_weights, FAR, max_out_program_index)

    """
    --- data ---
    g: graph
    o: voxel label (n[type])
    cw: (n[new_proportion] in global graph) -- program class ratio/weight 
    pw: (n[region_far] in local graph)
    far: (g[far] in global graph)
    pid: the selected program node id for each voxel node
    """

    for i, (g, o, cw, pw, far, pid) in enumerate(data):
        data_id_str = g["data_id_str"][0]
        new_data_id_str = g["data_id_str"][0] if new_data_id_strs is None else str(new_data_id_strs[i]).zfill(GraphConstructor.data_id_length)
        o = o.cpu().data.numpy().tolist()
        cw, pw, far = cw.cpu().data.numpy(), pw.cpu().data.numpy(), far.item()

        # Modify Global data
        global_graph = json.loads(g["raw_global_graph_json"][0])
        global_graph["new_far"] = far
        for n in global_graph["global_node"]:
            n["new_proportion"] = float(cw[n['type']])
        with open(os.path.join(output_dir, "global_graph_data", GraphConstructor.global_graph_prefix + new_data_id_str + ".json"), 'w') as f:
            json.dump(global_graph, f)

        # Modify Local data
        d = {}  # program id to its type and type id
        local_graph = json.loads(g["raw_local_graph_json"][0])
        for i, (n, c) in enumerate(zip(local_graph["node"], pw)):
            n["region_far"] = float(c)
            d[i] = [n["type"], n["type_id"]]
        with open(os.path.join(output_dir, "local_graph_data", GraphConstructor.local_graph_prefix + new_data_id_str + ".json"), 'w') as f:
            json.dump(local_graph, f)

        # Modify Voxel data
        voxel_graph = json.loads(g["raw_voxel_graph_json"][0])
        for n, label, _pid in zip(voxel_graph["voxel_node"], o, pid):
            query = d[_pid.item() - num_of_program_node_accum]
            n["type"] = query[0] if 1.0 in label else -1  # # label.index(1.0)
            n["type_id"] = query[1] if 1.0 in label else 0
        num_of_program_node_accum += pw.shape[0]
        with open(os.path.join(output_dir, "voxel_data", GraphConstructor.voxel_graph_prefix + new_data_id_str + ".json"),'w') as f:
            json.dump(voxel_graph, f)

        all_graphs = [global_graph, local_graph, voxel_graph]
        batch_all_g.append(all_graphs)
    return batch_all_g



@torch.no_grad()
def evaluate(
    dataloader, 
    generator,
    output_dir,
    follow_batch,
    device_ids,
    trunc=1.0
    ):
    generator.eval()
    device = next(generator.parameters()).device

    gan_stats = {
        "gan": {
            "conn_total_inter": 0.0, "conn_total_gt": 0.0, "conn_total_pred": 0.0,
            "tpr_huber_sum": 0.0, "tpr_l1_sum": 0.0, "tpr_cnt": 0,
            "far_bias_sum": 0.0, "far_mae_sum": 0.0, "far_mse_sum": 0.0, "far_cnt": 0,
            "batch_conn_precision": [],
            "batch_conn_recall": [],
            "batch_conn_f1": [],
            "batch_TPR_huber": [],
            "batch_TPR_l1": [],
            "batch_FAR_bias": [],
            "batch_FAR_mae": [],
            "batch_FAR_mse": [],
            "batch_scattered_ratio": [],
            "batch_fragmentation_office": [],
            "batch_dist_off_elev": [],
            "batch_dist_off_rest": [],
            "batch_dist_off_stair": [],
            "batch_psnr": [],
            "batch_ssim": [],
        } 
    }

    pca_feats_node = {"gan": {"gt": [], "x0": []}}  # node-level
    pca_feats_graph = {"gan": {"gt": [], "x0": []}} # graph-level

    def _safe_div(a, b, nan_val=float("nan")):
        a = float(a)
        b = float(b)
        return a / b if b != 0.0 else nan_val

    for bidx, batch in enumerate(dataloader):
        st =gan_stats["gan"]
        g = batch.to(device) if hasattr(batch, 'to') else batch
        program_z_shape = [g.program_class_feature.shape[0], generator.noise_dim]
        program_z = torch.rand(tuple(program_z_shape)).to(device)
        voxel_z_shape = [g.voxel_feature.shape[0], generator.noise_dim]
        voxel_z = torch.rand(tuple(voxel_z_shape)).to(device)
        if trunc < 1.0:
            program_z.clamp_(min=-trunc, max=trunc)
            voxel_z.clamp_(min=-trunc, max=trunc)

        out, soft_out, mask, att, max_out_program_index = generator(g, program_z, voxel_z)

        raw_pos_norm = g.voxel_feature[:, 3:6] # Indices 3,4,5
        batch_pos_float = (raw_pos_norm * 100.0).cpu().numpy()
        
        batch_idx = g.voxel_feature_batch.cpu().numpy()
        batch_instance_ids = g.voxel_label_program_node_index.cpu().numpy()
        pred_labels_all = torch.argmax(soft_out, dim=1).cpu().numpy()
        
        # GT Labels
        if hasattr(g, 'voxel_label'):
            gt_labels_all = g.voxel_label.cpu().numpy()
            if gt_labels_all.ndim > 1: gt_labels_all = gt_labels_all.argmax(axis=-1)
        else:
            gt_labels_all = pred_labels_all

        num_graphs = g.num_graphs
        for i in range(num_graphs):
            mask_i = (batch_idx == i)
            coords_i_float = batch_pos_float[mask_i]
            
            if len(coords_i_float) > 0:
                coords_min = coords_i_float.min(axis=0)
                coords_i = np.round(coords_i_float - coords_min).astype(int)
            else:
                coords_i = np.zeros((0, 3), dtype=int)

            labels_pred_i = pred_labels_all[mask_i]
            labels_gt_i = gt_labels_all[mask_i]
            instance_ids_i = batch_instance_ids[mask_i]

            phys_stats_pred, label_grid_pred = calculate_physical_metrics(coords_i, coords_i_float, labels_pred_i, instance_ids_i)
            phys_stats_gt, label_grid_gt   = calculate_physical_metrics(coords_i, coords_i_float, labels_gt_i, instance_ids_i)
            d2_stats = calculate_2d_metrics(label_grid_gt, label_grid_pred, target_floor=1)

            phys_keys = ['scattered_ratio', 'fragmentation_office', 'dist_off_elev', 'dist_off_rest', 'dist_off_stair']
            
            # 1. abosolute value
            for k_res in phys_keys:
                k_list = f"batch_{k_res}"
                val = phys_stats_pred.get(k_res, float('nan'))
                if not np.isnan(val):
                    if k_list not in st: st[k_list] = []
                    st[k_list].append(val)
            
            # 2.  Bias
            for k_res in phys_keys:
                k_list_bias = f"batch_{k_res}_bias"
                val_pred = phys_stats_pred.get(k_res, float('nan'))
                val_gt   = phys_stats_gt.get(k_res, float('nan'))
                if not np.isnan(val_pred) and not np.isnan(val_gt):
                    if k_list_bias not in st: st[k_list_bias] = []
                    st[k_list_bias].append(val_pred - val_gt)

            # 3.  PSNR/SSIM
            for k_res in ['psnr', 'ssim']:
                k_list = f"batch_{k_res}"
                val = d2_stats.get(k_res, float('nan'))
                if not np.isnan(val):
                    if k_list not in st: st[k_list] = []
                    st[k_list].append(val)


        # ===== 1) Connectivity =====
        inter_edges, missing_edges, gen_edges = check_connectivity(g, max_out_program_index, mask['hard'])
        inter = inter_edges.shape[1]
        gt    = g.program_edge.shape[1]
        predE = gen_edges.shape[1]

        p  = _safe_div(inter, predE)
        r  = _safe_div(inter, gt)
        f1 = (2*p*r)/(p+r) if (not math.isnan(p) and not math.isnan(r) and (p+r)>0) else float("nan")
        st["batch_conn_precision"].append(p)
        st["batch_conn_recall"].append(r)
        st["batch_conn_f1"].append(f1)
        st["conn_total_inter"] += inter
        st["conn_total_gt"]    += gt
        st["conn_total_pred"]  += predE


        # ===== 2) TPR =====
        normalized_program_class_weight, normalized_program_weight, FAR = get_program_ratio(g, att["hard"], mask["hard"], area_index_in_voxel_feature=6)
        vfb = getattr(g, 'voxel_feature_batch', None)
        G = int(vfb.max().item()) + 1
        C = 6
        pred = normalized_program_class_weight.view(G, C)
        tgt  = g.program_target_ratio.view(G, C).to(pred.dtype)

        huber   = F.smooth_l1_loss(pred, tgt)
        l1_mean = (pred - tgt).abs().mean()
        st["batch_TPR_huber"].append(float(huber))
        st["batch_TPR_l1"].append(float(l1_mean))

        st["tpr_huber_sum"] += float(huber)
        st["tpr_l1_sum"]    += float(l1_mean)
        st["tpr_cnt"]       += 1

        # ===== 3) FAR =====
        FAR = FAR.squeeze(-1) 
        target_far = getattr(g, 'far', g.FAR)
        target_far = torch.as_tensor(target_far, device=FAR.device, dtype=FAR.dtype).view(-1)
        target_far = target_far.expand_as(FAR)
        err      = FAR - target_far          # [G] 
        abs_err  = err.abs()                 # [G]  
        sq_err   = err.pow(2)                # [G]  
        st["batch_FAR_bias"].append(float(err.mean()))
        st["batch_FAR_mae"].append(float(abs_err.mean()))
        st["batch_FAR_mse"].append(float(sq_err.mean()))

        st["far_bias_sum"] += float(err.mean())
        st["far_mae_sum"]  += float(abs_err.mean())
        st["far_mse_sum"]  += float(sq_err.mean())
        st["far_cnt"]      += int(FAR.numel())

        gen_feat = (soft_out * 2.0) - 1.0

        if not hasattr(g, 'voxel_label') or g.voxel_label is None:
             raise ValueError("g.voxel_label is missing! Cannot compare distributions.")
        

        if g.voxel_label.dim() == 1:
            num_classes = gen_feat.shape[1]
            
            # A.  Mask 
            mask = (g.voxel_label != -1).float().unsqueeze(-1)
            
            # B. Label 
            safe_label = g.voxel_label.clone()
            safe_label[safe_label == -1] = 0
            
            # C. One-Hot [0, 1]
            real_feat_onehot = F.one_hot(safe_label.long(), num_classes=num_classes).float()
            real_feat_onehot = real_feat_onehot * mask
            real_feat = (real_feat_onehot * 2.0) - 1.0

        elif g.voxel_label.dim() == 2 and g.voxel_label.shape[1] == gen_feat.shape[1]:
            # already one-hot
            real_feat = (g.voxel_label.float() * 2.0) - 1.0
        else:
            raise ValueError(
                f"g.voxel_label shape ({g.voxel_label.shape}) is not compatible "
                f"with soft_out shape ({gen_feat.shape}) for one-hot conversion."
            )
            
        pca_feats_node["gan"]["gt"].append(real_feat.cpu())
        pca_feats_node["gan"]["x0"].append(gen_feat.cpu())

        if vfb is None:
            vfb = g.batch
        gt_graph_feat = global_mean_pool(real_feat, vfb)
        pred_graph_feat = global_mean_pool(gen_feat, vfb)
        pca_feats_graph["gan"]["gt"].append(gt_graph_feat.cpu())
        pca_feats_graph["gan"]["x0"].append(pred_graph_feat.cpu())


        save_output(
            dataloader.batch_size, g, normalized_program_class_weight, normalized_program_weight, FAR,
            max_out_program_index, out, follow_batch, output_dir
        )

    summary = {}
    for mode, st in gan_stats.items():
        metrics = {}
        # Connectivity
        prec = _safe_div(st["conn_total_inter"], st["conn_total_pred"])
        rec  = _safe_div(st["conn_total_inter"], st["conn_total_gt"])
        f1   = (2 * prec * rec) / (prec + rec) if (not math.isnan(prec) and not math.isnan(rec) and (prec + rec) > 0) else float("nan")
        metrics = {
            "conn_precision": prec, "conn_recall": rec, "conn_f1": f1,
            "TPR_huber_mean": _safe_div(st["tpr_huber_sum"], st["tpr_cnt"]),
            "TPR_l1_mean":    _safe_div(st["tpr_l1_sum"],    st["tpr_cnt"]),
            "FAR_bias_mean":  _safe_div(st["far_bias_sum"],  st["far_cnt"]),
            "FAR_mae_mean":   _safe_div(st["far_mae_sum"],   st["far_cnt"]),
            "FAR_mse_mean":   _safe_div(st["far_mse_sum"],   st["far_cnt"]),
            "batch_conn_precision": st["batch_conn_precision"],
            "batch_conn_recall": st["batch_conn_recall"],
            "batch_conn_f1": st["batch_conn_f1"],
            "batch_TPR_huber": st["batch_TPR_huber"],
            "batch_TPR_l1": st["batch_TPR_l1"],
            "batch_FAR_bias": st["batch_FAR_bias"],
            "batch_FAR_mae": st["batch_FAR_mae"],
            "batch_FAR_mse": st["batch_FAR_mse"],
        }
        all_batch_keys = [k for k in st.keys() if k.startswith("batch_")]
        
        for k in all_batch_keys:
            data_list = st[k]
            if len(data_list) > 0:
                metrics[k.replace('batch_', '') + "_mean"] = np.mean(data_list)
                metrics[k] = data_list
            else:
                metrics[k] = []
        
        summary[mode] = metrics

    for mode in ["gan"]:
        pca_embeds_node = {}
        gt_list_node = pca_feats_node["gan"]["gt"]
        x0_list_node = pca_feats_node["gan"]["x0"]
        gt_node = np.concatenate(gt_list_node, axis=0) if gt_list_node else np.array([])
        x0_node = np.concatenate(x0_list_node, axis=0) if x0_list_node else np.array([])
        pca_embeds_node["gan"] = {"gt": gt_node, "x0": x0_node}

        pca_embeds_graph = {}
        gt_list_graph = pca_feats_graph["gan"]["gt"]
        x0_list_graph = pca_feats_graph["gan"]["x0"]
        gt_graph = np.concatenate(gt_list_graph, axis=0) if gt_list_graph else np.array([])
        x0_graph = np.concatenate(x0_list_graph, axis=0) if x0_list_graph else np.array([])
        pca_embeds_graph["gan"] = {"gt": gt_graph, "x0": x0_graph}
        gt_graph = np.concatenate(gt_list_graph, axis=0) if gt_list_graph else np.array([])
        x0_graph = np.concatenate(x0_list_graph, axis=0) if x0_list_graph else np.array([])
        pca_embeds_graph["gan"] = {"gt": gt_graph, "x0": x0_graph}
        
        if gt_graph.shape[0] > 1 and x0_graph.shape[0] > 1:
            mu_gt, sigma_gt = np.mean(gt_graph, axis=0), np.cov(gt_graph, rowvar=False)
            mu_pred, sigma_pred = np.mean(x0_graph, axis=0), np.cov(x0_graph, rowvar=False)
            
            fid = calculate_fid(mu_gt, sigma_gt, mu_pred, sigma_pred)
        else:
            fid = -1.0 # insufficient or empty samples
        summary[mode]["FID"] = fid
    return summary, pca_embeds_node, pca_embeds_graph
    
def check_connectivity(g, max_out_program_index, mask):
    """
    Extract connectivity from the generated design
        inter_edges:    program edge observed in the generated output
        missing_edges:  program edges only in the input program graph
        gen_edges:      program edges only in the generated output
    """
    # Look at the g.voxel_edge and see if the two voxel nodes are masked
    voxel_edge_out_mask = mask.reshape([-1])[g.voxel_edge]  # Ev x 2
    sums = torch.sum(voxel_edge_out_mask, dim=0)  # Ev x 1
    masked_edges = g.voxel_edge[:, sums == 2]   # Ev x 2 sums ==2 means voxel edges observed in the generated output

    if masked_edges.shape[1] != 0:
        # Now put program index onto the voxel edge and delete duplicates
        predicted_program_edges = torch.unique(max_out_program_index[masked_edges], dim=1)
        # union of program edges and program edges from the generated output
        mixed_edges = torch.cat((g.program_edge, predicted_program_edges), dim=1)
        unique_mix_edges, mix_counts = mixed_edges.unique(return_counts=True, dim=1)
        inter_edges = unique_mix_edges[:, mix_counts > 1]

        # program edges only in the input program graph
        mixed_gt_edges = torch.cat((g.program_edge, inter_edges), dim=1)
        unique_gt_edges, mix_gt_counts = mixed_gt_edges.unique(return_counts=True, dim=1)
        missing_edges = unique_gt_edges[:, mix_gt_counts == 1]

        # program edges only in the generated output
        mixed_gen_edges = torch.cat((predicted_program_edges, inter_edges), dim=1)
        unique_gen_edges, mix_gen_counts = mixed_gen_edges.unique(return_counts=True, dim=1)
        gen_edges = unique_gen_edges[:, mix_gen_counts == 1]
    else:  # there is no voxel edge
        inter_edges = masked_edges
        missing_edges = g.program_edge
        gen_edges = masked_edges

    return inter_edges, missing_edges, gen_edges

