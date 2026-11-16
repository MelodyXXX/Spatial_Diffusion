import os, json
from utils.data.caption import ensure_dir
from utils.viz import render_voxel_isometric
from utils.util_graph import rebatch_for_multi_gpu
from utils.eval.metrics import calculate_fid, calculate_physical_metrics, calculate_2d_metrics
from scipy.ndimage import label as scipy_label
import numpy as np
from utils.data.GraphConstructor import GraphConstructor
from sklearn.cluster import KMeans
from torch_scatter import scatter
import torch, torch.nn.functional as F
from torch_geometric.nn import global_mean_pool



def _get_ground_truth_x0(batch, num_classes):
    """
    Build (N_voxels, 1+C) tensor using the "placeholder -1" logic
    """
    # a. build mask (N_voxels, 1)
    ground_truth_mask = (batch.voxel_label != -1).float().unsqueeze(-1)

    # b. build label (N_voxels, C)
    labels_for_onehot = batch.voxel_label.clone()
    labels_for_onehot[labels_for_onehot == -1] = 0 # replace placeholder -1 with 0 to avoid one-hot error
    labels_for_onehot = labels_for_onehot.long()
    ground_truth_label = F.one_hot(labels_for_onehot, num_classes=num_classes).float()
    ground_truth_label = ground_truth_label * ground_truth_mask

    # d. concatenate
    ground_truth_mask_scaled = (ground_truth_mask * 2.0) - 1.0
    ground_truth_label_scaled = (ground_truth_label * 2.0) - 1.0 #  [-1, 1]
    ground_truth_x0 = torch.cat([ground_truth_mask_scaled, ground_truth_label_scaled], dim=-1)
    return ground_truth_x0 #  (N_voxels, 1 + C)

def get_program_metrics_and_instances(g, final_voxel_program, num_classes, area_index_in_voxel_feature=6,
                                      xy_axes=(1, 2), group_log=None):
    """
    [K-Means forced split version]
    Resolves the issue where room merging causes all voxels to be assigned a single room ID (0).

    xy_axes: indices into the 3 coordinate columns used as the in-floor plane. Coordinates are
             ordered (z, x, y), so the default (1, 2) is the horizontal plane. "auto" reproduces
             the old behaviour (two largest-std axes), which picks the vertical axis for tall,
             narrow buildings and collapses each floor onto a line -- diagnostics only.
    group_log: optional list; if given, one dict per (graph, floor, class) group is appended
               describing which split branch was taken (diagnostics only).
    """
    device = g.voxel_feature.device
    Nv = g.voxel_feature.shape[0]
    Np = g.program_class_feature.shape[0]

    # --- 1. Retrieve ground truth structural info ---
    vfb = getattr(g, 'voxel_feature_batch', g.batch).cpu().numpy()
    pfb = g.program_class_feature_batch.cpu().numpy()
    program_node_types = g.program_class_feature.argmax(dim=-1).cpu().numpy()

    # Get floor index for each program node (after batch accumulation)
    if hasattr(g, 'program_floor_cluster'):
        prog_floors = g.program_floor_cluster.cpu().numpy()
    elif hasattr(g, 'program_floor_index'):
        prog_floors = g.program_floor_index.cpu().numpy()
    else:
        prog_floors = np.zeros(Np, dtype=int)

    # Get floor index for each voxel (after batch accumulation)
    if hasattr(g, 'voxel_floor_cluster'):
        voxel_floors = g.voxel_floor_cluster.cpu().numpy()
    else:
        # fallback: infer from Z axis
        voxel_floors = np.round(g.voxel_feature[:, 3].cpu().numpy() * 100).astype(int)

    # Extract coordinates for connected-component / K-Means computation
    raw_coords = g.voxel_feature[:, 3:6].cpu().numpy()

    # Initialize outputs
    voxel_pid_np = np.full(Nv, -1, dtype=np.int64)
    program_area_np = np.zeros(Np, dtype=np.float32)

    pred_labels = final_voxel_program.cpu().numpy()
    voxel_areas = g.voxel_feature[:, area_index_in_voxel_feature].cpu().numpy()

    for i in range(g.num_graphs):
        mask_voxel = (vfb == i)
        if not np.any(mask_voxel): continue

        curr_voxel_indices = np.where(mask_voxel)[0]
        c_v_floors = voxel_floors[mask_voxel]
        c_coords = raw_coords[mask_voxel]
        c_labels = pred_labels[mask_voxel]

        # In-floor plane axes (for 2D processing)
        if isinstance(xy_axes, str) and xy_axes == "auto":
            xy_idx = np.argsort(np.std(c_coords, axis=0))[-2:]
        else:
            xy_idx = np.asarray(xy_axes)
        c_x, c_y = c_coords[:, xy_idx[0]], c_coords[:, xy_idx[1]]

        # Discretize coordinates to build 2D grid
        x_uniq = np.sort(np.unique(c_x))
        y_uniq = np.sort(np.unique(c_y))
        x_map = {val: k for k, val in enumerate(x_uniq)}
        y_map = {val: k for k, val in enumerate(y_uniq)}

        local_x = np.array([x_map[v] for v in c_x])
        local_y = np.array([y_map[v] for v in c_y])

        grid_h = len(x_uniq)
        grid_w = len(y_uniq)

        unique_layers = np.sort(np.unique(c_v_floors))

        # --- 3. Per-floor processing (strict floor matching) ---
        for floor_id in unique_layers:
            mask_layer_voxel = (c_v_floors == floor_id)
            if not np.any(mask_layer_voxel): continue

            # Find all program nodes on this floor as specified by the input graph
            layer_prog_mask = (pfb == i) & (prog_floors == floor_id)
            layer_prog_indices = np.where(layer_prog_mask)[0]
            if len(layer_prog_indices) == 0: continue

            # --- Per-class processing ---
            for c in range(num_classes):
                # A. Find candidate slots in the input graph
                candidates = layer_prog_indices[program_node_types[layer_prog_indices] == c]
                candidates.sort()  # sort to ensure IDs map in order 0, 1, 2, ...

                target_count = len(candidates)
                if target_count == 0: continue

                # B. Find predicted voxels of this class
                mask_blob = mask_layer_voxel & (c_labels == c)
                if not np.any(mask_blob): continue

                # Get global indices and local coordinates for the blob voxels
                blob_global_indices = curr_voxel_indices[mask_blob]
                l_x = local_x[mask_blob]
                l_y = local_y[mask_blob]

                # List to store the final split blobs
                final_blobs = []

                # --- Strategy branch ---
                if target_count == 1:
                    # Only one target — assign everything directly, no clustering needed
                    b_area = np.sum(voxel_areas[blob_global_indices])
                    final_blobs.append({'area': b_area, 'indices': blob_global_indices})
                    if group_log is not None:
                        group_log.append({"graph": i, "floor": int(floor_id), "cls": c, "n_rooms": 1,
                                          "n_components": None, "branch": "single",
                                          "voxel_indices": blob_global_indices})

                else:
                    # Multiple targets — first try physical connectivity
                    grid = np.zeros((grid_h, grid_w), dtype=int)
                    grid[l_x, l_y] = 1
                    labeled, n_feat = scipy_label(grid, structure=np.ones((3,3)))
                    if group_log is not None:
                        group_log.append({"graph": i, "floor": int(floor_id), "cls": c, "n_rooms": target_count,
                                          "n_components": int(n_feat),
                                          "branch": "kmeans" if n_feat < target_count else
                                                    ("components" if n_feat == target_count else "overflow"),
                                          "voxel_indices": blob_global_indices})

                    if n_feat < target_count:
                        # [Core logic] Not enough connected components (rooms merged) -> force split with K-Means
                        # Use raw physical coords (c_x, c_y) for clustering — more accurate than grid coords
                        kmeans_data = np.stack([c_x[mask_blob], c_y[mask_blob]], axis=1)
                        unique_coords = np.unique(kmeans_data, axis=0)
                        n_unique = len(unique_coords)
                        n_clusters = min(target_count, len(kmeans_data), n_unique)
                        if n_clusters > 1:
                            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                            k_labels = kmeans.fit_predict(kmeans_data)
                        else:
                            # Only one cluster or cannot cluster — assign everything to cluster 0
                            k_labels = np.zeros(len(kmeans_data), dtype=int)
                            n_clusters = 1

                        for kid in range(n_clusters):
                            k_mask = (k_labels == kid)
                            k_ind = blob_global_indices[k_mask]
                            k_area = np.sum(voxel_areas[k_ind])
                            final_blobs.append({'area': k_area, 'indices': k_ind})

                    else:
                        # Enough connected components — use them directly
                        blob_ids = labeled[l_x, l_y]
                        for bid in range(1, n_feat + 1):
                            b_ind = blob_global_indices[blob_ids == bid]
                            b_area = np.sum(voxel_areas[b_ind])
                            final_blobs.append({'area': b_area, 'indices': b_ind})

                # C. Assignment logic
                # Sort by area descending so larger rooms map to lower-index nodes (e.g. master bedroom / living room)
                final_blobs.sort(key=lambda x: x['area'], reverse=True)

                for k, blob in enumerate(final_blobs):
                    if k < len(candidates):
                        target = candidates[k]
                    else:
                        # Overflow fragments go to the last candidate node
                        target = candidates[-1]

                    voxel_pid_np[blob['indices']] = target
                    program_area_np[target] += blob['area']


    voxel_pid_np[voxel_pid_np == -1] = 0
    voxel_pid = torch.tensor(voxel_pid_np, device=device)
    program_weight = torch.tensor(program_area_np, device=device)

    mask = (final_voxel_program != -1).float().unsqueeze(-1)
    labels = final_voxel_program.clone()
    labels[labels == -1] = 0
    labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
    
    voxel_area_t = g.voxel_feature[:, area_index_in_voxel_feature].view(-1, 1)
    masked_area = (mask * voxel_area_t).squeeze()
    FAR = scatter(masked_area, g.voxel_feature_batch.long(), dim=0, reduce="sum")
    
    per_graph_area = FAR[g.program_class_feature_batch.long()]
    normalized_program_weight = program_weight / (per_graph_area + 1e-16)

    program_class_weight = scatter(labels_one_hot * mask * voxel_area_t, g.voxel_feature_batch.long(), dim=0, reduce="sum")
    normalized_program_class_weight = program_class_weight / (FAR.unsqueeze(-1) + 1e-16)

    return normalized_program_class_weight, normalized_program_weight, FAR, voxel_pid

def assign_voxel_instance_ids(voxel_graph, final_labels, pid, pid_offset, node_lookup):
    """
    Writes per-voxel (type, type_id) into voxel_graph["voxel_node"] in place.
    final_labels: (Nv,) class per voxel (-1 = void); pid: (Nv,) program node index per voxel
    from get_program_metrics_and_instances; node_lookup: {local node idx: [type, type_id]}.
    """
    for n, label_index_float, _pid in zip(voxel_graph["voxel_node"], final_labels, pid):
        label_index = int(label_index_float)
        lookup_key = _pid.item() - pid_offset

        if label_index == -1:
            n["type"] = -1
            n["type_id"] = 0
        else:
            if lookup_key in node_lookup:
                query = node_lookup[lookup_key]
                n["type"] = label_index
                n["type_id"] = query[1]
            else:
                n["type"] = label_index
                n["type_id"] = 0


def save_output(batch_size, batch, class_weights, program_weights, FAR, max_out_program_index, out, follow_batch, output_dir, new_data_id_strs=None):
    """
    Save the evaluation results
    """
    if not os.path.exists(os.path.join(output_dir, "global_graph_data")):
        os.mkdir(os.path.join(output_dir, "global_graph_data"))
        os.mkdir(os.path.join(output_dir, "local_graph_data"))
        os.mkdir(os.path.join(output_dir, "voxel_data"))
        os.mkdir(os.path.join(output_dir, "prompts"))

    num_of_program_node_accum = 0
    batch_all_g = []
    
    # 2. Rebatch
    data = rebatch_for_multi_gpu(
        batch, list(range(batch_size)), follow_batch, 
        out, class_weights, program_weights, FAR, max_out_program_index
    )

    for i, (g, o, cw, pw, far, pid) in enumerate(data):
        data_id_str = g["data_id_str"][0]
        new_data_id_str = g["data_id_str"][0] if new_data_id_strs is None else str(new_data_id_strs[i]).zfill(GraphConstructor.data_id_length)
        
        final_labels = o.cpu().data.numpy() # (Nv,)
        cw, far = cw.squeeze(0).cpu().data.numpy(), far.item()

        # --- Modify Global data ---
        global_graph = json.loads(g["raw_global_graph_json"][0])
        global_graph["new_far"] = far
        for n in global_graph["global_node"]:
            class_index = n['type']
            if class_index < len(cw): 
                 n["new_proportion"] = float(cw[class_index])
            else:
                 n["new_proportion"] = 0.0
        with open(os.path.join(output_dir, "global_graph_data", GraphConstructor.global_graph_prefix + new_data_id_str + ".json"), 'w') as f:
            json.dump(global_graph, f)

        # --- Modify Local data ---
        d = {}
        local_graph = json.loads(g["raw_local_graph_json"][0])
        
        # pw = program_weights (normalized area)
        for idx, (n, c) in enumerate(zip(local_graph["node"], pw)):
            n["region_far"] = float(c)
            # map local idx -> [room type, room id]
            d[idx] = [n["type"], n["type_id"]]
            
        with open(os.path.join(output_dir, "local_graph_data", GraphConstructor.local_graph_prefix + new_data_id_str + ".json"), 'w') as f:
            json.dump(local_graph, f) 

        # --- Modify Voxel data ---
        voxel_graph = json.loads(g["raw_voxel_graph_json"][0])
        
        assign_voxel_instance_ids(voxel_graph, final_labels, pid, num_of_program_node_accum, d)

        num_of_program_node_accum += pw.shape[0]

        voxel_json_path = os.path.join(output_dir, "voxel_data", GraphConstructor.voxel_graph_prefix + new_data_id_str + ".json")
        with open(voxel_json_path, 'w') as f:
            json.dump(voxel_graph, f)
        render_dir = ensure_dir(os.path.join(output_dir, "renders"))
        render_voxel_isometric(voxel_json_path, os.path.join(render_dir, GraphConstructor.voxel_graph_prefix + new_data_id_str + ".png"), save_pdf=False)

        def _first_or(val, default=""):
            if isinstance(val, (list, tuple)):
                return val[0] if val else default
            return val if val is not None else default

        prompt_payload = {
            "data_id":              new_data_id_str,
            "prompt":               _first_or(getattr(g, "prompt_variants",        None)),
            "llm_variants":         list(getattr(g, "llm_variants",               []) or []),
            "rule_based_brief":     _first_or(getattr(g, "rule_based_brief",       None)),
            "rule_based_description": _first_or(getattr(g, "rule_based_description", None)),
        }
        with open(os.path.join(output_dir, "prompts", f"prompt_{new_data_id_str}.json"), 'w') as f:
            json.dump(prompt_payload, f, ensure_ascii=False, indent=2)

        all_graphs = [global_graph, local_graph, voxel_graph]
        batch_all_g.append(all_graphs)
        
    return batch_all_g


@torch.no_grad()
def evaluate_diffusion(
    model,
    dataloader,
    modes=("program","text","both"),
    writer_train = None,
    epoch = 1,
    out_dir = None,
):
    model.eval()
    device = "cuda:0"

    # ---------- accumulators ----------
    phys_keys = [
        'scattered_ratio', 
        'fragmentation_office', 
        'dist_off_elev', 
        'dist_off_rest', 
        'dist_off_stair'
    ]
    per_mode_stats = {
        m: {
            "tpr_huber_sum": 0.0, "tpr_l1_sum": 0.0, "tpr_cnt": 0,
            "far_bias_sum": 0.0, "far_mae_sum": 0.0, "far_mse_sum": 0.0, "far_cnt": 0,
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
            **{f"batch_{k}": [] for k in phys_keys},
            **{f"batch_{k}_bias": [] for k in phys_keys},
            "batch_psnr": [],
            "batch_ssim": [],
        } for m in modes
    }

    pca_feats_node = {m: {"gt": [], "x0": []} for m in modes}
    pca_feats_graph = {m: {"gt": [], "x0": []} for m in modes}

    def _safe_div(a, b, nan_val=float("nan")):
        a = float(a)
        b = float(b)
        return a / b if b != 0.0 else nan_val
    
    for bidx, batch in enumerate(dataloader):
        g = batch.to(device) if hasattr(batch, 'to') else batch
        prompts = getattr(g, "prompts", None)

        gt_x0 = _get_ground_truth_x0(g, model.num_classes) # (Nv, 1+C)
        vfb = getattr(g, 'voxel_feature_batch', g.batch)
        gt_x0_pca = gt_x0[:, 1:] # Shape becomes (Nv, C)
        gt_graph_feat_pca = global_mean_pool(gt_x0_pca, vfb) # Shape becomes (G, C)


            
        for mode in modes:
            st = per_mode_stats[mode]
            out = model.sample(g, prompts_per_graph=prompts, mode=mode)

            # Recover real coords from voxel_feature[:, 3:6] (encoded as coord/100)
            raw_pos_norm = g.voxel_feature[:, 3:6]  # (N, 3)
            batch_pos_float = (raw_pos_norm * 100.0).cpu().numpy()

            batch_idx = g.voxel_feature_batch.cpu().numpy()
            batch_instance_ids = g.voxel_label_program_node_index.cpu().numpy()
            pred_labels_all = out["pred_hard_labels"].cpu().numpy()
            
            # Prepare GT labels for PSNR/SSIM (one-hot -> index; fallback to pred labels if no GT)
            if hasattr(g, 'voxel_label'):
                gt_labels_all = g.voxel_label.cpu().numpy()
                if gt_labels_all.ndim > 1:
                    gt_labels_all = gt_labels_all.argmax(axis=-1)
            else:
                gt_labels_all = pred_labels_all

            # 2. per-graph physical metrics
            num_graphs = g.num_graphs
            for i in range(num_graphs):
                mask_i = (batch_idx == i)

                coords_i_float = batch_pos_float[mask_i]

                # zero-center coords and round to int
                if len(coords_i_float) > 0:
                    coords_min = coords_i_float.min(axis=0)
                    coords_i = np.round(coords_i_float - coords_min).astype(int)
                else:
                    coords_i = np.zeros((0, 3), dtype=int)

                labels_pred_i = pred_labels_all[mask_i]
                labels_gt_i = gt_labels_all[mask_i]
                instance_ids_i = batch_instance_ids[mask_i]

                # A. physical metrics (scattered ratio, fragmentation, distances)
                phys_stats, label_grid_pred = calculate_physical_metrics(coords_i, coords_i_float, labels_pred_i, instance_ids_i)

                # B. 2D metrics (PSNR/SSIM) -- restore GT grid first
                _, label_grid_gt = calculate_physical_metrics(coords_i, coords_i_float, labels_gt_i, instance_ids_i)
                d2_stats = calculate_2d_metrics(label_grid_gt, label_grid_pred, target_floor=1)

                full_stats = {**phys_stats, **d2_stats}
                
                metric_map = [
                    ('scattered_ratio', 'batch_scattered_ratio'),
                    ('fragmentation_office', 'batch_fragmentation_office'),
                    ('dist_off_elev', 'batch_dist_off_elev'),
                    ('dist_off_rest', 'batch_dist_off_rest'),
                    ('dist_off_stair', 'batch_dist_off_stair'),
                    ('psnr', 'batch_psnr'),
                    ('ssim', 'batch_ssim')
                ]
                
                for k_res, k_list in metric_map:
                    val = full_stats.get(k_res, float('nan'))
                    if not np.isnan(val):
                        # lazy-init list on first occurrence of this key
                        if k_list not in st: st[k_list] = []
                        st[k_list].append(val)

            #---------- PCA collection (updated) ----------
            soft_x0_labels = out["pred_soft_x0"][:, 1:]
            pred_hard_labels = out["pred_hard_labels"] # (Nv,)
            safe_hard_labels = pred_hard_labels.clone()
            safe_hard_labels[safe_hard_labels == -1] = 0 # replace-1
            pred_hard_x0_pca = F.one_hot(safe_hard_labels, num_classes=model.num_classes).float()
            pred_hard_x0_pca[pred_hard_labels == -1] = 0.0
            pred_graph_feat_pca = global_mean_pool(soft_x0_labels, vfb)
            pca_feats_node[mode]["gt"].append(gt_x0_pca.detach().cpu().numpy())
            pca_feats_node[mode]["x0"].append(soft_x0_labels.detach().cpu().numpy())
            pca_feats_graph[mode]["gt"].append(gt_graph_feat_pca.detach().cpu().numpy())
            pca_feats_graph[mode]["x0"].append(pred_graph_feat_pca.detach().cpu().numpy())
            
            pred_labels_all = out["pred_hard_labels"].cpu().numpy()
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
                instance_ids_i = batch_instance_ids[mask_i]

                # pred metrics
                phys_stats_pred, label_grid_pred = calculate_physical_metrics(coords_i, coords_i_float, labels_pred_i, instance_ids_i)

                # GT metrics (if available)
                phys_stats_gt = {}
                d2_stats = {'psnr': float('nan'), 'ssim': float('nan')}

                if gt_labels_all is not None:
                    labels_gt_i = gt_labels_all[mask_i]
                    phys_stats_gt, label_grid_gt = calculate_physical_metrics(coords_i, coords_i_float, labels_gt_i, instance_ids_i)
                    d2_stats = calculate_2d_metrics(label_grid_gt, label_grid_pred, target_floor=1)

                # --- store physical metrics & bias ---
                for key in phys_keys:
                    val_pred = phys_stats_pred.get(key, float('nan'))
                    val_gt = phys_stats_gt.get(key, float('nan'))

                    # store raw pred value
                    if not np.isnan(val_pred):
                        st[f"batch_{key}"].append(val_pred)

                    # store bias (pred - GT)
                    if not np.isnan(val_pred) and not np.isnan(val_gt):
                        bias = val_pred - val_gt
                        st[f"batch_{key}_bias"].append(bias)

                # --- store 2D metrics ---
                if not np.isnan(d2_stats['psnr']): st['batch_psnr'].append(d2_stats['psnr'])
                if not np.isnan(d2_stats['ssim']): st['batch_ssim'].append(d2_stats['ssim'])
            # ---------- TPR  ----------
            normalized_program_class_weight, normalized_program_weight, FAR, voxel_pid = get_program_metrics_and_instances(
                g, 
                final_voxel_program=out["pred_hard_labels"], 
                num_classes=model.num_classes,
                area_index_in_voxel_feature=6
            )
            G = g.num_graphs
            C = model.num_classes
            pred = normalized_program_class_weight.view(G, C)
            tgt = g.program_target_ratio.view(G, C).to(pred.dtype)

            huber   = F.smooth_l1_loss(pred, tgt)
            l1_mean = (pred - tgt).abs().mean()
            st["batch_TPR_huber"].append(float(huber))
            st["batch_TPR_l1"].append(float(l1_mean))
            st["tpr_huber_sum"] += float(huber)
            st["tpr_l1_sum"]    += float(l1_mean)
            st["tpr_cnt"]       += 1

            # ---------- FAR (this batch) ----------
            # NOTE: use reshape(-1) rather than squeeze(-1) -- FAR is already
            # shape (G,) from scatter(), so squeeze(-1) would collapse the
            # batch dim itself (into a 0-d scalar) whenever G == 1.
            FAR = FAR.reshape(-1)
            target_far = getattr(g, 'far', g.FAR)
            target_far = torch.as_tensor(target_far, device=FAR.device, dtype=FAR.dtype).view(-1)
            target_far = target_far.expand_as(FAR)
            err      = FAR - target_far          # [G]  signed
            abs_err  = err.abs()                 # [G]  absolute deviation
            sq_err   = err.pow(2)                # [G]  squared deviation
            st["batch_FAR_bias"].append(float(err.mean()))
            st["batch_FAR_mae"].append(float(abs_err.mean()))
            st["batch_FAR_mse"].append(float(sq_err.mean()))
            st["far_bias_sum"] += float(err.mean())
            st["far_mae_sum"]  += float(abs_err.mean())
            st["far_mse_sum"]  += float(sq_err.mean())
            st["far_cnt"]      += int(FAR.numel())

            # ---------- save outputs (one dir per mode) ----------
            mode_out_dir = ensure_dir(os.path.join(out_dir, mode))
            follow_batch = ['program_target_ratio', 'program_class_feature', 'voxel_feature']

            save_output(
                dataloader.batch_size, g,
                normalized_program_class_weight,
                normalized_program_weight,
                FAR,
                voxel_pid,                  # <-- must pass voxel_pid (all non-negative ints), not labels
                out["pred_hard_labels"],    # labels passed here (used for n["type"])
                follow_batch, mode_out_dir, new_data_id_strs=None
            )

            # additionally save the pre-refine (unsmoothed) labels to a sibling
            # dir — only for the smoothing on/off ablation (scattered_ratio /
            # fragmentation_office), does not touch the main output above.
            raw_mode_out_dir = ensure_dir(os.path.join(out_dir, mode + "_nosmooth"))
            save_output(
                dataloader.batch_size, g,
                normalized_program_class_weight,
                normalized_program_weight,
                FAR,
                voxel_pid,
                out["pred_hard_labels_raw"],
                follow_batch, raw_mode_out_dir, new_data_id_strs=None
            )
    # ---------- aggregate ----------
    summary = {}
    for mode, st in per_mode_stats.items():
        metrics = {
            "TPR_huber_mean": _safe_div(st["tpr_huber_sum"], st["tpr_cnt"]),
            "TPR_l1_mean":    _safe_div(st["tpr_l1_sum"],    st["tpr_cnt"]),
            "FAR_bias_mean":  _safe_div(st["far_bias_sum"],  st["far_cnt"]),
            "FAR_mae_mean":   _safe_div(st["far_mae_sum"],   st["far_cnt"]),
            "FAR_mse_mean":   _safe_div(st["far_mse_sum"],   st["far_cnt"]),

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
                # store mean (e.g. scattered_ratio_mean) and keep raw list for plotting
                metrics[k.replace('batch_', '') + "_mean"] = np.mean(data_list)
                metrics[k] = data_list
            else:
                metrics[k] = []
        summary[mode] = metrics

    # ---------- PCA viz embeddings: concat lists into (N,H) ----------
    pca_embeds_node = {}
    for mode in modes:
        gt_list = pca_feats_node[mode]["gt"]
        x0_list = pca_feats_node[mode]["x0"]
        gt = np.concatenate(gt_list, axis=0)
        x0 = np.concatenate(x0_list, axis=0)
        pca_embeds_node[mode] = {"gt": gt, "x0": x0}

    pca_embeds_graph = {}
    for mode in modes:
        gt_list = pca_feats_graph[mode]["gt"]
        x0_list = pca_feats_graph[mode]["x0"]
        if not gt_list or not x0_list:
             pca_embeds_graph[mode] = {"gt": np.array([]), "x0": np.array([])}
             continue
        gt = np.concatenate(gt_list, axis=0)
        x0 = np.concatenate(x0_list, axis=0)
        pca_embeds_graph[mode] = {"gt": gt, "x0": x0}

    device = "cuda:0"
    for mode in ["program", "text", "both"]:
    # for mode in ["program"]:
        gt = pca_embeds_node[mode]["gt"]
        x0 = pca_embeds_node[mode]["x0"]
        if gt.size == 0 or x0.size == 0:
            continue

        gt_t = torch.tensor(gt, dtype=torch.float32, device=device)
        x0_t = torch.tensor(x0, dtype=torch.float32, device=device)
        pca_embeds_node[mode]["gt"] = gt_t
        pca_embeds_node[mode]["x0"] = x0_t
    for mode in ["program", "text", "both"]:
    # for mode in ["program"]:
        # (ensure mode exists and data non-empty)
        if mode not in pca_embeds_graph: continue
        gt = pca_embeds_graph[mode]["gt"]
        x0 = pca_embeds_graph[mode]["x0"]
        if gt.size == 0 or x0.size == 0:
            continue
        
        gt_t = torch.tensor(gt, dtype=torch.float32, device=device)
        x0_t = torch.tensor(x0, dtype=torch.float32, device=device)
        # overwrite the numpy arrays
        pca_embeds_graph[mode]["gt"] = gt_t
        pca_embeds_graph[mode]["x0"] = x0_t

    # compute FID
    for mode in modes:
        if mode not in pca_embeds_graph: continue
        
        gt_feats = pca_embeds_graph[mode]["gt"]
        pred_feats = pca_embeds_graph[mode]["x0"]

        if hasattr(gt_feats, 'cpu'): 
            gt_feats = gt_feats.detach().cpu().numpy()
        if hasattr(pred_feats, 'cpu'): 
            pred_feats = pred_feats.detach().cpu().numpy()
            
        # concat if still a list (not yet merged)
        if isinstance(gt_feats, list) and len(gt_feats) > 0:
            gt_feats = np.concatenate(gt_feats, axis=0)
        if isinstance(pred_feats, list) and len(pred_feats) > 0:
            pred_feats = np.concatenate(pred_feats, axis=0)

        # 2. compute FID (guard against empty data)
        if hasattr(gt_feats, 'shape') and gt_feats.shape[0] > 1 and pred_feats.shape[0] > 1:
            # gt_feats now a numpy array, safe for np.mean
            mu_gt, sigma_gt = np.mean(gt_feats, axis=0), np.cov(gt_feats, rowvar=False)
            mu_pred, sigma_pred = np.mean(pred_feats, axis=0), np.cov(pred_feats, rowvar=False)
            
            fid = calculate_fid(mu_gt, sigma_gt, mu_pred, sigma_pred)
            summary[mode]["FID"] = fid
        else:
            summary[mode]["FID"] = -1.0
            
        # 3. aggregate other batch metrics (mean)
        extra_keys = ['batch_scattered_ratio', 'batch_fragmentation_office',
                      'batch_dist_off_elev', 'batch_dist_off_rest', 'batch_dist_off_stair',
                      'batch_psnr', 'batch_ssim']
        for k in extra_keys:
            if k in per_mode_stats[mode] and len(per_mode_stats[mode][k]) > 0:
                # store mean (for print/CSV) and raw list (needed by inference script's boxplots)
                summary[mode][k.replace('batch_', '') + "_mean"] = np.mean(per_mode_stats[mode][k])
                summary[mode][k] = per_mode_stats[mode][k]
    return summary, pca_embeds_node, pca_embeds_graph

