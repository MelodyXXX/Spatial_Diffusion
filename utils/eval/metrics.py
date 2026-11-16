import numpy as np
import scipy.linalg
from scipy.spatial.distance import cdist
from scipy.ndimage import label, convolve
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

# ==========================================
# 1. FID (Frechet Inception Distance)
# ==========================================
def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """FID score between two Gaussians (mu1, sigma1) and (mu2, sigma2)."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    # sqrt of the product of covariance matrices
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # numerical stability fallback
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot((sigma2 + offset)))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

# ==========================================
# 3. Standard floor-plan metrics (PSNR / SSIM)
# ==========================================
def calculate_2d_metrics(gt_label_grid, pred_label_grid, target_floor=1):

    # check enough floors exist
    if gt_label_grid.shape[2] <= target_floor or pred_label_grid.shape[2] <= target_floor:
        return {'psnr': float('nan'), 'ssim': float('nan')}

    gt_slice = gt_label_grid[:, :, target_floor]
    pred_slice = pred_label_grid[:, :, target_floor]

    # convert to grayscale image, shifting class ids by +1 to avoid confusing Class 0 (Lobby) with Empty (-1)
    def to_img(sl):
        img = sl.astype(np.float32)
        mask = (img != -1)
        img[~mask] = 0
        img[mask] += 1
        # normalize assuming max value 6 (0..5 -> 1..6), scale to 255
        img = (img / 7.0) * 255.0
        return img.astype(np.uint8)

    gt_img = to_img(gt_slice)
    pred_img = to_img(pred_slice)

    # [Fix] Handle identical images
    mse = np.mean((gt_img - pred_img) ** 2)
    if mse == 0:
        # return NaN only if both images are truly all-black (no building at all)
        if np.max(gt_img) == 0 and np.max(pred_img) == 0:
             return {'psnr': float('nan'), 'ssim': float('nan')}
        else:
             # content present and matches exactly
             return {'psnr': 100.0, 'ssim': 1.0}

    val_psnr = psnr_func(gt_img, pred_img, data_range=255)
    val_ssim = ssim_func(gt_img, pred_img, data_range=255)
    
    return {'psnr': val_psnr, 'ssim': val_ssim}

def calculate_connectivity_metrics(pred_edges_set, gt_edges_set):
    tp = len(pred_edges_set.intersection(gt_edges_set))
    fp = len(pred_edges_set) - tp
    fn = len(gt_edges_set) - tp
    
    p = tp / (tp + fp + 1e-6)
    r = tp / (tp + fn + 1e-6)
    f1 = 2 * p * r / (p + r + 1e-6)
    
    return p, r, f1

def calculate_physical_metrics(grid_locs, phys_coords, labels, instance_ids, voxel_dim=64):
    """
    Calculates physical stats separating Logic (Grid) and Physics (Meters).
    ** Now calculates distances based on Room Centroids (per Floor). **
    
    Args:
        grid_locs (np.array): (N, 3) int - Logical Grid Indices [z, x, y]
        phys_coords (np.array): (N, 3) float - Physical Coordinates [z, x, y]
        labels (np.array): (N,) int - Voxel types
        instance_ids (np.array): (N,) int - Room Instance IDs (type_id)
    """
    stats = {}

    # 1. Filter based on Grid Boundaries
    valid_mask = (grid_locs >= 0) & (grid_locs < voxel_dim)
    valid_mask = valid_mask.all(axis=1)
    
    grid_locs = grid_locs[valid_mask]
    phys_coords = phys_coords[valid_mask]
    labels = labels[valid_mask]
    instance_ids = instance_ids[valid_mask] 
    
    # 2. Build Binary Grid (for Scattered/Frag)
    grid = np.zeros((voxel_dim, voxel_dim, voxel_dim), dtype=np.int16)
    label_grid = np.zeros((voxel_dim, voxel_dim, voxel_dim), dtype=np.int16) - 1
    
    if len(grid_locs) > 0:
        grid[grid_locs[:, 0], grid_locs[:, 1], grid_locs[:, 2]] = 1
        label_grid[grid_locs[:, 0], grid_locs[:, 1], grid_locs[:, 2]] = labels
        
    # --- A. Scattered Voxels ---
    real_voxel_mask = (labels != -1)
    real_voxel_locs = grid_locs[real_voxel_mask]
    voxel_set = set(tuple(loc) for loc in real_voxel_locs)
    scattered_count = 0
    surface_count = 0
    directions = [
        (0, 0, 1), (0, 0, -1),
        (0, 1, 0), (0, -1, 0),
        (1, 0, 0), (-1, 0, 0)
    ]
    for loc in voxel_set:
        x, y, z = loc
        neighbor_hits = 0
        for dx, dy, dz in directions:
            neighbor_loc = (x + dx, y + dy, z + dz)
            if neighbor_loc in voxel_set:
                neighbor_hits += 1
        # denominator: surface voxels (at least 1 face touches air, neighbor_hits < 6)
        if neighbor_hits < 6:
            surface_count += 1
        # numerator: scattered voxels (few neighbors -> floating or too thin), threshold < 3
        if neighbor_hits < 3:
            scattered_count += 1

    # ratio
    if surface_count > 0:
        stats['scattered_ratio'] = scattered_count / surface_count
    else:
        stats['scattered_ratio'] = 0.0

    # --- B. Fragmentation Index (Office connectivity) ---
    office_binary = (label_grid == 4).astype(int)
    _, num_features = label(office_binary)
    stats['fragmentation_office'] = num_features

    # --- C. Room-to-Room Distances (Centroid based) ---
    target_map = {
        'elev': 3,  # verify elevator label id is 3 in your data
        'rest': 1,  # restroom
        'stair': 2  # stairs
    }

    def get_layer_room_dist(target_label):
        all_dists = []
        unique_floors = np.unique(grid_locs[:, 0])
        
        for floor_z in unique_floors:
            # per-floor mask and data
            mask = (grid_locs[:, 0] == floor_z)

            f_phys = phys_coords[mask]   # [N_f, 3] -> (z, x, y)
            f_lbls = labels[mask]        # [N_f,]
            f_insts = instance_ids[mask] # [N_f,]

            # compute room centroids: offices (Type 4) on this floor
            office_mask = (f_lbls == 4)
            if not np.any(office_mask): continue

            unique_off_ids = np.unique(f_insts[office_mask])
            off_centroids = []

            for oid in unique_off_ids:
                # voxels belonging to this specific room, centroid = mean of (x, y)
                room_voxels = f_phys[(f_lbls == 4) & (f_insts == oid)]
                centroid = np.mean(room_voxels[:, 1:3], axis=0)
                off_centroids.append(centroid)

            # target facility (Type = target_label)
            target_mask = (f_lbls == target_label)
            if not np.any(target_mask): continue
            
            unique_tgt_ids = np.unique(f_insts[target_mask])
            tgt_centroids = []
            
            for tid in unique_tgt_ids:
                room_voxels = f_phys[(f_lbls == target_label) & (f_insts == tid)]
                centroid = np.mean(room_voxels[:, 1:3], axis=0)
                tgt_centroids.append(centroid)
            
            # distance between centroids: off_centroids (N_offices,2), tgt_centroids (N_targets,2)
            if len(off_centroids) > 0 and len(tgt_centroids) > 0:
                dists = cdist(off_centroids, tgt_centroids, metric='euclidean')
                # nearest target for each office
                min_dists = np.min(dists, axis=1)
                all_dists.extend(min_dists)

        if len(all_dists) == 0:
            return float('nan')
        
        return np.mean(all_dists)

    stats['dist_off_elev'] = get_layer_room_dist(target_map['elev'])
    stats['dist_off_rest'] = get_layer_room_dist(target_map['rest'])
    stats['dist_off_stair'] = get_layer_room_dist(target_map['stair'])

    return stats, label_grid


