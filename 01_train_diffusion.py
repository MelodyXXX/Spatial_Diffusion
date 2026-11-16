# -*- coding: utf-8 -*-

import os, torch
import numpy as np
import random
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.model.diffusion import DiffusionGenerator
from utils.eval.evaldi import evaluate_diffusion
from utils.model.t2p import T5TextEncoder
from utils.args import Args
from utils.data.caption import _common_filenames, CaptionAugmentedDataset, FixVoxelLabelTransform
from pprint import pprint

os.environ["TOKENIZERS_PARALLELISM"] = "false"




if __name__ == "__main__":
    args = Args()
    for d in [os.path.join(args.run_dir, args.run_id), args.viz_dir, args.model_dir]:
        os.makedirs(d, exist_ok=True)

    torch.manual_seed(7); random.seed(7); np.random.seed(7)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    device_ids = list(range(torch.cuda.device_count()))
    if device_ids:
        device = torch.device('cuda:' + str(device_ids[0]))
        print('Using GPU {}'.format(os.environ.get('CUDA_VISIBLE_DEVICES')))
    else:
        device = torch.device('cpu')
        print("No GPU found, using CPU")

    writer_train = SummaryWriter(log_dir=os.path.join(args.run_dir, args.run_id))
    fix_transform = FixVoxelLabelTransform()

    # ---------- Load Data ----------
    all_fns = _common_filenames(args.dir_graph)
    train_size = args.train_size // args.batch_size * args.batch_size
    test_size  = args.test_size  // args.batch_size * args.batch_size
    train_fns  = all_fns[:train_size]
    test_fns   = all_fns[train_size + 4000: train_size + 4000 + test_size]

    train_ds = CaptionAugmentedDataset(args.dir_graph, filenames=train_fns, transform=fix_transform)
    test_ds  = CaptionAugmentedDataset(args.dir_graph, filenames=test_fns,  transform=fix_transform)
    train_loader = DataLoader(train_ds, follow_batch=args.follow_batch,
                            batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
    test_loader  = DataLoader(test_ds, follow_batch=args.follow_batch,
                            batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu)


    sample = train_ds[0]
    print(sample)     

    # Calculate class weights for the loss function based on the training data distribution
    print("Calculating class weights...")
    class_counts = torch.zeros(args.num_classes, dtype=torch.float64)
    empty_counts = 0
    total_voxels = 0
    fix_transform = FixVoxelLabelTransform()

    # single pass over training data to accumulate stats
    for batch in train_loader:
        batch = fix_transform(batch) 
        labels = batch.voxel_label # (Nv,)
        empty_counts += (labels == -1).sum()
        total_voxels += labels.numel()
        for i in range(args.num_classes):
            class_counts[i] += (labels == i).sum()
    
    # 1. functional vs empty -> w_mask in DDPM loss (Eq. 20)
    functional_voxels = class_counts.sum()
    mask_weights = torch.tensor([2.0, 1.0], device=device, dtype=torch.float32)
    mask_weights = mask_weights / mask_weights.mean() # normalize
    print(f"Computed Mask Weights (Empty, Func): {mask_weights}")

    # 2. between classes -> w_class in DDPM loss (Eq. 20)
    raw_weights = functional_voxels / (class_counts + 1e-6)
    class_weights = torch.sqrt(raw_weights)
    class_weights = class_weights / class_weights.mean() # normalize
    class_weights = class_weights.to(device, dtype=torch.float32)
    print(f"Computed Class Weights: {class_weights}")

    _probe = test_ds[0]
    program_input_dim = _probe.program_class_feature.size(-1) + 1
    voxel_input_dim   = _probe.voxel_feature.size(-1)
    num_classes       = args.num_classes
    hid               = args.latent_dim

    # MODULE: Text encoder (frozen T5) -- Section 3.2.3 (2), see utils/model/t2p.py
    text_encoder = T5TextEncoder(hid=hid, model_name=args.t5_model, max_len=args.t5_max_len, freeze=True).to(device).eval()
    # MODULE: full SpatialDiffusion model (diffusion network + denoising GNN + conditioning) -- see utils/model/diffusion.py
    generator = DiffusionGenerator(program_input_dim, voxel_input_dim, num_classes,
                                    hid, args.program_layer, args.voxel_layer,
                                    T=args.T, beta_start=args.beta_start, beta_end=args.beta_end, schedule='linear',
                                    text_encoder=text_encoder, cfg_scale=args.cfg_scale,
                                    class_weights=class_weights,mask_weights=mask_weights).to(device)

    if args.resume is not None:
        assert os.path.isfile(args.resume), f"Resume ckpt not found: {args.resume}"
        sd = torch.load(args.resume, map_location="cpu")
        missing, unexpected = generator.load_state_dict(sd, strict=False)
        print(f"[Resume] loaded: {args.resume}")
        if missing:   print("[Resume] missing keys:", missing)
        if unexpected:print("[Resume] unexpected keys:", unexpected)

    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=3e-6)
    torch.autograd.set_detect_anomaly(True)

    # ---------- Train Loop ----------
    for epoch in range(args.n_epochs):
        ddpm_meter = total_meter = tpr_meter = far_meter = tv_meter = 0.0
        step_cnt = 0
        generator.train()
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            prompts = getattr(batch, "prompts", None)
            # one training step: Eq. 3 (add noise) -> denoising GNN -> Eq. 20-23 (losses)
            total_loss, loss_dict = generator.forward_train(batch, prompts_per_graph=prompts)
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()

            ddpm_meter  += float(loss_dict["ddpm"].item())
            total_meter += float(loss_dict["total"].item())
            tpr_meter   += float(loss_dict["tpr"].item())
            far_meter   += float(loss_dict["far"].item())
            tv_meter    += float(loss_dict["tv_weighted"].item())
            step_cnt    += 1

        # ---- Logging (per-epoch) ----
        ddpm_avg  = ddpm_meter  / step_cnt
        total_avg = total_meter / step_cnt
        tpr_avg   = tpr_meter   / step_cnt
        far_avg   = far_meter   / step_cnt
        tv_avg    = tv_meter    / step_cnt

        print(f"[Epoch {epoch}/{args.n_epochs}] "
              f"Total={total_avg:.4f} | DDPM={ddpm_avg:.4f} | TPR={tpr_avg:.4f} | FAR={far_avg:.4f} | TV={tv_avg:.4f}")

        writer_train.add_scalar('loss/total',       total_avg, epoch)
        writer_train.add_scalar('loss/ddpm',        ddpm_avg,  epoch)
        writer_train.add_scalar('loss/tpr',         tpr_avg,   epoch)
        writer_train.add_scalar('loss/far',         far_avg,   epoch)
        writer_train.add_scalar('loss/tv_weighted', tv_avg,    epoch)

        # ----- Eval + save -----
        if epoch % args.eval_period == 0:
            generator.current_epoch = epoch
            modes_metrics, pca_embeds_node, pca_embeds_graph = evaluate_diffusion(
                generator, test_loader, ("program","text","both"), writer_train, epoch, args.viz_dir
            )
            # modes_metrics, pca_embeds_node, pca_embeds_graph = evaluate_diffusion(
            #     generator, test_loader, ["program"], writer_train, epoch, args.viz_dir
            # )
            
            for m in modes_metrics:
                print(f"=== {m} ===")
                keys = ["FAR_bias_mean","FAR_mae_mean","FAR_mse_mean",
                        "TPR_huber_mean","TPR_l1_mean"]
                pprint({k: modes_metrics[m].get(k) for k in keys})

        torch.save(generator.state_dict(), os.path.join(args.model_dir, f'{epoch}.ckpt'))
        scheduler.step() 

    writer_train.close()
