# infer_all_noargs.py
# -*- coding: utf-8 -*-

import os, datetime
import torch
from torch_geometric.data import DataLoader
from utils.eval.eval import evaluate
from utils.data.LargeFilenameDataset import LargeFilenameDataset
from utils.model.modelsOriginal import Generator
from tensorboardX import SummaryWriter
from utils.model.t2p import T5TextEncoder
from utils.data.caption import _common_filenames,CaptionAugmentedDataset, FixVoxelLabelTransform

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1" 
from utils.model.diffusion import DiffusionGenerator
from utils.eval.evaldi import evaluate_diffusion
from utils.args import Args
from utils.data.caption import ensure_dir



if __name__ == "__main__":
    args = Args()
    device_ids = 0
    device = torch.device('cuda:0')

    all_fns = _common_filenames(args.dir_graph, take_all_graph=False)
    train_size = args.train_size
    test_size  = 4000
    batch_size = 40 #args.batch_size
    test_size  = test_size  // batch_size * batch_size
    test_fns   = all_fns[train_size: train_size + test_size]
 
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    INFER_DIR = ensure_dir(os.path.join("inference", ts)) 
    DIFFUSION_CKPT = "/data/melody/buildinggan/Building_GAN/runs/May17_18-35-56_diff_new/checkpoints/185.ckpt"
    GAN_CKPT = "/data/melody/buildinggan/Building_GAN/runs/iccv2021/checkpoints/70.ckpt"
    out_root = ensure_dir(os.path.join(INFER_DIR, f"diffusion"))
    epoch = 0
    fix_transform = FixVoxelLabelTransform()

    # Diffusion
    test_ds_diffusion = CaptionAugmentedDataset(args.dir_graph, test_fns, transform=fix_transform)
    test_loader_diffusion = DataLoader(test_ds_diffusion, follow_batch=args.follow_batch, batch_size=batch_size,
                             shuffle=False, num_workers=args.n_cpu)

    _probe = test_ds_diffusion[0]
    program_input_dim = _probe.program_class_feature.size(-1) + 1
    voxel_input_dim   = _probe.voxel_feature.size(-1)
    num_classes       = args.num_classes
    hid               = args.latent_dim
    
    print("Calculating weights (Skipped for brevity, using defaults if not recalc)...")
    mask_weights = torch.tensor([0.5, 1.5], device=device)
    class_weights = torch.ones(args.num_classes, device=device)

    text_encoder = T5TextEncoder(hid=hid, model_name=args.t5_model, max_len=args.t5_max_len, freeze=True).to(device).eval()
    generator_diffusion = DiffusionGenerator(program_input_dim, voxel_input_dim, args.num_classes, args.latent_dim,
        args.program_layer, args.voxel_layer, args.T, args.beta_start, args.beta_end, 'linear',
        text_encoder, args.cfg_scale, class_weights, mask_weights).to(device)

    generator_diffusion.load_state_dict(torch.load(DIFFUSION_CKPT, map_location=device), strict=False)
    writer_train = SummaryWriter(log_dir=os.path.join(out_root,"writer"))
    generator_diffusion.eval()
    modes_metrics, *_ = evaluate_diffusion(
        generator_diffusion, test_loader_diffusion, ("program","text","both"), writer_train, epoch, out_root)

    # GAN
    out_root     = ensure_dir(os.path.join(INFER_DIR, f"gan"))
    test_ds_gan = LargeFilenameDataset(args.dir_graph, test_fns)
    test_loader_gan = DataLoader(test_ds_gan, follow_batch=args.follow_batch, batch_size=batch_size,
                             shuffle=False, num_workers=args.n_cpu)

    program_input_dim = _probe.program_class_feature.size(-1) + 1
    voxel_input_dim = _probe.voxel_feature.size(-1)
    noise_dim = 32
    program_layer = 4
    voxel_layer = 12
    latent_dim = 128
    generator_gan = Generator(program_input_dim, voxel_input_dim, latent_dim,
                            noise_dim, program_layer, voxel_layer, device).to(device).eval()

    generator_gan.load_state_dict(torch.load(GAN_CKPT, map_location=device), strict=False)
    writer_train = SummaryWriter(log_dir=os.path.join(out_root,"writer"))

    gan_metrics, *_ = evaluate(
        test_loader_gan, generator_gan, out_root,
        args.follow_batch, device_ids
    )

    print(f"Done! saved to {INFER_DIR}")
