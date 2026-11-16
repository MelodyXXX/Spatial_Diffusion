import os
import datetime

class Args:
    # Base
    cuda = '0'
    comment = 'diff_new'

    # Data
    batch_size = 96
    train_size = 96000 # 96000
    test_size = 256
    n_cpu = 24
    
    # Model
    latent_dim = 256
    program_layer = 4
    voxel_layer = 12
    num_classes = 6

    # Diffusion
    T = 1000
    beta_start = 1e-4
    beta_end = 0.02

    # Train
    n_epochs = 200
    eval_period = 5

    # Optim
    lr = 2e-4
    b1 = 0.9
    b2 = 0.99

    # Text & CFG
    t5_model = 't5-base'
    t5_max_len = 512
    cfg_scale = 2.0

    # Program-drop
    eval_disable_mask = False
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    run_id = current_time + "_" + comment
    run_dir = "runs"
    viz_dir = os.path.join(run_dir, run_id, "output")
    model_dir = os.path.join(run_dir, run_id, "checkpoints")
    resume =None
    # resume = "/data/melody/buildinggan/Building_GAN/runs/Oct30_17-17-32_diff_new/checkpoints/30.ckpt"
    dir_graph = "Dataset/6types-processed_data_caption_v2"
    follow_batch = ['program_target_ratio', 'program_class_feature', 'voxel_feature']

