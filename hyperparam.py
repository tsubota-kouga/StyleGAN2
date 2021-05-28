import torch


class HyperParam:
    dataroot = "./datasets/"
    dataset = "afhq-dog"
    dataset_path = {
        "celeba-hq": "/mnt/My Files/celeba-hq",
        "ffhq": "/mnt/My Files/ffhq",
        "afhq-dog": "/mnt/My Files/afhq/train/dog/"
        }
    log_dir = "./log"
    profile_dir = "./profile"
    model_dir = "/mnt/My Files/stylegan-model/"
    batch_sizeD = 16
    batch_sizeG = 16
    regularize_batch_sizeD = 8
    regularize_batch_sizeG = 8
    use_fp16 = False
    reload = False
    drift_param = None  # 0.001
    gp_param = None  # 10.0

    r1_per = 16
    r1_param = 0.0002 * 256 ** 2 / batch_sizeD  # 0.0002 * resolution ** 2 / B

    path_length_per = 8
    path_length_param = 2.

    regularize_with_main_loss = True

    gan_loss = "non-saturating"
    optimizer = "adam"
    smlr = 2.5e-3
    dlr = 2.5e-3
    glr = 2.5e-3
    if r1_per is not None:
        dlr = dlr * (r1_per - 1) / r1_per
    move_average_rate = 0.995
    non_blocking = False
    discriminator_augmentation_speed = 2e-6  # 1 / (500 * 10^3)
    use_adaptive_discriminator_augmentation = True
    use_contrastive_discriminator = False
    projection_dim = 512
    truncation_trick_rate = 0.7
    w_avg_rate = 0.995
    n_mix = 2
    mixing_regularization_rate = 0.5
    noise_mode = "random"  # ["const-deterministic", "const-random", "deterministic", "random"]
    latent_dim = 256
    channel_info = [
        (latent_dim, latent_dim),  # 4x4
        (latent_dim, latent_dim),  # 8x8
        (latent_dim, latent_dim),  # 16x16
        # (latent_dim, latent_dim),  # 32x32
        (latent_dim, 256),  # 64x64
        (256, 128),  # 128x128
        (128, 64),  # 256x256
        # (64, 32),  # 512x512
        # (32, 16)
        ]
    latent_layers = 7
    betas = (0.0, 0.99)
    # betas = (0.5, 0.999)
    if r1_per is not None:
        betas = (betas[0] * (r1_per - 1) / r1_per, betas[1] * (r1_per - 1) / r1_per)
    # eps = 1e-6 if use_fp16 else 1e-8
    eps = 1e-8
    device = torch.device("cuda")
    num_epoch = 5000
    weight_decay = 0
    Gmode = "wavelet"  # [wavelet, skip]
    Dmode = "wavelet"  # [wavelet, skip, resnet]
    max_level = len(channel_info) if Gmode != "wavelet" else len(channel_info) + 1
    resolution = 2 ** (max_level + 1)
    use_scaleSM = True
    sm_lrmul = 0.01
    use_scaleG = True
    use_scaleD = True
    use_minibatch_stddev_all = False
    use_unet_decoder = False
    minibatch_stddev_groups_size = 4
    assert batch_sizeD % minibatch_stddev_groups_size == 0
    assert batch_sizeG % minibatch_stddev_groups_size == 0
    assert regularize_batch_sizeD % minibatch_stddev_groups_size == 0
    assert regularize_batch_sizeG % minibatch_stddev_groups_size == 0
    activationG = "leaky_relu"
    activationG_args = {
        "negative_slope": 0.2,
        "use_scale": True,
        }
    activationD = "leaky_relu"
    activationD_args = {
        "negative_slope": 0.2,
        "use_scale": True,
        }

