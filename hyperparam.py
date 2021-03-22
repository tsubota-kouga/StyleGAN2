import torch


class HyperParam:
    dataroot = "./datasets/"
    dataset = "afhq-dog"
    dataset_path = {
        "celeba-hq": "/mnt/My Files/celeba-hq",
        "ffhq": "/mnt/My Files/ffhq",
        "afhq-dog": "/mnt/My Files/afhq/train/dog/"
        }
    if dataset == "celeba-hq":
        # attr_file = "/mnt/My Files/celeba-hq/Anno/list_attr_celeba.txt"
        attr_file = None
        valid_attr = ["Male", "No_Beard", "Wearing_Hat"]
    else:
        attr_file = None
    batch_sizeD = 24
    batch_sizeG = 24
    use_fp16 = True
    reload = False
    drift_param = None  # 0.001
    gp_param = None  # 10.0
    r1_param = 0.5
    r1_per = 16
    path_length_per = 8
    pl_param = 2.
    gan_loss = "non-saturating"
    optimizer = "adam"
    multi_resolution = False
    d_loss_threshold = None  # 0.0
    g_loss_threshold = None  # 50.0
    critic_range = 8
    nd_critic = 1
    ng_critic = 1
    smlr = 2.5e-3
    dlr = 2.5e-3
    glr = 2.5e-3
    if r1_per is not None:
        dlr = dlr * (r1_per - 1) / r1_per
    dlr_decay = 1  # 0.99999
    glr_decay = 1  # 0.99999
    move_average_rate = 0.995
    non_blocking = False
    discriminator_augmentation_speed = 5e-3
    use_adaptive_discriminator_augmentation = True
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
    latent_layers = 8
    betas = (0.0, 0.99)
    if r1_per is not None:
        betas = (betas[0] * (r1_per - 1) / r1_per, betas[1] * (r1_per - 1) / r1_per)
    eps = 1e-6 if use_fp16 else 1e-8
    device = torch.device("cuda")
    num_epoch = 5000
    weight_decay = 0
    Gmode = "wavelet"  # [wavelet, skip]
    Dmode = "wavelet"  # [wavelet, skip, resnet]
    max_level = len(channel_info) if Gmode != "wavelet" else len(channel_info) + 1
    use_scaleSM = True
    sm_lrmul = 0.01
    use_scaleG = True
    use_scaleD = True
    use_minibatch_stddev_all = False
    use_unet_decoder = False
    minibatch_stddev_groups_size = 4
    assert batch_sizeD % minibatch_stddev_groups_size == 0
    assert batch_sizeG % minibatch_stddev_groups_size == 0
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

