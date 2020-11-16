import torch


class HyperParam:
    dataroot = "./datasets/"
    dataset = "celeba-hq"  # "ffhq"
    dataset_path = {
        "celeba-hq": "/mnt/My Files/celeba-hq",
        "ffhq": "/mnt/My Files/ffhq",
        }
    if dataset == "celeba-hq":
        # attr_file = "/mnt/My Files/celeba-hq/Anno/list_attr_celeba.txt"
        attr_file = None
        valid_attr = ["Male", "No_Beard", "Wearing_Hat"]
    else:
        attr_file = None
    batch_size = 32
    use_fp16 = True
    reload = False
    drift_param = None  # 0.001
    gp_param = None  # 10.0
    gan_loss = "relativistic_hinge"
    optimizer = "adam"
    critic_range = 8
    d_loss_threshold = None  # 0.0
    g_loss_threshold = None  # 50.0
    nd_critic = 1
    ng_critic = 1
    dlr = 3e-3
    glr = 3e-3
    dlr_decay = 1  # 0.99999
    glr_decay = 1  # 0.99999
    move_average_rate = None #  0.999
    non_blocking = False
    dropout = 0.0
    leakiness = 0.2
    latent_dim = 512
    channel_info = [
        (latent_dim, latent_dim),  # 4x4
        (latent_dim, latent_dim),  # 8x8
        (latent_dim, latent_dim),  # 16x16
        (latent_dim, latent_dim),  # 32x32
        (latent_dim, 256),  # 64x64
        (256, 128),  # 128x128
        # (128, 64),  # 256x256
        # (64, 32),  # 512x512
        ]
    beta1 = 0.0
    device = torch.device("cuda")
    num_epoch = 999
    weight_decay = 0
    max_level = len(channel_info)
    use_tanh = False
    use_sigmoid = False
    activationG = "leaky_relu"
    activationD = "leaky_relu"

