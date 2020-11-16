
import sys
import torch
import copy
import time
from torch import nn, optim
from torch_optimizer import RAdam, AdaBelief
from torchvision import datasets, transforms, utils
from network import Discriminator, Generator
from loss import \
    GANDiscriminatorLoss, \
    GANGeneratorLoss, \
    WGANgpDiscriminatorLoss, \
    WGANgpGeneratorLoss, \
    LSGANDiscriminatorLoss, \
    LSGANGeneratorLoss, \
    RelativisticAverageHingeDiscriminatorLoss, \
    RelativisticAverageHingeGeneratorLoss
from torch.utils.tensorboard import SummaryWriter
from utils import weights_init, \
        FFHQDataset, CelebAHQDataset, \
        preprocess, \
        hypersphere, \
        update_average, adjust_dynamic_range, \
        load_generator
from tqdm import tqdm, trange
from datetime import datetime
from hyperparam import HyperParam as hp


# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

if len(sys.argv) == 2 and sys.argv[1] == "preprocess":
    preprocess(hp.dataset_path[hp.dataset], hp.max_level, hp.dataroot)

if hp.dataset == "celeba-hq":
    dataset = CelebAHQDataset(
        root=hp.dataroot,
        attr_file=hp.attr_file,
        valid_attr=hp.valid_attr,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        use_fp16=hp.use_fp16)
elif hp.dataset == "ffhq":
    dataset = FFHQDataset(
        root=hp.dataroot,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        use_fp16=hp.use_fp16)
else:
    assert False


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=hp.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=16)

discriminator = Discriminator(
    leakiness=hp.leakiness,
    activation=hp.activationD,
    channel_info=hp.channel_info,
    use_sigmoid=hp.use_sigmoid,
    v=2).to(hp.device, non_blocking=hp.non_blocking)
discriminator.train()

generator = Generator(
    leakiness=hp.leakiness,
    activation=hp.activationG,
    channel_info=hp.channel_info,
    use_tanh=hp.use_tanh).to(hp.device, non_blocking=hp.non_blocking)
generator.train()

if hp.reload:
    generator, discriminator = load_generator(generator, discriminator, path=sys.argv[1] if len(sys.argv) > 1 else None)
# else:
#     generator.apply(weights_init)
#     discriminator.apply(weights_init)


if hp.gan_loss == "gan":
    criterionG = GANGeneratorLoss()
    criterionD = GANDiscriminatorLoss(hp.gp_param)
elif hp.gan_loss == "wgangp":
    criterionG = WGANgpGeneratorLoss()
    criterionD = WGANgpDiscriminatorLoss(
        gp_param=hp.gp_param,
        drift_param=hp.drift_param)
elif hp.gan_loss == "lsgan":
    criterionG = LSGANGeneratorLoss()
    criterionD = LSGANDiscriminatorLoss(
        gp_param=hp.gp_param,
        drift_param=hp.drift_param)
elif hp.gan_loss == "relativistic_hinge":
    criterionG = RelativisticAverageHingeGeneratorLoss()
    criterionD = RelativisticAverageHingeDiscriminatorLoss(
        gp_param=hp.gp_param,
        drift_param=hp.drift_param)


if hp.optimizer == "adam":
    optimizerG = optim.Adam(
        generator.parameters(),
        lr=hp.glr,
        eps=1e-8,
        betas=(hp.beta1, 0.99),
        weight_decay=hp.weight_decay)
    optimizerD = optim.Adam(
        discriminator.parameters(),
        lr=hp.dlr,
        eps=1e-8,
        betas=(hp.beta1, 0.99))
elif hp.optimizer == "radam":
    optimizerG = RAdam(
        generator.parameters(),
        lr=hp.glr,
        eps=1e-8,
        betas=(hp.beta1, 0.99),
        weight_decay=hp.weight_decay)
    optimizerD = RAdam(
        discriminator.parameters(),
        lr=hp.dlr,
        eps=1e-8,
        betas=(hp.beta1, 0.99))
elif hp.optimizer == "adablief":
    optimizerG = AdaBelief(
        generator.parameters(),
        lr=hp.glr,
        eps=1e-8,
        betas=(hp.beta1, 0.99),
        weight_decay=hp.weight_decay)
    optimizerD = AdaBelief(
        discriminator.parameters(),
        lr=hp.dlr,
        eps=1e-8,
        betas=(hp.beta1, 0.99))
elif hp.optimizer == "rmsprop":
    optimizerG = optim.RMSprop(
        generator.parameters(),
        lr=hp.glr,
        eps=1e-8,
        weight_decay=hp.weight_decay)
    optimizerD = optim.RMSprop(
        discriminator.parameters(),
        lr=hp.dlr,
        eps=1e-8)


if hp.use_fp16:
    scaler = torch.cuda.amp.GradScaler()

writer = SummaryWriter(log_dir="./log")

fixed_noise = hypersphere(
    torch.randn(
        16, hp.latent_dim, 1, 1,
        dtype=torch.float16 if hp.use_fp16 else torch.float32,
        device=hp.device))

global_step = 0

if hp.move_average_rate is not None:
    generator_ = copy.deepcopy(generator)
    generator_.eval()
    update_average(generator_, generator, beta=0)


for epoch in trange(hp.num_epoch):
    for idx, imgs in enumerate(tqdm(dataloader)):

        try:
            imgs = [img.to(hp.device, non_blocking=hp.non_blocking)
                    for img in imgs[:hp.max_level]]

            b_size = imgs[0].shape[0]

            def discriminator_train():
                if hp.gp_param is None:
                    for p in generator.parameters():
                        p.requires_grad = False
                for p in discriminator.parameters():
                    p.requires_grad = True

                noise = hypersphere(
                    torch.randn(
                        b_size, hp.latent_dim, 1, 1,
                        dtype=torch.float16 if hp.use_fp16 else torch.float32,
                        device=hp.device))

                if hp.use_fp16:
                    with torch.cuda.amp.autocast():
                        fakes = generator(noise)
                        lossD = criterionD(discriminator, fakes, imgs, scaler)
                else:
                    fakes = generator(noise)
                    lossD = criterionD(discriminator, fakes, imgs)

                if type(lossD) is tuple:
                    lossD = sum(lossD)

                discriminator.zero_grad(set_to_none=True)

                if hp.use_fp16:
                    scaler.scale(lossD).backward()
                    scaler.step(optimizerD)
                    scaler.update()
                else:
                    lossD.backward()
                    optimizerD.step()
                loss = lossD.item()
                del lossD
                return loss

            if idx % hp.nd_critic == 0:
                lossD = discriminator_train()
            # torch.cuda.empty_cache()

            def generator_train():
                for p in generator.parameters():
                    p.requires_grad = True
                for p in discriminator.parameters():
                    p.requires_grad = False

                noise = hypersphere(
                    torch.randn(
                        b_size, hp.latent_dim, 1, 1,
                        dtype=torch.float16 if hp.use_fp16 else torch.float32,
                        device=hp.device))

                if hp.use_fp16:
                    with torch.cuda.amp.autocast():
                        fakes = generator(noise)
                        lossG = criterionG(discriminator, fakes, imgs)
                else:
                    fakes = generator(noise)
                    lossG = criterionG(discriminator, fakes, imgs)
                generator.zero_grad(set_to_none=True)

                if hp.use_fp16:
                    scaler.scale(lossG).backward()
                    scaler.step(optimizerG)
                    scaler.update()
                else:
                    lossG.backward()
                    optimizerG.step()

                if hp.move_average_rate is not None:
                    update_average(generator_, generator, hp.move_average_rate)
                loss = lossG.item()
                del lossG
                return loss

            if idx % hp.ng_critic == 0:
                lossG = generator_train()
            for img in imgs:
                del img
            # torch.cuda.empty_cache()

            # LOG
            def evaluate():
                writer.add_scalar(
                    "discriminator loss",
                    lossD,
                    global_step)
                writer.add_scalar(
                    "generator loss",
                    lossG,
                    global_step)
                with torch.no_grad():
                    if hp.move_average_rate is not None:
                        if hp.use_fp16:
                            with torch.cuda.amp.autocast():
                                fakes = generator_(fixed_noise)
                        else:
                            fakes = generator_(fixed_noise)
                    else:
                        if hp.use_fp16:
                            with torch.cuda.amp.autocast():
                                fakes = generator(fixed_noise)
                        else:
                            fakes = generator(fixed_noise)
                    fakes = [ adjust_dynamic_range(fake.float()) for fake in fakes ]
                    for idx, fake in enumerate(fakes):
                        fake = fake.detach().cpu()
                        size = 2 ** (idx + 2)
                        writer.add_image(
                            f"generated imgs {size}x{size}",
                            utils.make_grid(
                                fake,
                                padding=1,
                                normalize=True),
                            global_step)
            if global_step % 50 == 0:
                evaluate()
                torch.cuda.empty_cache()

            if global_step % 1000 == 0:
                t = datetime.today()
                torch.save({
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                }, f"./model/{t}.model")
            global_step += 1
        except RuntimeError as e:
            print(e)
            t = datetime.today()
            print(f"save as {t}.model")
            torch.save({
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizerG": optimizerG.state_dict(),
                "optimizerD": optimizerD.state_dict()
            }, f"./model/{t}.model")
            time.sleep(10)
            torch.cuda.empty_cache()


writer.close()
