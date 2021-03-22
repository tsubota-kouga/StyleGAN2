
import copy
from datetime import datetime
import math
import os
import random
import argparse

import numpy as np
import torch
from torch import cuda, nn, optim, backends
from torch.utils.tensorboard import SummaryWriter
from torch_optimizer import AdaBelief, RAdam
from torchvision import transforms, utils
from torchvision.transforms import functional as VF
from tqdm import tqdm, trange

from hyperparam import HyperParam as hp
from loss import (
    GANDiscriminatorLoss,
    GANGeneratorLoss,
    LSGANDiscriminatorLoss,
    LSGANGeneratorLoss,
    NonSaturatingDiscriminatorLoss,
    NonSaturatingGeneratorLoss,
    RelativisticAverageHingeDiscriminatorLoss,
    RelativisticAverageHingeGeneratorLoss,
    WGANgpDiscriminatorLoss,
    WGANgpGeneratorLoss,
    r1_regularization,
    PathLengthRegularization,
)
from network import DiscriminatorV2, GeneratorV2, LatentLayers
from utils import (
    AdaptiveAugmentation,
    CelebAHQDataset,
    FFHQDataset,
    adjust_dynamic_range,
    load_generator,
    preprocess,
    update_average,
)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", help="execute preprocess", action="store_true")
    parser.add_argument("-p", "--path", help="load model", type=str, default=None)
    args = parser.parse_args()

    if args.preprocess:
        os.mkdir("./datasets")
        preprocess(hp.dataset_path[hp.dataset], hp.max_level, hp.dataroot, multi_resolution=False)

    if hp.dataset == "celeba-hq":
        dataset = CelebAHQDataset(
            root=hp.dataroot,
            attr_file=hp.attr_file,
            valid_attr=hp.valid_attr,
            multi_resolution=hp.multi_resolution,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            use_fp16=hp.use_fp16)
    else:  # hp.dataset == "ffhq" or others
        dataset = FFHQDataset(
            root=hp.dataroot,
            multi_resolution=hp.multi_resolution,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            use_fp16=hp.use_fp16)


    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=hp.batch_sizeD,
        shuffle=True,
        pin_memory=True,
        num_workers=16)

    discriminator = DiscriminatorV2(
        activation=hp.activationD,
        activation_args=hp.activationD_args,
        channel_info=hp.channel_info,
        use_scale=hp.use_scaleD,
        use_minibatch_stddev_all=hp.use_minibatch_stddev_all,
        minibatch_stddev_groups_size=hp.minibatch_stddev_groups_size,
        mode=hp.Dmode,
        eps=hp.eps).to(hp.device, non_blocking=hp.non_blocking)
    discriminator.train()

    style_mapper = LatentLayers(
        num_layers=hp.latent_layers,
        style_channels=hp.latent_dim,
        activation=hp.activationG,
        activation_args=hp.activationG_args,
        use_scale=hp.use_scaleSM,
        lrmul=hp.sm_lrmul,
        eps=hp.eps,
        num_styles=len(hp.channel_info),
        w_avg_rate=hp.w_avg_rate,
        ).to(hp.device, non_blocking=hp.non_blocking)
    style_mapper.train()

    generator = GeneratorV2(
        style_channels=hp.latent_dim,
        activation=hp.activationG,
        activation_args=hp.activationG_args,
        channel_info=hp.channel_info,
        eps=hp.eps,
        use_scale=hp.use_scaleG,
        noise_mode=hp.noise_mode,
        mode=hp.Gmode,
        ).to(hp.device, non_blocking=hp.non_blocking)
    generator.train()

    if hp.use_adaptive_discriminator_augmentation:
        data_augmentation = AdaptiveAugmentation(
            speed=hp.discriminator_augmentation_speed,
            ).to(hp.device, non_blocking=hp.non_blocking)
    else:
        data_augmentation = None


    if hp.reload:
        style_mapper, generator, discriminator = \
            load_generator(
                style_mapper,
                generator,
                discriminator,
                path=args.path)


    criterionG: nn.Module
    criterionD: nn.Module

    if hp.gan_loss == "gan":
        criterionG = GANGeneratorLoss()
        criterionD = GANDiscriminatorLoss(
            gp_param=hp.gp_param,
            multi_resolution=hp.multi_resolution,
            eps=hp.eps)
    elif hp.gan_loss == "wgangp":
        criterionG = WGANgpGeneratorLoss()
        criterionD = WGANgpDiscriminatorLoss(
            gp_param=hp.gp_param,
            drift_param=hp.drift_param,
            multi_resolution=hp.multi_resolution,
            eps=hp.eps)
    elif hp.gan_loss == "lsgan":
        criterionG = LSGANGeneratorLoss()
        criterionD = LSGANDiscriminatorLoss(
            gp_param=hp.gp_param,
            drift_param=hp.drift_param,
            multi_resolution=hp.multi_resolution,
            eps=hp.eps)
    elif hp.gan_loss == "relativistic_hinge":
        criterionG = RelativisticAverageHingeGeneratorLoss()
        criterionD = RelativisticAverageHingeDiscriminatorLoss(
            gp_param=hp.gp_param,
            drift_param=hp.drift_param,
            multi_resolution=hp.multi_resolution,
            eps=hp.eps)
    elif hp.gan_loss == "non-saturating":
        criterionG = NonSaturatingGeneratorLoss(use_unet_decoder=hp.use_unet_decoder)
        criterionD = NonSaturatingDiscriminatorLoss(
            gp_param=hp.gp_param,
            drift_param=hp.drift_param,
            r1_param=hp.r1_param,
            r1_per=hp.r1_per,
            adaptive_augmentation=data_augmentation,
            use_unet_decoder=hp.use_unet_decoder,
            multi_resolution=hp.multi_resolution,
            eps=hp.eps)
    path_length_regularization = PathLengthRegularization(eps=hp.eps, pl_param=hp.pl_param).to(hp.device)

    optimizerSM: optim.Optimizer
    optimizerG: optim.Optimizer
    optimizerD: optim.Optimizer

    if hp.optimizer == "adam":
        optimizerSM = optim.Adam(
            style_mapper.parameters(),
            lr=hp.smlr,
            eps=hp.eps,
            betas=hp.betas,
            weight_decay=hp.weight_decay)
        optimizerG = optim.Adam(
            generator.parameters(),
            lr=hp.glr,
            eps=hp.eps,
            betas=hp.betas,
            weight_decay=hp.weight_decay)
        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=hp.dlr,
            eps=hp.eps,
            betas=hp.betas)
    elif hp.optimizer == "radam":
        optimizerSM = RAdam(
            style_mapper.parameters(),
            lr=hp.smlr,
            eps=hp.eps,
            betas=hp.betas,
            weight_decay=hp.weight_decay)
        optimizerG = RAdam(
            generator.parameters(),
            lr=hp.glr,
            eps=hp.eps,
            betas=hp.betas,
            weight_decay=hp.weight_decay)
        optimizerD = RAdam(
            discriminator.parameters(),
            lr=hp.dlr,
            eps=hp.eps,
            betas=hp.betas)
    elif hp.optimizer == "adabelief":
        optimizerSM = AdaBelief(
            style_mapper.parameters(),
            lr=hp.smlr,
            eps=hp.eps,
            betas=hp.betas,
            weight_decay=hp.weight_decay)
        optimizerG = AdaBelief(
            generator.parameters(),
            lr=hp.glr,
            eps=hp.eps,
            betas=hp.betas,
            weight_decay=hp.weight_decay)
        optimizerD = AdaBelief(
            discriminator.parameters(),
            lr=hp.dlr,
            eps=hp.eps,
            betas=hp.betas)
    elif hp.optimizer == "rmsprop":
        optimizerSM = optim.RMSprop(
            style_mapper.parameters(),
            lr=hp.smlr,
            eps=hp.eps,
            weight_decay=hp.weight_decay)
        optimizerG = optim.RMSprop(
            generator.parameters(),
            lr=hp.glr,
            eps=hp.eps,
            weight_decay=hp.weight_decay)
        optimizerD = optim.RMSprop(
            discriminator.parameters(),
            lr=hp.dlr,
            eps=hp.eps)

    if hp.use_fp16:
        scaler = cuda.amp.GradScaler()

    writer = SummaryWriter(log_dir="./log")

    fixed_noise = torch.randn(
            16, hp.latent_dim, 1 , 1,
            dtype=torch.float16 if hp.use_fp16 else torch.float32,
            device=hp.device)

    global_step = 0

    if hp.move_average_rate is not None:
        style_mapper_ = copy.deepcopy(style_mapper)
        generator_ = copy.deepcopy(generator)
        style_mapper_.train()
        generator_.train()
        update_average(generator_, generator, beta=0)
        update_average(style_mapper_, style_mapper, beta=0)


    lossR1_stat, lossPL_stat = 0., 0.
    for epoch in trange(hp.num_epoch):
        for idx, imgs in enumerate(tqdm(dataloader)):
            b_size = imgs.shape[0]

            if b_size % hp.minibatch_stddev_groups_size != 0:
                continue

            with cuda.amp.autocast(enabled=hp.use_fp16):
                imgs = imgs.to(hp.device, non_blocking=hp.non_blocking).requires_grad_(True)
                noise = torch.randn(
                    b_size, hp.latent_dim, hp.n_mix, 1,
                    dtype=torch.float16 if hp.use_fp16 else torch.float32,
                    device=hp.device)

                styles = style_mapper(
                    noise,
                    mixing_regularization_rate=hp.mixing_regularization_rate)
                fakes = generator(styles=styles)[-1]

                if hp.use_adaptive_discriminator_augmentation:
                    with cuda.amp.autocast(False):
                        if hp.use_unet_decoder:
                            imgs, fakes = data_augmentation(imgs.float(), fakes.float())
                        else:
                            imgs, = data_augmentation(imgs.float())
                            fakes, = data_augmentation(fakes.float())

            # Discriminator train
            with cuda.amp.autocast(enabled=hp.use_fp16):
                discriminator.train()
                for p in discriminator.parameters():
                    p.requires_grad = True

                lossD = criterionD(discriminator, fakes.detach(), imgs.detach(), scaler)

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
                lossD_item = lossD.item()
                del lossD
                cuda.synchronize(hp.device)

                if global_step % hp.r1_per == 0:
                    lossR1 = hp.r1_param * r1_regularization(
                            discriminator=discriminator,
                            reals=imgs,
                            eps=hp.eps,
                            scaler=scaler)
                    discriminator.zero_grad(set_to_none=True)
                    if hp.use_fp16:
                        scaler.scale(lossR1).backward()
                        scaler.step(optimizerD)
                        scaler.update()
                    else:
                        lossR1.backward()
                        optimizerD.step()
                    lossR1_item = lossR1.item()
                    del lossR1
                    cuda.synchronize(hp.device)
                    lossR1_stat = lossR1_item

                lossD = lossD_item
                cuda.empty_cache()

            # Generator train
            with cuda.amp.autocast(enabled=hp.use_fp16):
                discriminator.eval()
                lossG = criterionG(discriminator, fakes=fakes, reals=imgs)
                style_mapper.zero_grad(set_to_none=True)
                generator.zero_grad(set_to_none=True)

                if hp.use_fp16:
                    scaler.scale(lossG).backward()
                    scaler.step(optimizerSM)
                    scaler.step(optimizerG)
                    scaler.update()
                else:
                    lossG.backward()
                    optimizerSM.step()
                    optimizerG.step()

                if hp.move_average_rate is not None:
                    update_average(style_mapper_, style_mapper, hp.move_average_rate)
                    update_average(generator_, generator, hp.move_average_rate)
                lossG_item = lossG.item()
                del lossG
                cuda.synchronize(hp.device)

                if global_step % hp.path_length_per == 0:
                    noise = torch.randn(
                        b_size, hp.latent_dim, hp.n_mix, 1,
                        dtype=torch.float16 if hp.use_fp16 else torch.float32,
                        device=hp.device)
                    styles = style_mapper(noise, mixing_regularization_rate=hp.n_mix)
                    fakes = generator(styles=styles)[-1]
                    lossPL = path_length_regularization(
                            fakes=fakes,
                            latent=styles,
                            scaler=scaler)

                    generator.zero_grad(set_to_none=True)

                    if hp.use_fp16:
                        scaler.scale(lossPL).backward()
                        scaler.step(optimizerG)
                        scaler.update()
                    else:
                        lossPL.backward()
                        optimizerG.step()
                    lossPL_item = lossPL.item()
                    del lossPL
                    cuda.synchronize(hp.device)
                    lossPL_stat = lossPL_item
                lossG = lossG_item
                cuda.empty_cache()

            # LOG
            def evaluate():
                style_mapper.eval()
                generator.eval()
                writer.add_scalar(
                    "discriminator loss",
                    lossD,
                    global_step)
                writer.add_scalar(
                    "discriminator R1",
                    lossR1_stat,
                    global_step)
                writer.add_scalar(
                    "generator loss",
                    lossG,
                    global_step)
                writer.add_scalar(
                    "generator PL",
                    lossPL_stat,
                    global_step)
                if hp.use_adaptive_discriminator_augmentation:
                    writer.add_scalar(
                        "data_augmentation p",
                        data_augmentation.p,
                        global_step)
                    writer.add_scalar(
                        "rt",
                        criterionD.rt,
                        global_step)
                with torch.no_grad():
                    with cuda.amp.autocast(enabled=hp.use_fp16):
                        if hp.move_average_rate is not None:
                            styles = style_mapper_(fixed_noise)
                            fakes = generator_(styles=styles)
                        else:
                            styles = style_mapper(fixed_noise)
                            fakes = generator(styles=styles)
                        if hp.use_unet_decoder:
                            _, fake_unet_out = discriminator(fakes[-1], unet_out=True)
                    fakes = [ adjust_dynamic_range(fake.float()) for fake in fakes ]
                    for idx, fake in enumerate(fakes):
                        fake = fake.detach().cpu()
                        if hp.Gmode == "wavelet":
                            size = 2 ** (idx + 3)
                        else:
                            size = 2 ** (idx + 2)
                        writer.add_image(
                            f"generated imgs {size}x{size}",
                            utils.make_grid(
                                fake,
                                nrow=4,
                                padding=1,
                                normalize=True),
                            global_step)
                    if hp.use_unet_decoder:
                        writer.add_image(
                            f"unet out imgs {size}x{size}",
                            utils.make_grid(
                                torch.sigmoid(fake_unet_out),
                                padding=1, nrow=4, normalize=False, range=(0, 1)),
                            global_step)
            if global_step % 50 == 0:
                evaluate()
                cuda.empty_cache()

            if global_step % 500 == 0:
                t = datetime.today()
                torch.save({
                    "style_mapper": style_mapper.state_dict(),
                    "generator": generator.state_dict(),
                    "style_mapper_": hp.move_average_rate and style_mapper_.state_dict(),
                    "generator_": hp.move_average_rate and generator_.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "optimizerG": optimizerG.state_dict(),
                    "optimizerD": optimizerD.state_dict(),
                    "optimizerSM": optimizerSM.state_dict(),
                    "path_length_regularization": path_length_regularization.state_dict(),
                }, f"./model/{t}.model")
            global_step += 1

    writer.close()
