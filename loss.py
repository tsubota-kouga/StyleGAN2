
import math
from typing import Any, List, Optional, Union, Tuple

import numpy as np
import torch
from torch import autograd, nn, cuda
from torch.nn import functional as F
from torchvision import transforms

from utils import AdaptiveAugmentation, SimCLRAugmentation


def _gradiend_penalty(
        discriminator: nn.Module,
        fake: torch.Tensor,
        real: torch.Tensor,
        eps: Optional[float] = None, scaler=None):
    batch_size = fake.shape[0]
    weight = torch.rand(
            batch_size, 1, 1, 1,
            dtype=fake.dtype,
            device=fake.device)
    interpolate = (fake * (1 - weight) + real * weight).requires_grad_(True)
    disc_interpolate = discriminator(interpolate)
    if scaler is not None:
        disc_interpolate = scaler.scale(disc_interpolate)
        inv_scale = 1. / (scaler.get_scale() + eps)
    grad = autograd.grad(
        outputs=disc_interpolate.sum(),
        inputs=interpolate,
        grad_outputs=torch.ones_like(disc_interpolate),
        create_graph=True)[0]
    if scaler is not None:
        grad = grad * inv_scale
    grad_penalty_loss = (grad.view(batch_size, -1)
                             .norm(2, dim=1) - 1).square().mean()
    return grad_penalty_loss


def _gradiend_penalty_for_multi_scale(
        discriminator,
        fakes, reals,
        eps: Optional[float] = None,
        scaler=None):
    batch_size = fakes[0].shape[0]
    weight = torch.rand(
            batch_size, 1, 1, 1,
            dtype=fakes[0].dtype,
            device=fakes[0].device)
    interpolates = [fake * (1 - weight) + real * weight
                    for fake, real in zip(fakes, reals)]
    interpolates = list(map(lambda x: x.requires_grad_(True), interpolates))
    disc_interpolates = discriminator(interpolates)
    if scaler is not None:
        disc_interpolates = scaler.scale(disc_interpolates)
        inv_scale = 1. / (scaler.get_scale() + eps)
    grad = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True)
    grad_penalty_loss = 0.0
    for g in grad:
        if scaler is not None:
            g = g * inv_scale
        grad_penalty_loss += (g.view(batch_size, -1)
                               .norm(2, dim=1) - 1).square().mean()
    grad_penalty_loss /= len(grad)
    return grad_penalty_loss


def gradiend_penalty(discriminator, fakes, reals, scaler, multi_resolution: bool, eps: Optional[float] = None):
    if not multi_resolution:
        return _gradiend_penalty(discriminator, fakes, reals, eps, scaler)
    else:
        return _gradiend_penalty_for_multi_scale(discriminator, fakes, reals, eps, scaler)


class R1Regularization(nn.Module):
    def __init__(self,
                 eps: Optional[float] = None,
                 resolution: Optional[int] = None,
                 use_contrastive_discriminator: bool = False):
        super(R1Regularization, self).__init__()
        self.eps = eps
        if use_contrastive_discriminator:
            assert resolution is not None
            self.transform = SimCLRAugmentation(resolution=resolution)

    def forward(self,
                reals: torch.Tensor, *,
                discriminator: nn.Module,
                reals_out: Optional[torch.Tensor] = None,
                scaler: Optional[cuda.amp.GradScaler] = None) -> torch.Tensor:

        if reals_out is None:  # re-compute
            reals_out = discriminator(reals, r1_regularize=True)

        if scaler is not None:
            reals_out = scaler.scale(reals_out)
            inv_scale = 1. / (scaler.get_scale() + self.eps)
        grad = autograd.grad(
            outputs=reals_out.sum(),
            inputs=reals,
            create_graph=True)[0]
        if scaler is not None:
            grad = grad * inv_scale
        r1 = grad.flatten(start_dim=1).square().sum(dim=1).mean()
        return r1

def r1_regularization(
    discriminator: nn.Module,
    reals: torch.Tensor,
    eps: float,
    reals_out: Optional[torch.Tensor] = None,
    scaler: Optional[cuda.amp.GradScaler] = None):

    if reals_out is None:  # re-compute
        # reals.requires_grad = True
        reals_out = discriminator(reals, r1_regularize=True)

    if scaler is not None:
        reals_out = scaler.scale(reals_out)
        inv_scale = 1. / (scaler.get_scale() + eps)
    grad = autograd.grad(
        outputs=reals_out.sum(),
        inputs=reals,
        create_graph=True)[0]
    if scaler is not None:
        grad = grad * inv_scale
    r1 = grad.flatten(start_dim=1).square().sum(dim=1).mean()
    return r1


class PathLengthRegularization(nn.Module):
    def __init__(self,
                 eps: float,
                 coefficient: float = 0.99,
                 pl_param: Optional[float] = None):
        super(PathLengthRegularization, self).__init__()
        self.register_buffer("moving_average", torch.zeros([1]))
        self.coefficient = coefficient
        self.eps = eps
        self.pl_param = pl_param

    def forward(self,
                fakes: torch.Tensor,
                latent: List[torch.Tensor],
                scaler: Optional[cuda.amp.GradScaler] = None) -> torch.Tensor:
        noise = torch.randn_like(fakes) / math.sqrt(math.prod(fakes.shape[-2:]))
        outputs = (noise * fakes).sum()

        if self.pl_param is None:
            resolution = fakes.shape[-1]
            pl_param = math.log(2) / ((resolution ** 2) * (math.log(resolution) - math.log(2)))
        else:
            pl_param = self.pl_param

        if scaler is not None:
            outputs = scaler.scale(outputs)
            inv_scale = 1. / (scaler.get_scale() + self.eps)
        grad = autograd.grad(
                outputs=outputs,
                inputs=latent,
                create_graph=True)[0]
        if scaler is not None:
            grad = grad * inv_scale
        path_length = grad.square().sum(dim=2).mean(dim=1).sqrt()
        penalty = (self.moving_average - path_length).square().mean()
        self.moving_average.data = (
                path_length.mean() * (1 - self.coefficient) +
                self.moving_average * self.coefficient)
        return penalty * pl_param


class GANDiscriminatorLoss(nn.Module):
    def __init__(self, gp_param: Optional[float], eps: Optional[float] = None, multi_resolution: bool = False):
        super(GANDiscriminatorLoss, self).__init__()
        self.criterion = nn.BCELoss()
        self.gp_param = gp_param
        self.multi_resolution = multi_resolution
        self.eps = eps

    def forward(self, discriminator: nn.Module, fakes, reals, scaler=None):
        if not self.multi_resolution:
            batch_size = reals.shape[0]
            labels = torch.full((batch_size, ), 1.0).to(reals.device)
        else:
            batch_size = reals[0].shape[0]
            labels = torch.full((batch_size, ), 1.0).to(reals[0].device)
        reals_out = discriminator(reals).view(-1)
        lossD = self.criterion(reals_out, labels)
        if self.gp_param is not None:
            grad_penalty_loss = self.gp_param * gradiend_penalty(
                discriminator=discriminator,
                fakes=fakes, reals=reals,
                scaler=scaler,
                multi_resolution=self.multi_resolution,
                eps=self.eps)
        else:
            grad_penalty_loss = 0.0
        return lossD, grad_penalty_loss


class GANGeneratorLoss(nn.Module):
    def __init__(self):
        super(GANGeneratorLoss, self).__init__()

    def forward(self, discriminator, fakes, reals):
        return -discriminator(fakes).mean()


class WGANgpDiscriminatorLoss(nn.Module):
    def __init__(self,
                 gp_param: Optional[float],
                 drift_param: Optional[float],
                 eps: Optional[float] = None,
                 multi_resolution: bool = False):
        super(WGANgpDiscriminatorLoss, self).__init__()
        self.gp_param = gp_param
        self.drift_param = drift_param
        self.multi_resolution = multi_resolution
        self.eps = eps

    def forward(self,
                discriminator: nn.Module,
                fakes: Union[torch.Tensor, List[torch.Tensor]],
                reals: Union[torch.Tensor, List[torch.Tensor]],
                scaler: Optional[cuda.amp.GradScaler] = None):
        fakes_out = discriminator(fakes)
        reals_out = discriminator(reals)
        wgan_loss = fakes_out.mean() - reals_out.mean()

        if self.gp_param is not None:
            grad_penalty_loss = self.gp_param * gradiend_penalty(
                discriminator=discriminator,
                fakes=fakes,
                reals=reals,
                scaler=scaler,
                multi_resolution=self.multi_resolution,
                eps=self.eps)
        else:
            grad_penalty_loss = 0.0

        if self.drift_param is not None:
            drift_loss = self.drift_param * reals_out.square().mean()
        else:
            drift_loss = 0.0

        return wgan_loss, grad_penalty_loss, drift_loss


class WGANgpGeneratorLoss(nn.Module):
    def __init__(self):
        super(WGANgpGeneratorLoss, self).__init__()

    def forward(self,
                discriminator: nn.Module,
                fakes: Union[torch.Tensor, List[torch.Tensor]],
                reals: Union[torch.Tensor, List[torch.Tensor]]):
        loss = -discriminator(fakes)
        return loss.mean()


class LSGANDiscriminatorLoss(nn.Module):
    def __init__(self,
                 gp_param: Optional[float],
                 drift_param: Optional[float],
                 a: float = 0.0, b: float = 1.0,
                 eps: Optional[float] = None,
                 multi_resolution: bool = False):
        super(LSGANDiscriminatorLoss, self).__init__()
        self.gp_param = gp_param
        self.drift_param = drift_param
        self.a = a
        self.b = b
        self.eps = eps
        self.multi_resolution = multi_resolution

    def forward(self,
                discriminator: nn.Module,
                fakes: Union[torch.Tensor, List[torch.Tensor]],
                reals: Union[torch.Tensor, List[torch.Tensor]],
                scaler: Optional[cuda.amp.GradScaler]=None):
        reals_out = discriminator(reals)
        fakes_out = discriminator(fakes)
        lsgan_loss = 0.5 * (reals_out - self.b).square().mean() + \
                0.5 * (fakes_out - self.a).square().mean()
        if self.gp_param is not None:
            grad_penalty_loss = self.gp_param * gradiend_penalty(
                discriminator=discriminator,
                fakes=fakes,
                reals=reals,
                scaler=scaler,
                multi_resolution=self.multi_resolution,
                eps=self.eps)
        else:
            grad_penalty_loss = 0.0

        if self.drift_param is not None:
            drift_loss = self.drift_param * reals_out.square().mean()
        else:
            drift_loss = 0.0

        return lsgan_loss, grad_penalty_loss, drift_loss


class LSGANGeneratorLoss(nn.Module):
    def __init__(self, c: float = 1.0):
        super(LSGANGeneratorLoss, self).__init__()
        self.c = c

    def forward(self,
                discriminator: nn.Module,
                fakes: Union[torch.Tensor, List[torch.Tensor]],
                reals: Union[torch.Tensor, List[torch.Tensor]]):
        fakes_out = discriminator(fakes)
        return 0.5 * (fakes_out - self.c).square().mean()


class RelativisticAverageHingeDiscriminatorLoss(nn.Module):
    def __init__(self,
                 gp_param: Optional[float] = None,
                 drift_param: Optional[float] = None,
                 eps: Optional[float] = None,
                 multi_resolution: bool = False):
        super(RelativisticAverageHingeDiscriminatorLoss, self).__init__()
        self.gp_param = gp_param
        self.drift_param = drift_param
        self.eps = eps
        self.multi_resolution = multi_resolution

    def forward(self,
                discriminator: nn.Module,
                fakes: Union[torch.Tensor, List[torch.Tensor]],
                reals: Union[torch.Tensor, List[torch.Tensor]],
                scaler=None):
        reals_out = discriminator(reals)
        fakes_out = discriminator(fakes)
        if self.gp_param is not None:
            grad_penalty_loss = self.gp_param * gradiend_penalty(
                discriminator=discriminator,
                fakes=fakes,
                reals=reals,
                scaler=scaler,
                multi_resolution=self.multi_resolution,
                eps=self.eps)
        else:
            grad_penalty_loss = 0.0
        if self.drift_param is not None:
            drift_loss = self.drift_param * reals_out.square().mean()
        else:
            drift_loss = 0.0
        return F.relu(1.0 - (reals_out - fakes_out.mean())).mean() + \
               F.relu(1.0 + (fakes_out - reals_out.mean())).mean(), \
               grad_penalty_loss, \
               drift_loss

class RelativisticAverageHingeGeneratorLoss(nn.Module):
    def forward(self,
                discriminator: nn.Module,
                fakes: Union[torch.Tensor, List[torch.Tensor]],
                reals: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        reals_out = discriminator(reals)
        fakes_out = discriminator(fakes)
        return F.relu(1.0 + (reals_out - fakes_out.mean())).mean() + \
               F.relu(1.0 - (fakes_out - reals_out.mean())).mean()


class ContrastiveDiscriminatorLoss(nn.Module):
    def __init__(self, tau: float = 0.1, eps=1e-8):
        super(ContrastiveDiscriminatorLoss, self).__init__()
        self.tau = tau
        self.eps = eps

    def positive(self, vr1: torch.Tensor, vr2: torch.Tensor) -> torch.Tensor:
        '''
        SimCLR
        vr1, vr2: [N, C]
        '''
        N = vr1.shape[0]
        device = vr1.device
        x = torch.cat((vr1, vr2), dim=0)
        x = self.cosine_cdist(x, x)
        mask = torch.eye(n=2 * N, device=device) == 1
        x = -torch.log_softmax(x.masked_fill(mask=mask, value=-math.inf), dim=0)  # [2N, 2N]
        loss = (torch.einsum('ii', x[N:, :N]) + torch.einsum('ii', x[:N, N:])) / (2 * N)
        return loss

    def negative(self, vr1: torch.Tensor, vr2: torch.Tensor, vf: torch.Tensor) -> torch.Tensor:
        '''
        Supervised Contrastive Loss
        vr1, vr2, vf: [N, C]
        '''
        N = vf.shape[0]
        device = vf.device
        x = vf
        y = torch.cat((vf, vr1, vr2), dim=0)
        d = self.cosine_cdist(x, y)
        mask = torch.eye(n=N, m=3 * N, device=device) == 1
        d = -torch.log_softmax(d.masked_fill(mask=mask, value=-math.inf), dim=1)  # [N, 3N]
        loss = d.masked_fill(mask=mask, value=0)[:N, :N].sum() / (N * (N - 1))
        return loss

    def forward(self,
                vr1p: torch.Tensor, vr2p: torch.Tensor,
                vr1n: torch.Tensor, vr2n: torch.Tensor, vfn: torch.Tensor) -> torch.Tensor:
        '''
        vr1, vr2, vf: [N, C]
        '''
        return self.positive(vr1=vr1p, vr2=vr2p) + self.negative(vr1=vr1n, vr2=vr2n, vf=vfn)

    def cosine_cdist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''
        x, y: [N1, C], [N2, C]
        Returns:
            [N1, N2]
        '''
        x, y = F.normalize(x, eps=self.eps).unsqueeze(1), F.normalize(y, eps=self.eps).unsqueeze(0)
        return (x * y).sum(dim=2) / self.tau


class NonSaturatingGeneratorLoss(nn.Module):
    def __init__(self,
                 use_unet_decoder: bool = False,
                 use_contrastive_discriminator: bool = False,
                 resolution: Optional[int] = None):
        super(NonSaturatingGeneratorLoss, self).__init__()
        self.use_unet_decoder = use_unet_decoder
        self.use_contrastive_discriminator = use_contrastive_discriminator
        if use_contrastive_discriminator:
            assert resolution is not None
            self.transform = SimCLRAugmentation(resolution)

    def forward(self,
                discriminator: nn.Module,
                fakes: Union[torch.Tensor, List[torch.Tensor]],
                reals: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        if self.use_contrastive_discriminator:
            fakes = self.transform(fakes)
        fakes_out, fakes_unet_out = discriminator(
            fakes,
            unet_out=self.use_unet_decoder,
            contrastive_out="generator" if self.use_contrastive_discriminator else None)
        loss = F.softplus(-fakes_out).mean()
        if self.use_unet_decoder:
            loss += F.softplus(-fakes_unet_out).mean()
        return loss


class NonSaturatingDiscriminatorLoss(nn.Module):
    def __init__(self,
                 gp_param: Optional[float] = None,
                 drift_param: Optional[float] = None,
                 r1_param: Optional[float] = None,
                 eps: Optional[float] = None,
                 multi_resolution: bool = False,
                 adaptive_augmentation: Optional[AdaptiveAugmentation] = None,
                 use_unet_decoder: bool = False,
                 use_contrastive_discriminator: bool = False,
                 resolution: Optional[int] = None,
                 r1_per: Optional[int] = None,
                 set_p_per: int = 4):
        super(NonSaturatingDiscriminatorLoss, self).__init__()
        self.gp_param = gp_param
        self.drift_param = drift_param
        self.use_unet_decoder = use_unet_decoder
        self.use_contrastive_discriminator = use_contrastive_discriminator
        self.set_p_per = set_p_per
        self.cnt = 0
        self.eps = eps
        self.rt = 0.
        self.multi_resolution = multi_resolution
        self.loss = nn.LogSigmoid()
        self.adaptive_augmentation = adaptive_augmentation

        if use_contrastive_discriminator:
            assert resolution is not None
            self.transform = SimCLRAugmentation(resolution)
            self.contrastive_discriminator_loss = ContrastiveDiscriminatorLoss(eps=eps)

    def is_next_set_p(self):
        return self.adaptive_augmentation is not None and (self.cnt + 1) % self.set_p_per == 0

    def forward(self,
                discriminator: nn.Module,
                fakes: torch.Tensor,
                reals: torch.Tensor,
                scaler=None) -> Tuple[Union[torch.Tensor, float], ...]:
        self.cnt += 1

        if self.use_contrastive_discriminator:
            vr1 = self.transform(reals).detach()
            vr2 = self.transform(reals).detach()
            vf = self.transform(fakes).detach()
            reals_out, reals_out_unet = discriminator(
                vr2,
                unet_out=self.use_unet_decoder,
                contrastive_out="discriminator")
            fakes_out, fakes_out_unet = discriminator(
                vf,
                unet_out=self.use_unet_decoder,
                contrastive_out="discriminator")
            # vr1p = discriminator(vr1, contrastive_out="positive")
            # vr2p = discriminator(vr2, contrastive_out="positive")
            B = vr1.shape[0]
            vr1p, vr2p = discriminator(
                    torch.cat((vr1, vr2)),
                    contrastive_out="positive").split((B, B))
            vr1n, vr2n = discriminator(
                    torch.cat((vr1, vr2)),
                    contrastive_out="negative").split((B, B))
            # vr1n = discriminator(vr1, contrastive_out="negative")
            # vr2n = discriminator(vr2, contrastive_out="negative")
            vfn = discriminator(vf, contrastive_out="negative")
            contrastive_loss = self.contrastive_discriminator_loss(
                    vr1p=vr1p, vr2p=vr2p,
                    vr1n=vr1n, vr2n=vr2n, vfn=vfn)
        else:
            reals_out, reals_out_unet = discriminator(reals, unet_out=self.use_unet_decoder)
            fakes_out, fakes_out_unet = discriminator(fakes, unet_out=self.use_unet_decoder)
            contrastive_loss = 0.

        if self.use_unet_decoder:
            real_unet_loss = F.softplus(-reals_out_unet).mean()
            fake_unet_loss = F.softplus(fakes_out_unet).mean()
            eps = torch.ones(  # fake: 0, real: 1
                1, 1, *reals.shape[-2:],
                device=reals.device,
                dtype=reals.dtype)
            i, j, h, w = transforms.RandomResizedCrop.get_params(  # small scale
                    reals_out_unet, scale=[0.08, 1.], ratio=[3 / 4, 4 / 3])
            eps[:, :, i: h + i, j: w + j] = 0.
            mixes1 = torch.lerp(fakes, reals, weight=eps)
            mix_out1, mix_out_unet1 = discriminator(mixes1, unet_out=True)
            mixes2 = torch.lerp(fakes, reals, weight=1 - eps)
            mix_out2, mix_out_unet2 = discriminator(mixes1, unet_out=True)

            # mix
            mix_loss1 = F.softplus(mix_out1).mean()  # as fakes
            mix_unet_loss1 = F.softplus(((-1) ** eps) * mix_out_unet1).mean()
            mix_loss2 = F.softplus(mix_out2).mean()  # as fakes
            mix_unet_loss2 = F.softplus(((-1) ** (1 - eps)) * mix_out_unet2).mean()
            # consistency reguralization
            out_mix_unet1 = torch.lerp(fakes_out_unet, reals_out_unet, weight=eps)
            consistency_loss1 = (mix_out_unet1 - out_mix_unet1).square().mean()
            out_mix_unet2 = torch.lerp(fakes_out_unet, reals_out_unet, weight=1 - eps)
            consistency_loss2 = (mix_out_unet2 - out_mix_unet2).square().mean()
        else:
            real_unet_loss, fake_unet_loss = 0., 0.
            mix_loss1, mix_loss2, mix_unet_loss1, mix_unet_loss2 = 0., 0., 0., 0.
            consistency_loss1, consistency_loss2 = 0., 0.

        if self.gp_param is not None:
            grad_penalty_loss = self.gp_param * gradiend_penalty(
                discriminator=discriminator,
                fakes=fakes,
                reals=reals,
                scaler=scaler,
                multi_resolution=self.multi_resolution,
                eps=self.eps)
        else:
            grad_penalty_loss = 0.0

        if self.drift_param is not None:
            drift_loss = self.drift_param * reals_out.square().mean()
        else:
            drift_loss = 0.0

        real_loss = F.softplus(-reals_out).mean()
        fake_loss = F.softplus(fakes_out).mean()
        if self.is_next_set_p():
            nimg = reals_out.shape[0]
            self.rt = reals_out.sign().mean().item()
            adjust = np.sign(self.rt - self.adaptive_augmentation.target_rt) * \
                    nimg * self.adaptive_augmentation.speed * self.set_p_per / 1000
            self.adaptive_augmentation.p = np.clip(
                a=self.adaptive_augmentation.p + adjust, a_min=0.0, a_max=1.0)
        return real_loss + fake_loss + mix_loss1 + mix_loss2, \
               real_unet_loss + fake_unet_loss + mix_unet_loss1 + mix_unet_loss2, \
               consistency_loss1 + consistency_loss2, \
               contrastive_loss, \
               grad_penalty_loss, \
               drift_loss

