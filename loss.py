
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional


def gradiend_penalty(discriminator, fakes, reals, scaler=None):
    batch_size = fakes[0].size(0)
    eps = torch.rand(
            batch_size, 1, 1, 1,
            dtype=fakes[0].dtype
            ).to(fakes[0].device, memory_format=torch.channels_last)
    print([fake.shape for fake in fakes])
    print([real.shape for real in reals])
    interpolates = [fake * (1 - eps) + real * eps
                    for fake, real in zip(fakes, reals)]
    disc_interpolates = discriminator(interpolates)
    if scaler is not None:
        disc_interpolates = scaler.scale(disc_interpolates)
    grad = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True)
    grad_penalty_loss = 0.0
    if scaler is not None:
        inv_scale = 1. / (scaler.get_scale() + 1e-8)
    for g in grad:
        if scaler is not None:
            g = g * inv_scale
        grad_penalty_loss += (g.view(batch_size, -1)
                               .norm(2, dim=1) - 1).square().mean()
    grad_penalty_loss /= len(grad)
    # grad_penalty_loss = None
    # for g in grad:
    #     tmp_grad_penalty = (g.reshape(batch_size, -1) \
    #                          .norm(2, dim=1) - 1).square().mean()
    #     if grad_penalty_loss is None or grad_penalty_loss < tmp_grad_penalty:
    #         grad_penalty_loss = tmp_grad_penalty
    return grad_penalty_loss


class GANDiscriminatorLoss(nn.Module):
    def __init__(self, gp_param: Optional[float]):
        super(GANDiscriminatorLoss, self).__init__()
        self.criterion = nn.BCELoss()
        self.gp_param = gp_param

    def forward(self, discriminator: nn.Module, fakes, reals, scaler=None):
        batch_size = reals[0].shape[0]
        labels = torch.full((batch_size, ), 1.0).to(reals[0].device)
        output = discriminator(reals).view(-1)
        lossD = self.criterion(output, labels)
        if self.gp_param is not None:
            grad_penalty_loss = self.gp_param * gradiend_penalty(discriminator, fakes, reals, scaler)
        else:
            grad_penalty_loss = 0.0
        return lossD, grad_penalty_loss


class GANGeneratorLoss(nn.Module):
    def __init__(self):
        super(GANGeneratorLoss, self).__init__()

    def forward(self, discriminator, fakes, reals):
        return -discriminator(fakes).mean()


class WGANgpDiscriminatorLoss(nn.Module):
    def __init__(self, gp_param: Optional[float], drift_param: Optional[float]):
        super(WGANgpDiscriminatorLoss, self).__init__()
        self.gp_param = gp_param
        self.drift_param = drift_param

    def forward(self,
                discriminator: nn.Module,
                fakes: torch.Tensor,
                reals: torch.Tensor,
                scaler: torch.cuda.amp.GradScaler=None):
        fakes_out = discriminator(fakes)
        reals_out = discriminator(reals)
        wgan_loss = fakes_out.mean() - reals_out.mean()

        if self.gp_param is not None:
            grad_penalty_loss = self.gp_param * gradiend_penalty(discriminator, fakes, reals, scaler)
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

    def forward(self, discriminator: nn.Module, fakes: torch.Tensor, reals: torch.Tensor):
        loss = -discriminator(fakes)
        return loss.mean()


class LSGANDiscriminatorLoss(nn.Module):
    def __init__(self, gp_param: Optional[float], drift_param: Optional[float], a: float = 0.0, b: float = 1.0):
        super(LSGANDiscriminatorLoss, self).__init__()
        self.gp_param = gp_param
        self.drift_param = drift_param
        self.a = a
        self.b = b

    def forward(self,
                discriminator: nn.Module,
                fakes: torch.Tensor,
                reals: torch.Tensor,
                scaler: torch.cuda.amp.GradScaler=None):
        reals_out = discriminator(reals)
        fakes_out = discriminator(fakes)
        lsgan_loss = 0.5 * (reals_out - self.b).square().mean() + \
                0.5 * (fakes_out - self.a).square().mean()
        if self.gp_param is not None:
            grad_penalty_loss = self.gp_param * gradiend_penalty(discriminator, fakes, reals, scaler)
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

    def forward(self, discriminator: nn.Module, fakes: torch.Tensor, reals: torch.Tensor):
        fakes_out = discriminator(fakes)
        return 0.5 * (fakes_out - self.c).square().mean()


class RelativisticAverageHingeGeneratorLoss(nn.Module):
    def forward(self, discriminator, fakes, reals):
        reals_out = discriminator(reals)
        fakes_out = discriminator(fakes)
        return F.relu(1.0 + (reals_out - fakes_out.mean())).mean() + \
               F.relu(1.0 - (fakes_out - reals_out.mean())).mean()


class RelativisticAverageHingeDiscriminatorLoss(nn.Module):
    def __init__(self, gp_param: Optional[float] = None, drift_param: Optional[float] = None):
        super(RelativisticAverageHingeDiscriminatorLoss, self).__init__()
        self.gp_param = gp_param
        self.drift_param = drift_param

    def forward(self, discriminator, fakes, reals, scaler=None):
        reals_out = discriminator(reals)
        fakes_out = discriminator(fakes)
        if self.gp_param is not None:
            grad_penalty_loss = self.gp_param * gradiend_penalty(discriminator, fakes, reals, scaler)
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

