
import math
import random
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, Final

from kornia.filters import filter2D
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse
import torch
from torch import nn, cuda
from torch.nn import functional as F
from utils import Up, Down


class Print(nn.Module):
    def forward(self, input):
        print(input.shape)
        return input


class Activation(nn.Module):
    activation: Union[nn.Module, Callable[[Any], torch.Tensor]]

    def __init__(self, activation: str, use_scale: bool, *args, **kwargs):
        super(Activation, self).__init__()
        self.use_scale = use_scale
        if activation == "relu":
            self.activation = nn.ReLU(*args, **kwargs)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(*args, **kwargs)
        elif activation == "prelu":
            self.activation = nn.PReLU(*args, **kwargs)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = nn.Identity()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            assert False, f"unsupported activation: {activation}"

    def forward(self, input):
        if self.use_scale:
            return self.activation(input) * math.sqrt(2)
        else:
            return self.activation(input)


class Blur(nn.Module):
    def __init__(self, kernel_size: int):
        super(Blur, self).__init__()
        if kernel_size == 4:
            filter = torch.tensor([1., 3., 3., 1.])
            # filter = fuse_weight(filter)
        else:
            filter = torch.tensor([1., 2., 1.])
        filter = filter[None, None, :] * filter[None, :, None]  # [1, K, K]
        self.register_buffer("filter", filter.requires_grad_(False))

    def forward(self, input: torch.Tensor):
        return filter2D(
            input=input,
            kernel=self.filter,
            normalized=True)


def fuse_weight(weight: torch.Tensor):
    weight = F.pad(weight, [1, 1, 1, 1])
    weight = (
        weight[:, 1:, 1:] +
        weight[:, :-1, 1:] +
        weight[:, 1:, :-1] +
        weight[:, :-1, :-1]) / 4
    return weight


class UpConvDown(nn.Conv2d):
    def __init__(self, *args, up: int = 0, down: int = 0, **kwargs):
        super(UpConvDown, self).__init__(*args, **kwargs)
        self.up = up
        self.down = down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.up != 0:
            B, C, H, W = x.shape
            x = F.pad(
                    x[:, :, :, None, :, None],
                    (0, self.up,
                     0, 0,
                     0, self.up,
                     0, 0), mode="constant") \
                 .reshape(B, C, self.up * H, self.up * W)
        x = super().forward(x)
        if self.down != 0:
            x = x[:, :, ::self.down, ::self.down]
        return x


class WaveletInterpolate(nn.Module):
    def __init__(self, scale_factor: float):
        super(WaveletInterpolate, self).__init__()
        self.iwt = DWTInverse(wave="haar", mode="reflect")
        self.scale_factor = scale_factor
        self.dwt = DWTForward(wave="haar", mode="reflect")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, CW, H, W = x.shape  # CW: Channels of Wavelets
        C = CW // 4
        # yl: [B, 3, H, W], yh: [B, 9, H, W]
        xl, xh = torch.split(x, split_size_or_sections=(C, 3 * C), dim=1)
        xh = xh.reshape(B, C, 3, H, W)
        x = self.iwt((xl, (xh,)))
        x = F.interpolate(x,
                          scale_factor=self.scale_factor,
                          mode="bilinear",
                          align_corners=False)
        xl, (xh, ) = self.dwt(x)
        xh = xh.reshape(B, -1, int(self.scale_factor * H), int(self.scale_factor * W))
        x = torch.cat((xl, xh), dim=1)
        return x


class WeightScaledLinear(nn.Linear):
    def __init__(self, use_scale: bool, lrmul: float = 1., *args, **kwargs):
        super(WeightScaledLinear, self).__init__(*args, **kwargs)
        he_std = math.sqrt(1 / np.prod(self.weight.shape[1:]))
        self.lrmul = lrmul
        if use_scale:
            init_std = 1. / lrmul
            self.scale = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.scale = lrmul
        nn.init.normal_(self.weight.data, std=init_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, input: torch.Tensor):
        return F.linear(
            input=input,
            weight=self.weight * self.scale,
            bias=None if self.bias is None else self.bias * self.lrmul)


class WeightScaledConv(nn.Conv2d):
    def __init__(self, use_scale: bool, lrmul: float = 1., use_fp16: bool = False, *args, **kwargs):
        super(WeightScaledConv, self).__init__(*args, **kwargs)
        he_std = math.sqrt(1 / np.prod(self.weight.shape[1:]))
        self.lrmul = lrmul
        self.use_fp16 = use_fp16
        if use_scale:
            init_std = 1. / lrmul
            self.scale = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.scale = lrmul
        nn.init.normal_(self.weight.data, std=init_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.conv2d(
            input=input,
            weight=self.weight * self.scale,
            bias=None if self.bias is None else self.bias * self.lrmul,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            dilation=self.dilation)
        if self.use_fp16:
            abs = 2 ** 8
            output.clamp_(min=-abs, max=abs)
        return output


class WeightScaledConvTrans(nn.ConvTranspose2d):
    def __init__(self, use_scale: bool, lrmul: float = 1., use_fp16: bool = False, *args, **kwargs):
        super(WeightScaledConvTrans, self).__init__(*args, **kwargs)
        he_std = math.sqrt(1 / (self.in_channels * np.prod(self.kernel_size)))
        self.lrmul = lrmul
        self.use_fp16 = use_fp16
        if use_scale:
            init_std = 1. / lrmul
            self.scale = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.scale = lrmul
        nn.init.normal_(self.weight.data, std=init_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, input, output_size=None):
        weight = self.weight * self.scale
        output_padding = self._output_padding(
            input,
            output_size,
            list(self.stride),
            list(self.padding),
            list(self.kernel_size))
        output = F.conv_transpose2d(
            input=input,
            weight=weight,
            bias=None if self.bias is None else self.bias * self.lrmul,
            stride=self.stride,
            padding=self.padding,
            output_padding=output_padding,
            groups=self.groups,
            dilation=self.dilation)
        if self.use_fp16:
            abs = 2 ** 8
            output.clamp_(min=-abs, max=abs)
        return output


class WeightModulatedConv(nn.Conv2d):
    def __init__(self,
                 style_channels: int,
                 use_scale: bool,
                 eps: float,
                 demodulate: bool = True,
                 lrmul: float = 1.,
                 use_fp16: bool = False,
                 *args, **kwargs):
        super(WeightModulatedConv, self).__init__(*args, **kwargs)
        self.eps = eps
        self.demodulate = demodulate
        he_std = math.sqrt(1 / np.prod(self.weight.shape[1:]))
        self.lrmul = lrmul
        self.use_fp16 = use_fp16
        if use_scale:
            init_std = 1. / lrmul
            self.scale = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.scale = lrmul
        nn.init.normal_(self.weight.data, std=init_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

        self.style_affine = WeightScaledConv(
            in_channels=style_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            use_scale=use_scale)

    def forward(self, input: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        B, _, *HW = input.shape
        x = input.reshape(1, -1, *HW)
        style = self.style_affine(style).unsqueeze(1)  # [B, 1, Ci, 1, 1]
        weight = self.weight.unsqueeze(0) \
                .expand(B, -1, -1, -1, -1)  # [B, Co, Ci, K, K]
        if self.use_fp16:
            style = F.normalize(style)
            d = (weight.square().sum(dim=(2, 3, 4), keepdim=True) + self.eps).rsqrt()
            weight = weight * d
        weight = weight * (style + 1.) * self.scale
        if self.demodulate:
            d = (weight.square().sum(dim=(2, 3, 4), keepdim=True) + self.eps).rsqrt()
            weight = weight * d
        weight = weight.reshape(1, -1, self.in_channels, *self.kernel_size) \
                       .squeeze(0)  # [B * Co, Ci, K, K]
        bias: Optional[torch.Tensor]
        if self.bias is not None:
            bias = self.bias.repeat(B) * self.lrmul
        else:
            bias = None
        output = F.conv2d(
            input=x,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=B)
        HW = list(output.shape[-2:])
        output = output.reshape(B, -1, *HW)

        if self.use_fp16:
            abs = 2 ** 8
            output.clamp_(min=-abs, max=abs)
        return output


class WeightModulatedConvTrans(nn.ConvTranspose2d):
    def __init__(self,
                 style_channels: int,
                 use_scale: bool,
                 eps: float,
                 demodulate: bool = True,
                 lrmul: float = 1.,
                 use_fp16: bool = False,
                 *args, **kwargs):
        super(WeightModulatedConvTrans, self).__init__(*args, **kwargs)
        self.eps = eps
        self.demodulate = demodulate
        he_std = math.sqrt(1 / (self.in_channels * np.prod(self.kernel_size)))
        self.lrmul = lrmul
        self.use_fp16 = use_fp16
        if use_scale:
            init_std = 1. / lrmul
            self.scale = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.scale = lrmul
        nn.init.normal_(self.weight.data, std=init_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

        self.style_affine = WeightScaledConv(
            in_channels=style_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            use_scale=use_scale)

    def forward(self, input: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        B, _, *HW = input.shape
        x = input.reshape(1, -1, *HW)
        style = self.style_affine(style).unsqueeze(2)  # [B, Ci, 1, 1, 1]
        weight = self.weight.unsqueeze(0) \
                            .expand(B, -1, -1, -1, -1)  # [B, Ci, Co, K, K]
        if self.use_fp16:
            style = F.normalize(style)
            d = (weight.square().sum(dim=(1, 3, 4), keepdim=True) + self.eps).rsqrt()
            weight = weight * d
        weight = weight * (style + 1.) * self.scale
        if self.demodulate:
            d = (weight.square().sum(dim=(1, 3, 4), keepdim=True) + self.eps).rsqrt()
            weight = weight * d
        weight = weight.reshape(1, -1, self.out_channels, *self.kernel_size) \
                       .squeeze(0)  # [B * Ci, Co, K, K]
        bias: Optional[torch.Tensor]
        if self.bias is not None:
            bias = self.bias.repeat(B) * self.lrmul
        else:
            bias = None
        output = F.conv_transpose2d(
            input=x,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            output_padding=self.output_padding,
            groups=B)
        HW = list(output.shape[-2:])
        output = output.reshape(B, -1, *HW)
        if self.use_fp16:
            abs = 2 ** 8
            output.clamp_(min=-abs, max=abs)
        return output


class NoiseBlock(nn.Module):
    bias: Optional[torch.Tensor]

    def __init__(self, channels: int, resolution: int, mode: str, bias: bool = False):
        super(NoiseBlock, self).__init__()
        self.mode = mode
        self.resolution = resolution
        self.channels = channels
        if mode == "deterministic":
            self.register_buffer("noise", torch.randn(1, 1, resolution, resolution))
        self.scale = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros([channels, 1, 1]), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, noise: Optional[torch.Tensor] = None):
        B = x.shape[0]
        if self.mode == "deterministic":
            return x + self.scale * self.noise
        elif self.mode in ["const-random", "const-deterministic"]:
            return x + self.scale * noise
        elif self.mode == "random":
            noise = torch.randn(B, 1, self.resolution, self.resolution, device=x.device)
            return x + self.scale * noise
        else:
            assert False


class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super(PixelNorm, self).__init__()
        self.eps = eps

    def forward(self, input):
        return input * torch.rsqrt(
            input.square().mean(dim=1, keepdim=True) + self.eps)


class AdaIN(nn.Module):
    def __init__(self, in_channels: int, style_channels: int, use_scale: bool):
        super(AdaIN, self).__init__()
        self.in_channels = in_channels
        self.style_affine = WeightScaledConv(
            in_channels=style_channels,
            out_channels=in_channels * 2,
            kernel_size=1,
            use_scale=use_scale)
        self.instance_norm = nn.InstanceNorm2d(num_features=in_channels)

    def forward(self, input: torch.Tensor, style: torch.Tensor):
        style = self.style_affine(style)
        ys, yb = torch.split(style, self.in_channels, dim=1)
        return (ys + 1) * self.instance_norm(input) + yb


class GeneratorBlockV1(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 style_channels: int,
                 activation: str,
                 activation_args: Dict,
                 use_scale: bool,
                 eps: float,
                 noise_mode: str,
                 idx: bool = False):
        super(GeneratorBlockV1, self).__init__()
        self.first = idx == 0
        self.resolution = 2 ** (idx + 2)
        self.activation = Activation(
            activation=activation,
            **activation_args)
        self.noise_mode = noise_mode
        if self.noise_mode == "const-deterministic":
            self.register_buffer("noise", torch.randn(1, 1, self.resolution, self.resolution))
        else:
            self.noise = None

        if self.first:
            self.const = nn.Parameter(torch.randn(1, in_channels, 4, 4))
            self.noise1 = NoiseBlock(
                channels=in_channels,
                resolution=self.resolution,
                mode=noise_mode,)
            self.adain1 = AdaIN(
                in_channels=in_channels,
                style_channels=style_channels,
                use_scale=use_scale)
            self.conv1 = WeightScaledConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
                use_scale=use_scale)
            self.noise2 = NoiseBlock(
                channels=out_channels,
                resolution=self.resolution,
                mode=noise_mode)
            self.adain2 = AdaIN(
                in_channels=out_channels,
                style_channels=style_channels,
                use_scale=use_scale)
        else:
            self.conv1 = WeightScaledConvTrans(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                padding=1,
                stride=2,
                use_scale=use_scale)
            self.blur = Blur(kernel_size=3)
            self.noise1 = NoiseBlock(
                channels=out_channels,
                resolution=self.resolution,
                mode=noise_mode)
            self.adain1 = AdaIN(
                in_channels=out_channels,
                style_channels=style_channels,
                use_scale=use_scale)
            self.conv2 = WeightScaledConv(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
                use_scale=use_scale)
            self.noise2 = NoiseBlock(
                channels=out_channels,
                resolution=self.resolution,
                mode=noise_mode)
            self.adain2 = AdaIN(
                in_channels=out_channels,
                style_channels=style_channels,
                use_scale=use_scale)

    def forward(self,
                x: Optional[torch.Tensor] = None,
                *,
                style: torch.Tensor):
        B = style.shape[0]
        if self.noise_mode == "const-random":
            noise = torch.randn(1, 1, self.resolution, self.resolution, device=style.device)
        else:
            noise = self.noise
        if self.first:
            x = self.const.expand(B, -1, -1, -1)
            x = self.noise1(x, noise)
            x = self.activation(x)
            x = self.adain1(x, style)
            x = self.conv1(x)
            x = self.noise2(x, noise)
            x = self.activation(x)
            x = self.adain2(x, style)
            return x
        else:
            # x = self.upsample(x)
            x = self.conv1(x)
            x = self.blur(x)
            x = self.noise1(x, noise)
            x = self.activation(x)
            x = self.adain1(x, style)
            x = self.conv2(x)
            x = self.noise2(x, noise)
            x = self.activation(x)
            x = self.adain2(x, style)
            return x


class GeneratorBlockV2(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 style_channels: int,
                 activation: str,
                 activation_args: Dict,
                 use_scale: bool,
                 eps: float,
                 noise_mode: str,
                 idx: int,
                 use_fp16: bool):
        super(GeneratorBlockV2, self).__init__()
        self.first = idx == 0
        self.resolution = 2 ** (idx + 2)
        self.activation = Activation(
            activation=activation,
            **activation_args)
        self.noise_mode = noise_mode
        if self.noise_mode == "const-deterministic":
            self.register_buffer("noise", torch.randn(1, 1, self.resolution, self.resolution))
        else:
            self.noise = None

        if self.first:
            self.const = nn.Parameter(torch.randn(1, in_channels, 4, 4))
            self.conv1 = WeightModulatedConv(
                in_channels=in_channels,
                out_channels=out_channels,
                style_channels=style_channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
                use_scale=use_scale,
                eps=eps,
                use_fp16=use_fp16,
                bias=True)
            self.noise1 = NoiseBlock(
                channels=1,
                resolution=self.resolution,
                mode=noise_mode)
        else:
            self.upsample = nn.Sequential(
                Up(scale=2),
                Blur(kernel_size=3))
            self.conv1 = WeightModulatedConv(
                in_channels=in_channels,
                out_channels=out_channels,
                style_channels=style_channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
                use_scale=use_scale,
                eps=eps,
                use_fp16=use_fp16,
                bias=True)

            self.noise1 = NoiseBlock(
                channels=1,
                resolution=self.resolution,
                mode=noise_mode)
            self.conv2 = WeightModulatedConv(
                in_channels=out_channels,
                out_channels=out_channels,
                style_channels=style_channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
                use_scale=use_scale,
                eps=eps,
                use_fp16=use_fp16,
                bias=True)
            self.noise2 = NoiseBlock(
                channels=1,
                resolution=self.resolution,
                mode=noise_mode)

    def forward(self,
                x: Optional[torch.Tensor] = None,
                *,
                style: torch.Tensor):
        B = style.shape[0]
        if self.noise_mode == "const-random":
            noise = torch.randn(1, 1, self.resolution, self.resolution, device=style.device)
        else:
            noise = self.noise
        if self.first:
            x = self.conv1(self.const.expand(B, -1, -1, -1), style)
            x = self.noise1(x, noise)
            x = self.activation(x)
            return x
        else:
            x = self.upsample(x)

            x = self.conv1(x, style)
            x = self.noise1(x, noise)
            x = self.activation(x)

            x = self.conv2(x, style)
            x = self.noise2(x, noise)
            x = self.activation(x)
            return x


class LatentLayers(nn.Module):
    def __init__(self,
                 num_layers: int,
                 style_channels: int,
                 activation: str,
                 activation_args: Dict,
                 use_scale: bool,
                 lrmul: float,
                 eps: float,
                 num_styles: int,
                 w_avg_rate: float):
        super(LatentLayers, self).__init__()
        self.num_styles = num_styles
        self.layers = nn.Sequential(
            *[nn.Sequential(
                WeightScaledConv(
                    in_channels=style_channels,
                    out_channels=style_channels,
                    kernel_size=1,
                    use_scale=use_scale,
                    lrmul=lrmul),
                Activation(
                    activation=activation,
                    **activation_args))
              for _ in range(num_layers)])

        self.eps = eps

        self.w_avg_rate = w_avg_rate
        self.register_buffer("w_avg", torch.randn(style_channels, 1, 1))

    def forward(self,
                latent: torch.Tensor,
                *,
                mixing_regularization_rate: Optional[float] = None,
                truncation_trick_rate: Optional[float] = None) -> List[torch.Tensor]:
        latent = F.normalize(latent, p=2, dim=1, eps=self.eps)  # [B, style_channels, N, 1]
        styles = self.layers(latent)
        if self.training:
            style_mean = styles.mean(dim=(0, 2), keepdim=True).squeeze(0)
            self.w_avg = (1. - self.w_avg_rate) * self.w_avg + self.w_avg_rate * style_mean

        if mixing_regularization_rate is not None and \
                random.uniform(0., 1.) < mixing_regularization_rate:
            # use 2 latent vectors
            switch_point = random.randint(0, self.num_styles - 1)
        else:
            # use single latent vector
            switch_point = self.num_styles

        idx = 0
        mixed_style = []
        for styles_idx in range(self.num_styles):
            if switch_point == styles_idx:
                idx = 1
            if truncation_trick_rate is None:
                mixed_style.append(styles[:, :, idx].unsqueeze(2))
            elif not self.training:
                style = styles[:, :, idx].unsqueeze(2)
                mixed_style.append(truncation_trick_rate * style + (1. - truncation_trick_rate) * self.w_avg)
            else:
                assert False
        return mixed_style


class GeneratorV2(nn.Module):
    def __init__(self,
                 style_channels: int,
                 activation: str,
                 activation_args: Dict,
                 use_scale: bool,
                 eps: float,
                 noise_mode: str,
                 channel_info: List[Tuple[int, int]],
                 use_fp16: bool,
                 mode: str = "skip"  # [skip, wavelet]
                 ):
        super(GeneratorV2, self).__init__()
        self.mode = mode
        self.use_fp16 = use_fp16
        self.convolutions = nn.ModuleList([
            GeneratorBlockV2(
                in_channels=in_channels,
                out_channels=out_channels,
                style_channels=style_channels,
                activation=activation,
                activation_args=activation_args,
                use_scale=use_scale,
                eps=eps,
                noise_mode=noise_mode,
                use_fp16=use_fp16,
                idx=idx)
            for idx, (in_channels, out_channels) in enumerate(channel_info)
            ])
        self.to_rgb = nn.ModuleList([
            WeightModulatedConv(
                in_channels=out_channels,
                out_channels=3 if mode == "skip" else 12,
                style_channels=style_channels,
                kernel_size=1,
                use_scale=use_scale,
                eps=eps,
                use_fp16=use_fp16,
                demodulate=False,
                bias=True)
            for _, out_channels in channel_info
        ])
        if mode == "wavelet":
            self.upsample = WaveletInterpolate(scale_factor=2)
            self.iwt = DWTInverse(wave="haar", mode="reflect")
        elif mode == "skip":
            self.upsample = nn.Sequential(
                nn.Upsample(mode="nearest", scale_factor=2., align_corners=False),
                Blur(kernel_size=3))
        else:
            assert False

    def forward(self, *, styles: List[torch.Tensor]):
        imgs = []
        x = None
        img = None
        for idx, (convolution, to_rgb, style) in enumerate(zip(self.convolutions, self.to_rgb, styles)):
            x = convolution(x, style=style)
            if idx == 0:
                img = to_rgb(x, style)
            else:
                img = to_rgb(x, style) + self.upsample(img)
            if self.mode == "wavelet":
                B, _, H, W = x.shape
                C = 3
                imgl, imgh = torch.split(img, split_size_or_sections=(C, 3 * C), dim=1)
                imgh = imgh.reshape(B, C, 3, H, W)
                imgs.append(self.iwt((imgl, [imgh])))
            elif self.mode == "skip":
                imgs.append(img)
        return imgs


class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size: int = 4, eps: float = 1e-8):
        super(MiniBatchStdDev, self).__init__()
        self.group_size = group_size
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B, C, *HW = input.shape
        G = min(B, self.group_size)
        M = B // G
        x = input.reshape([G, M, C, *HW])  # [G, M, C, H, W]
        # x = x.std(dim=0, unbiased=False)
        x = (x.var(dim=0, unbiased=False) + self.eps).sqrt()
        x = x.mean(dim=(1, 2, 3), keepdim=True)  # [M, 1, 1, 1]
        x = x.repeat(G, 1, *HW)
        return torch.cat((input, x), dim=1)

    # def forward(self, input):
    #     B, C, *HW = input.shape
    #     y = input - input.mean(dim=0, keepdim=True)
    #     y = (y.square().mean(dim=0) + self.eps).sqrt()
    #     y = y.mean().view(1, 1, 1, 1)
    #     std = y.repeat(B, 1, *HW)
    #     return torch.cat((input, std), dim=1)


class DiscriminatorBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: str,
                 activation_args: Dict,
                 use_minibatch_stddev_all: bool,
                 minibatch_stddev_groups_size: int,
                 use_scale: bool,
                 eps: float,
                 mode: str,
                 use_fp16:bool):
        super(DiscriminatorBlock, self).__init__()
        self.mode = mode
        self.use_minibatch_stddev_all = use_minibatch_stddev_all
        if mode == "resnet":
            self.residual = nn.Sequential(
                Blur(kernel_size=3), Down(scale=2),
                WeightScaledConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    use_scale=use_scale,
                    padding_mode="replicate"))
                # WeightScaledConv(
                #     in_channels=in_channels,
                #     out_channels=out_channels,
                #     kernel_size=1,
                #     stride=2,
                #     use_scale=use_scale,
                #     bias=True))
        if use_minibatch_stddev_all:
            self.minibatch_stddev = MiniBatchStdDev(
                eps=eps,
                group_size=minibatch_stddev_groups_size)
        self.net = nn.Sequential(
            WeightScaledConv(
                in_channels=in_channels + 1 if use_minibatch_stddev_all else in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                use_scale=use_scale,
                padding_mode="replicate",
                use_fp16=use_fp16),
            Activation(
                activation=activation,
                **activation_args),
            Blur(kernel_size=3), Down(scale=2),
            WeightScaledConv(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                use_scale=use_scale,
                padding_mode="replicate",
                use_fp16=use_fp16),
            Activation(
                activation=activation,
                **activation_args))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "resnet":  # img is None
            residual = self.residual(x)
            if self.use_minibatch_stddev_all:
                x = self.minibatch_stddev(x)
            x = self.net(x)
            x = (x + residual) / np.sqrt(2)
        elif self.mode == "skip" or self.mode == "wavelet":  # x may be None
            if self.use_minibatch_stddev_all:
                x = self.minibatch_stddev(x)
            x = self.net(x)
        else:
            assert False
        return x


class DiscriminatorDecoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: str,
                 activation_args: Dict,
                 use_scale: bool,
                 use_fp16: bool):
        super(DiscriminatorDecoderBlock, self).__init__()
        self.residual = WeightScaledConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_scale=use_scale,
            use_fp16=use_fp16)
        self.activation = Activation(activation=activation, **activation_args)
        self.conv1 = WeightScaledConvTrans(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            padding=1,
            stride=2,
            use_scale=use_scale,
            use_fp16=use_fp16)
        self.conv2 = WeightScaledConv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            use_scale=use_scale,
            use_fp16=use_fp16)
        self.to_gray = WeightScaledConv(
            in_channels=out_channels,
            out_channels=4,
            kernel_size=1,
            use_scale=use_scale,
            use_fp16=use_fp16)
        self.iwt = DWTInverse(wave="haar", mode="reflect")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x


class DiscriminatorV2(nn.Module):
    def __init__(self,
                 activation: str,
                 activation_args: Dict,
                 use_minibatch_stddev_all: bool,
                 minibatch_stddev_groups_size: int,
                 use_scale: bool,
                 eps: float,
                 channel_info: List[Tuple[int, int]],
                 use_fp16: bool,
                 mode: str = "skip",  # [resnet, skip, wavelet]
                 use_unet_decoder: bool = True,
                 use_contrastive_discriminator: bool = False,
                 projection_dim: Optional[float] = None):
        super(DiscriminatorV2, self).__init__()
        self.mode = mode
        self.use_unet_decoder = use_unet_decoder
        self.use_contrastive_discriminator = use_contrastive_discriminator
        self.use_fp16 = use_fp16
        if mode == "resnet":
            self.from_rgb = nn.Sequential(
                WeightScaledConv(
                    in_channels=3,
                    out_channels=channel_info[-1][1],
                    kernel_size=1,
                    use_scale=use_scale),
                Activation(
                    activation=activation,
                    **activation_args))
        elif mode == "skip" or mode == "wavelet":
            self.from_rgb = nn.ModuleList([
                nn.Sequential(
                    WeightScaledConv(
                        in_channels=3 if mode == "skip" else 12,
                        out_channels=out_channels,
                        kernel_size=1,
                        use_scale=use_scale),
                    Activation(
                        activation=activation,
                        **activation_args))
                    for _, out_channels in reversed(channel_info)])
            if mode == "skip":
                self.downsample = lambda x: \
                    F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
            elif mode == "wavelet":
                self.dwt = DWTForward(wave="haar", mode="reflect")
                self.downsample = WaveletInterpolate(scale_factor=0.5)
        else:
            assert False
        self.convolutions = nn.ModuleList([
            DiscriminatorBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=activation,
                activation_args=activation_args,
                use_minibatch_stddev_all=use_minibatch_stddev_all,
                minibatch_stddev_groups_size=minibatch_stddev_groups_size,
                use_scale=use_scale,
                eps=eps,
                mode=mode,
                use_fp16=use_fp16)
            for out_channels, in_channels in reversed(channel_info[1:])
        ])
        self.feature = nn.Sequential(  # 4x4 -> 2x2
            MiniBatchStdDev(
                eps=eps,
                group_size=minibatch_stddev_groups_size),
            WeightScaledConv(
                in_channels=channel_info[0][0] + 1,
                out_channels=channel_info[0][0],
                kernel_size=3,
                use_scale=use_scale),
            Activation(activation=activation, **activation_args),
            nn.Flatten(),
            WeightScaledLinear(
                in_features=channel_info[0][0] * 4,
                out_features=channel_info[0][0],
                use_scale=use_scale),
            Activation(activation=activation, **activation_args))

        if use_unet_decoder:
            self.unet_convolutions = nn.ModuleList([
                DiscriminatorDecoderBlock(
                    in_channels=2 * in_channels if idx != 0 else in_channels,
                    out_channels=out_channels,
                    activation=activation,
                    activation_args=activation_args,
                    use_scale=use_scale,
                    use_fp16=use_fp16)
                for idx, (in_channels, out_channels) in enumerate(channel_info[1:])
            ])
            self.to_gray = nn.ModuleList([
                WeightScaledConv(
                    in_channels=in_channels,
                    out_channels=4,
                    kernel_size=1,
                    use_scale=use_scale)
                for _, in_channels in channel_info[1:]])
            self.unet_upsample = WaveletInterpolate(scale_factor=2)
            self.iwt = DWTInverse(wave="haar", mode="reflect")

        if use_contrastive_discriminator:
            assert projection_dim is not None
            self.projection_positive = nn.Sequential(
                WeightScaledLinear(
                    in_features=channel_info[0][0],
                    out_features=projection_dim,
                    use_scale=use_scale),
                Activation(activation=activation, **activation_args),
                WeightScaledLinear(
                    in_features=projection_dim,
                    out_features=projection_dim,
                    use_scale=use_scale),
                Activation(activation=activation, **activation_args))
            self.projection_negative = nn.Sequential(
                WeightScaledLinear(
                    in_features=channel_info[0][0],
                    out_features=projection_dim,
                    use_scale=use_scale),
                Activation(activation=activation, **activation_args),
                WeightScaledLinear(
                    in_features=projection_dim,
                    out_features=projection_dim,
                    use_scale=use_scale),
                Activation(activation=activation, **activation_args))
            self.out = nn.Sequential(
                WeightScaledLinear(
                    in_features=channel_info[0][0],
                    out_features=channel_info[0][0],
                    use_scale=use_scale),
                Activation(activation=activation, **activation_args),
                WeightScaledLinear(
                    in_features=channel_info[0][0],
                    out_features=projection_dim,
                    use_scale=use_scale),
                Activation(activation=activation, **activation_args),
                WeightScaledLinear(
                    in_features=projection_dim,
                    out_features=1,
                    use_scale=use_scale))
        else:
            self.out = WeightScaledLinear(
                in_features=channel_info[0][0],
                out_features=1,
                use_scale=use_scale)

    def get_feature_with_resnet(self, img: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        skips: List[torch.Tensor] = []
        x = self.from_rgb(img)
        for convolution in self.convolutions:
            if self.use_unet_decoder:
                skips.append(x)
            x = convolution(x)
        return x, self.feature(x), skips

    def get_feature_with_skip(self, img: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        skips: List[torch.Tensor] = []
        x: Optional[torch.Tensor] = None
        for from_rgb, convolution in zip(self.from_rgb, self.convolutions):
            if x is None:
                x = from_rgb(img)
            else:
                x = x + from_rgb(img)
            if self.use_unet_decoder:
                assert x is not None
                skips.append(x)
            x = convolution(x)
            img = self.downsample(img)
        assert x is not None
        return x, self.feature(x), skips

    def get_feature_with_wavelet(self, img: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        skips: List[torch.Tensor] = []
        x: Optional[torch.Tensor] = None
        imgl, (imgh, ) = self.dwt(img)
        B, _, H, W = img.shape
        img = torch.cat((imgl, imgh.reshape(B, -1, H // 2, W // 2)), dim=1)
        for from_rgb, convolution in zip(self.from_rgb, self.convolutions):
            if x is None:
                x = from_rgb(img)
            else:
                x = x + from_rgb(img)
            if self.use_unet_decoder:
                assert x is not None
                skips.append(x)
            x = convolution(x)
            img = self.downsample(img)
        assert x is not None
        return x, self.feature(x), skips

    def get_unet_out(self,
                     x: torch.Tensor,
                     skips: List[torch.Tensor],
                     contrastive_out: Optional[str]) -> torch.Tensor:
        gray: Optional[torch.Tensor] = None
        y = x
        if contrastive_out == "discriminator":
            y = y.detach()
        for idx, (skip, to_gray, convolution) in \
                enumerate(zip(reversed(skips), self.to_gray, self.unet_convolutions)):
            if idx != 0:
                if contrastive_out == "discriminator":
                    skip = skip.detach()
                y = torch.cat((y, skip), dim=1)
            y = convolution(y)
            if idx == 0:
                gray = to_gray(y)
            else:
                gray = to_gray(y) + self.unet_upsample(gray)
        assert gray is not None
        B, CW, H, W = gray.shape
        C = CW // 4
        grayl, grayh = torch.split(gray, split_size_or_sections=(C, 3 * C), dim=1)
        grayh = grayh.reshape(B, C, -1, H, W)
        gray = self.iwt((grayl, [grayh]))
        assert gray is not None
        return gray

    def forward(self,
                img: Optional[torch.Tensor] = None, *,
                feature: Optional[torch.Tensor] = None,
                unet_out: bool = False,
                contrastive_out: Optional[str] = None,  # [positive, negative, discriminator, generator, feature]
                r1_regularize: bool = False,
                ) -> Union[
                        Tuple[torch.Tensor, Optional[torch.Tensor]],
                        torch.Tensor]:
        '''
        img: [B, C, resolution, resolution]
        unet_out:
            weather outputs unet output or not
        contrastive_out:
            positive -> project feature with projection_positive
            negative -> project feature with projection_negative
            discriminator -> project feature with out
            feature -> return feature
        r1_regularize:
            if use contrastive discriminator, return feature, otherwise return output of GAN discriminator
        '''
        if feature is not None:
            if contrastive_out == "positive":
                return self.projection_positive(feature)
            elif contrastive_out == "negative":
                return self.projection_negative(feature)
            elif contrastive_out == "discriminator":
                assert not unet_out, "if unet_out, cannot get output of unet decoder from feature."
                return self.out(feature)
            else:
                assert False, "contrastive_out must be positive or negative when feature was given."
        assert img is not None

        skips: List[torch.Tensor] = []
        grad_switch: Final = contrastive_out != "discriminator" and not r1_regularize
        torch.set_grad_enabled(grad_switch)
        if self.mode == "resnet":
            x, feature, skips = self.get_feature_with_resnet(img)
        elif self.mode == "skip":
            x, feature, skips = self.get_feature_with_skip(img)
        elif self.mode == "wavelet":
            x, feature, skips = self.get_feature_with_wavelet(img)
        else:
            assert False

        if contrastive_out == "positive":
            return self.projection_positive(feature)
        elif contrastive_out == "negative":
            return self.projection_negative(feature)
        elif contrastive_out == "feature":
            return feature
        elif r1_regularize:
            torch.set_grad_enabled(True)
            feature.requires_grad_(True)
            return self.out(feature), feature

        gray: Optional[torch.Tensor] = None
        if self.use_unet_decoder and unet_out:
            gray = self.get_unet_out(x, skips, contrastive_out)

        # contrastive
        if contrastive_out == "discriminator":
            torch.set_grad_enabled(True)
            return self.out(feature), gray
        elif contrastive_out == "generator":
            return self.out(feature), gray
        # not contrastive
        elif contrastive_out is None:
            return self.out(feature), gray
        else:
            assert False

