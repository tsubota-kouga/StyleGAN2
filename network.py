
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm
from typing import List, Tuple, Union, Callable, Any, Optional


class Print(nn.Module):
    def forward(self, input):
        print(input.shape)
        return input


class SkipConnection(nn.Module):
    def __init__(self, *layers):
        super(SkipConnection, self).__init__()
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return input + self.net(input)


class Activation(nn.Module):
    activation: Union[nn.Module, Callable[[Any], torch.Tensor]]

    def __init__(self, activation: str, *args, **kwargs):
        super(Activation, self).__init__()
        if activation == "relu":
            self.activation = nn.ReLU(*args, **kwargs)
        elif activation == "prelu":
            self.activation = nn.PReLU(*args, **kwargs)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(*args, **kwargs)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "glu":
            self.activation = lambda x: F.glu(x, dim=1)
        elif activation == "selu":
            self.activation = nn.SELU(*args, **kwargs)
        elif activation == "softplus":
            self.activation = nn.Softplus(*args, **kwargs)
        elif activation == "none":
            self.activation = nn.Identity()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "swish":
            self.activation = Swish()
        elif activation == "mish":
            self.activation = Mish()
        elif activation == "tanhshrink":
            self.activation = nn.Tanhshrink()
        elif activation == "tanhexp":
            self.activation = TanhExp()
        elif activation == "hardswish":
            self.activation = HardSwish()
        else:
            assert False, f"unsupported activation: {activation}"

    def forward(self, input):
        return self.activation(input)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, input):
        return input * torch.tanh(F.softplus(input))


class HardSwish(nn.Module):
    def __init__(self):
        super(HardSwish, self).__init__()

    def forward(self, input):
        return input * F.relu6(input + 3.0) / 6.0


class TanhExp(nn.Module):
    def __init__(self):
        super(TanhExp, self).__init__()

    def forward(self, input):
        return input * input.exp().tanh()


class WeightScaledConv(nn.Conv2d):
    def __init__(self, init: str = "normal", *args, **kwargs):
        super(WeightScaledConv, self).__init__(*args, **kwargs)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)
        if init == "normal":
            nn.init.normal_(self.weight.data)
        elif init == "zeros":
            nn.init.zeros_(self.weight.data)
        else:
            assert False
        self.scale = np.sqrt(2 / (self.in_channels * np.prod(self.kernel_size)))

    def forward(self, input):
        return super(WeightScaledConv, self) \
            ._conv_forward(input, self.weight * self.scale)


class WeightScaledLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(WeightScaledLinear, self).__init__(*args, **kwargs)
        nn.init.zeros_(self.bias.data)
        nn.init.normal_(self.weight.data)
        self.scale = np.sqrt(2 / self.in_features)

    def forward(self, input):
        return F.linear(input, self.weight * self.scale, self.bias)


class WeightScaledConvTrans(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super(WeightScaledConvTrans, self).__init__(*args, **kwargs)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)
        nn.init.normal_(self.weight.data)
        self.scale = np.sqrt(2 / self.in_channels)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(
                input,
                output_size,
                self.stride,
                self.padding,
                self.kernel_size)

        return F.conv_transpose2d(
                input,
                self.weight * self.scale,
                self.bias,
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation)


class Permute(nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.args = args

    def forward(self, input):
        return input.permute(*self.args)


class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super(PixelNorm, self).__init__()
        self.eps = eps

    def forward(self, input):
        return input * torch.rsqrt(
            input.square().mean(dim=1, keepdim=True) + self.eps)


class GeneratorBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 leakiness: float,
                 activation: str,
                 first: bool = False):
        super(GeneratorBlock, self).__init__()
        self.first = first
        if first:
            kernel_size = (4, 3)
            padding_size1 = (kernel_size[1] - 1) // 2
            self.net = nn.Sequential(
                WeightScaledConvTrans(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=1,
                    kernel_size=kernel_size[0],
                    bias=True),
                Activation(
                    activation=activation,
                    negative_slope=leakiness),
                WeightScaledConv(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    padding=padding_size1,
                    kernel_size=kernel_size[1],
                    bias=True),
                Activation(
                    activation=activation,
                    negative_slope=leakiness),
                PixelNorm())
        else:
            kernel_size = (3, 3)
            padding_size0 = (kernel_size[0] - 1) // 2
            padding_size1 = (kernel_size[1] - 1) // 2
            self.upsample = nn.Upsample(scale_factor=2)
            self.net = nn.Sequential(
                WeightScaledConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    padding=padding_size0,
                    kernel_size=kernel_size[0],
                    bias=True),
                Activation(
                    activation=activation,
                    negative_slope=leakiness),
                PixelNorm(),
                WeightScaledConv(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    padding=padding_size1,
                    kernel_size=kernel_size[1],
                    bias=True),
                Activation(
                    activation=activation,
                    negative_slope=leakiness),
                PixelNorm(),
            )

    def forward(self, input):
        if self.first:
            return self.net(input)
        else:
            x = self.upsample(input)
            return self.net(x)


class Generator(nn.Module):
    def __init__(self,
                 leakiness: float,
                 activation: str,
                 channel_info: List[Tuple[int, int]],
                 use_tanh: bool = False):
        super(Generator, self).__init__()
        self.convolutions = nn.ModuleList([
            GeneratorBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                leakiness=leakiness,
                activation=activation,
                first=(idx == 0))
            for idx, (in_channels, out_channels) in enumerate(channel_info)
            ])
        self.use_tanh = use_tanh
        self.to_rgb = nn.ModuleList([
            WeightScaledConv(
                in_channels=out_channels,
                out_channels=3,
                kernel_size=1,
                bias=True)
            for _, out_channels in channel_info
            ])

    def forward(self, x):
        imgs = []
        for convolution, to_rgb in zip(self.convolutions, self.to_rgb):
            x = convolution(x)
            img = to_rgb(x)
            if self.use_tanh:
                imgs.append(img.tanh())
            else:
                imgs.append(img)
        return imgs


class MiniBatchStdDev(nn.Module):
    def __init__(self, eps=1e-8):
        super(MiniBatchStdDev, self).__init__()
        self.eps = eps

    def forward(self, input):
        B, C, *HW = input.shape
        y = input - input.mean(dim=0, keepdim=True)
        y = (y.square().mean(dim=0) + self.eps).sqrt()
        y = y.mean().view(1, 1, 1, 1)
        std = y.repeat(B, 1, *HW)
        return torch.cat((input, std), dim=1)


class DiscriminatorBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 from_rgb_channels: int,
                 leakiness: float,
                 activation: str,
                 size: int,
                 first: bool = False,
                 last: bool = False):
        super(DiscriminatorBlock, self).__init__()
        self.last = last
        hidden_channels = in_channels
        if not first:
            # original flow + generator flow
            in_channels = from_rgb_channels + in_channels + 1
        else:
            in_channels = in_channels + 1
        if last:
            kernel_size = (3, 4)
            padding_size0 = (kernel_size[0] - 1) // 2
            padding_size1 = 0
            self.out = nn.Sequential(
                WeightScaledConv(
                    in_channels=out_channels,
                    out_channels=1,
                    kernel_size=1,
                    bias=True),
                nn.Flatten())
        else:
            kernel_size = (3, 3)
            padding_size0 = (kernel_size[0] - 1) // 2
            padding_size1 = (kernel_size[1] - 1) // 2
            self.out = nn.AvgPool2d(kernel_size=2, stride=2)

        self.net = nn.Sequential(
            MiniBatchStdDev(),
            WeightScaledConv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                padding=padding_size0,
                kernel_size=kernel_size[0],
                bias=True),
            Activation(
                activation=activation,
                negative_slope=leakiness),
            WeightScaledConv(
                in_channels=hidden_channels,
                out_channels=out_channels,
                padding=padding_size1,
                kernel_size=kernel_size[1],
                bias=True),
            Activation(
                activation=activation,
                negative_slope=leakiness),
        )

    def forward(self, x):
        x = self.net(x)
        return self.out(x)


class Discriminator(nn.Module):
    def __init__(self,
                 leakiness: float,
                 activation: str,
                 channel_info: List[Tuple[int, int]],
                 use_sigmoid: bool = False,
                 v: int = 2):
        super(Discriminator, self).__init__()
        from_rgb_channels = 3
        out_channels = channel_info[-1][1]
        self.v = v
        self.use_sigmoid = use_sigmoid
        if v == 2:
            self.from_rgb = WeightScaledConv(
                    in_channels=3,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=True)
        elif v == 1:
            self.from_rgb = nn.ModuleList([
                WeightScaledConv(
                    in_channels=3,
                    out_channels=out_channels if idx == 0 else from_rgb_channels,
                    kernel_size=1,
                    bias=True)
                for idx, (_, out_channels) in enumerate(reversed(channel_info))
                ])
        else:
            assert False
        self.convolutions = nn.ModuleList([
            DiscriminatorBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                from_rgb_channels=from_rgb_channels,
                leakiness=leakiness,
                activation=activation,
                size=2 ** (idx + 2),
                first=(idx == 0),
                last=(idx == len(channel_info) - 1))
            for idx, (out_channels, in_channels) in enumerate(reversed(channel_info))
            ])

    def forward(self, input):
        input = reversed(input)
        for idx, (img, convolution) in enumerate(zip(input, self.convolutions)):
            if idx == 0:
                if self.v == 2:
                    x = self.from_rgb(img)
                elif self.v == 1:
                    x = self.from_rgb[idx](img)
                else:
                    assert False
            else:
                if self.v == 1:
                    img = self.from_rgb[idx](img)
                x = torch.cat((x, img), dim=1)
            x = convolution(x)
        if self.use_sigmoid:
            return torch.sigmoid(x)
        else:
            return x

