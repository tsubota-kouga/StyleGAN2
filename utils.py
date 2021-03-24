
import csv
import datetime
from glob import glob
import math
from multiprocessing import Pool
import os
import random
import re
from typing import List, Optional, Tuple, Union

from PIL import Image
import kornia
from kornia import augmentation, geometry
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse
from skimage import io
import torch
from torch import distributions, nn, jit
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm


def weights_init(m):
    if type(m) is nn.Conv2d or type(m) is nn.ConvTranspose2d:
        nn.init.normal_(m.weight.data, 0.0)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) is nn.Linear:
        nn.init.normal_(m.weight.data, 0.0)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) is nn.Parameter:
        nn.init.normal_(m.weight.data, 0.0)


@jit.script
def hypersphere(z: torch.Tensor, dim: int=1):
    return F.normalize(z, p=2., dim=dim)


class Processor:
    def __init__(self, dest: str, max_level: int, multi_resolution: bool):
        self.dest = dest
        self.max_level = max_level
        self.multi_resolution = multi_resolution

    def __call__(self, path):
        if not self.multi_resolution:
            img = io.imread(path)
            size = 2 ** (self.max_level + 1)
            data = transforms.Compose([
                transforms.Resize(size),
                ])(Image.fromarray(img))
            fname, _ = os.path.splitext(os.path.basename(path))
            torch.save(data, os.path.join(self.dest, fname + ".pt"))
        else:
            img = io.imread(path)
            resized = []
            for i in range(self.max_level):
                size = 2 ** (i + 2)
                tmp = transforms.Compose([
                    transforms.Resize(size),
                    ])(Image.fromarray(img))
                resized.append(tmp)
            fname, _ = os.path.splitext(os.path.basename(path))
            data = resized
            torch.save(data, os.path.join(self.dest, fname + ".pt"))


def preprocess(root: str,
               max_level: int,
               dest: Optional[str] = None,
               multi_resolution: bool = False):
    dest = dest if dest is not None else root
    pool = Pool()
    files = [p for p in glob(os.path.join(root, "**", "*"), recursive=True)
                if re.search(r"\.(png|jpg)", p)]
    processor = Processor(dest=dest, max_level=max_level, multi_resolution=multi_resolution)
    with tqdm(total=len(files)) as t:
        for _ in pool.imap_unordered(processor, files):
            t.update(1)


class FFHQDataset(data.Dataset):
    def __init__(self,
                 root: str,
                 level: int = 0,
                 multi_resolution: bool = False,
                 use_fp16: bool = False,
                 transform = None):
        self.root = root
        self.level = level
        self.multi_resolution = multi_resolution
        fpath = os.path.join(root, "*")
        self.files = list(glob(fpath))
        self.use_fp16 = use_fp16
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fname = self.files[idx]
        if not self.multi_resolution:
            img = torch.load(fname)
            if self.transform is not None:
                img = self.transform(img)
            if self.use_fp16:
                img = np.array(img).astype(np.float16)
            else:
                img = np.array(img).astype(np.float32)
            return img
        else:
            imgs = torch.load(fname)
            if self.transform is not None:
                imgs = [self.transform(img) for img in imgs]
            if self.use_fp16:
                imgs = [np.array(img).astype(np.float16) for img in imgs]
            else:
                imgs = [np.array(img).astype(np.float32) for img in imgs]
            return imgs


class CelebAHQDataset(data.Dataset):
    def __init__(self,
                 root: str,
                 attr_file: Optional[str] = None,
                 valid_attr: List[str] = None,
                 level: int = 0,
                 multi_resolution: bool = False,
                 use_fp16: bool = False,
                 transform = None):
        self.root = root
        self.level = level
        self.multi_resolution = multi_resolution
        fpath = os.path.join(root, "*")
        self.files = list(glob(fpath))
        self.use_fp16 = use_fp16
        self.transform = transform
        self.attr = None
        self.valid_attr = valid_attr
        if attr_file is not None:
            self.read_attr(attr_file)

    def __len__(self):
        return len(self.files)

    def read_attr(self, attr_file: str):
        attr_dict = {}
        with open(attr_file, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=" ")
            for i, row in enumerate(reader):
                if i == 0:
                    # N
                    print(*row, "classes")
                    continue
                elif i == 1:
                    # classes
                    cls = list(filter(lambda x: x != "", row))
                    cls = [cls.index(c) for c in self.valid_attr]
                    continue
                fname, *cls = row
                tmp = list(map(lambda x: int(x == "1"), cls))
                attr_dict[fname] = [tmp[i] for i in cls]
        self.attr = attr_dict

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fname = self.files[idx]
        last_fname = os.path.basename(fname)
        attr = None
        if self.attr is not None:
            attr = self.attr[last_fname]
        if not self.multi_resolution:
            img = torch.load(fname)
            if self.transform is not None:
                img = self.transform(img)
            if self.use_fp16:
                img = np.array(img).astype(np.float16)
            else:
                img = np.array(img).astype(np.float32)

            if attr is not None:
                return img, attr
            else:
                return img
        else:
            imgs = torch.load(fname)
            if self.transform is not None:
                imgs = [self.transform(img) for img in imgs]
            if self.use_fp16:
                imgs = [np.array(img).astype(np.float16) for img in imgs]
            else:
                imgs = [np.array(img).astype(np.float32) for img in imgs]

            if attr is not None:
                return imgs, attr
            else:
                return imgs


def update_average(target, source, beta):
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    toggle_grad(target, False)
    toggle_grad(source, False)
    param_dict_src = dict(source.named_parameters())
    for p_name, p_target in target.named_parameters():
        p_source = param_dict_src[p_name]
        if p_source.isnan().any():
            print("detect nan")
            continue
        assert p_source is not p_target
        p_target.copy_(beta * p_target + (1.0 - beta) * p_source)
    toggle_grad(target, True)
    toggle_grad(source, True)


def adjust_dynamic_range(data, in_data_range=(-1, 1), out_data_range=(0, 1)):
    if in_data_range != out_data_range:
        scale = (out_data_range[1] - out_data_range[0]) / (in_data_range[1] - in_data_range[0])
        bias = (out_data_range[1] - out_data_range[0]) * scale
        data = data * scale + bias
    return torch.clamp(data, min=out_data_range[0], max=out_data_range[1])


def load_generator(
    style_mapper,
    generator,
    discriminator=None,
    eval_mode: bool = False,
    path: str = None,
    ext: str = "model") -> Tuple[nn.Module, ...]:
    if path is not None:
        print(f"load {path}")
        state = torch.load(path, map_location=lambda storage, loc: storage)
    else:
        path = glob("model/*")
        path = list(map(lambda p: os.path.splitext(p)[0], path))
        path = list(map(lambda p: os.path.basename(p), path))
        date = list(map(lambda d: datetime.datetime.fromisoformat(d), path))
        latest = sorted(date)[-1]
        print(f"load latest {latest}")
        state = torch.load(f"model/{latest}.{ext}", map_location=lambda storage, loc: storage)

    if eval_mode and "generator_" in state.keys():
        style_mapper.load_state_dict(state["style_mapper_"])
        generator.load_state_dict(state["generator_"])
        print("load generator_")
    else:
        style_mapper.load_state_dict(state["style_mapper"])
        generator.load_state_dict(state["generator"])
        print("load generator")
    if discriminator is not None:
        discriminator.load_state_dict(state["discriminator"])
        return style_mapper, generator, discriminator
    return style_mapper, generator


def uniform(
    size: Tuple[int, ...],
    range: Tuple[float, float],
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype=torch.float32):
    return (range[0] - range[1]) * torch.rand(size=size, device=device, dtype=dtype) + range[1]


class AdaptiveAugmentation(nn.Module):
    def __init__(self, target_rt: float = 0.6, speed: float = 2e-3, p: float = 0.0):
        super(AdaptiveAugmentation, self).__init__()
        self.p = p
        self.target_rt = target_rt
        self.speed = speed
        self.halfnorm = distributions.half_normal.HalfNormal(0.1 ** 2)
        self.cutout = augmentation.RandomErasing(p=1.0, scale=(0.2, 0.25), ratio=(0.8, 1/0.8))
        # self.dwt = DWTForward(wave="sym2", mode="reflect")
        # self.idwt = DWTInverse(wave="sym2", mode="reflect")

    # def upsample(self, img: torch.Tensor):
    #     yl, yh = self.dwt(img)  # [B, C, H, W]

    def set_p(self, r: float, nimg: int):
        self.p += nimg * (r - self.target_rt) * self.speed / 1000
        self.p = np.clip(self.p, a_min=0.0, a_max=1.0)

    def forward(self, *input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.augmentation(*input)

    def augmentation(self, *x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        B, _, H, W = x[0].shape
        device = x[0].device
        dtype = x[0].dtype
        G_inv = torch.eye(n=3, device=device, dtype=dtype).repeat(B, 1, 1)

        with torch.no_grad():
            # x-flip
            is_valid = torch.rand((B, ), device=device, dtype=dtype) < 1 - self.p
            i = torch.tensor(random.choices([0., 1.], k=B), device=device, dtype=dtype)
            i[is_valid] = 0.
            G_inv = scale2d_inv(G_inv, s=(1 - 2 * i, torch.ones((B, ), device=device)))

            # pi/2 rot
            is_valid = torch.rand((B, ), device=device, dtype=dtype) < 1 - self.p
            i = torch.tensor(random.choices([0., 1.], k=B), device=device, dtype=dtype)
            i[is_valid] = 0.
            G_inv = rotate2d_inv(G_inv, theta=math.pi / 2 * i)

            # integer translation
            is_valid = torch.rand((B, ), device=device, dtype=dtype) < 1 - self.p
            t_x = uniform(size=(B, ), range=(-0.125, 0.125), device=device)
            t_y = uniform(size=(B, ), range=(-0.125, 0.125), device=device)
            t_x[is_valid], t_y[is_valid] = 0., 0.
            G_inv = translate2d_inv(G_inv, t=(torch.round(t_x * W), torch.round(t_y * H)))

            # isotropic scaling
            is_valid = torch.rand((B, ), device=device, dtype=dtype) < 1 - self.p
            s = torch.tensor(
                np.random.lognormal(size=(B, ), sigma=(0.2 * math.log(2)) ** 2),
                dtype=dtype,
                device=device)
            s[is_valid] = 1.
            G_inv = scale2d_inv(G_inv, s=(s, s))

            p_rot = 1 - math.sqrt(1 - self.p)
            # before anisotropic scaling
            is_valid = torch.rand((B, ), device=device, dtype=dtype) < 1 - p_rot
            theta = uniform(size=(B, ), range=(-math.pi, math.pi), device=device, dtype=dtype)
            theta[is_valid] = 0.
            G_inv = rotate2d_inv(G_inv, theta=theta)
            # anisotropic scaling
            is_valid = torch.rand((B, ), device=device, dtype=dtype) < 1 - self.p
            s = torch.tensor(
                np.random.lognormal(size=(B, ), sigma=(0.2 * math.log(2)) ** 2),
                dtype=dtype,
                device=device)
            s[is_valid] = 1.
            G_inv = scale2d_inv(G_inv, s=(s, 1 / s))
            # after anisotropic scaling
            is_valid = torch.rand((B, ), device=device, dtype=dtype) < 1 - p_rot
            theta = uniform(size=(B, ), range=(-math.pi, math.pi), device=device, dtype=dtype)
            theta[is_valid] = 0.
            G_inv = rotate2d_inv(G_inv, theta=theta)

            # fractional translation
            is_valid = torch.rand((B, ), device=device, dtype=dtype) < 1 - self.p
            t_x = torch.normal(mean=0.0, std=0.125 ** 2, size=(B, ), device=device, dtype=dtype)
            t_y = torch.normal(mean=0.0, std=0.125 ** 2, size=(B, ), device=device, dtype=dtype)
            t_x[is_valid], t_y[is_valid] = 0., 0.
            G_inv = translate2d_inv(G_inv, t=(t_x, t_y))

            width, height = x[0].shape[-2:]
            cx, cy = width / 2, height / 2
            G_inv = translate2d_inv(
                G_inv,
                t=(torch.tensor([cx - 1/2,], device=device, dtype=dtype).expand(B),
                   torch.tensor([cy - 1/2,], device=device, dtype=dtype).expand(B)))
            G_inv = translate2d(
                G_inv,
                t=(torch.tensor([cx - 1/2,], device=device, dtype=dtype).expand(B),
                   torch.tensor([cy - 1/2,], device=device, dtype=dtype).expand(B)))
            G_inv = scale2d_inv(
                G_inv,
                s=(torch.tensor([2.,], device=device).expand(B),
                   torch.tensor([2.,], device=device).expand(B)))
            G_inv = scale2d(
                G_inv,
                s=(torch.tensor([2.,], device=device).expand(B),
                   torch.tensor([2.,], device=device).expand(B)))

        x = tuple(F.interpolate(_x, scale_factor=(2, 2)) for _x in x)
        x = tuple(geometry.warp_affine(
            _x, G_inv[:, :2],
            dsize=(2 * height, 2 * width),
            padding_mode="reflection") for _x in x)
        x = tuple(F.avg_pool2d(_x, kernel_size=(2, 2)) for _x in x)

        with torch.no_grad():
            C = torch.eye(n=4, device=device, dtype=dtype).repeat(B, 1, 1)

            # brightness
            is_valid = torch.rand((B, ), device=device, dtype=dtype) < 1 - self.p
            b = torch.normal(mean=0.0, std=0.2 ** 2, size=(B, ), device=device)
            b[is_valid] = 0.
            C = translate3d(C, t=(b, b, b))

            # contrast
            is_valid = torch.rand((B, ), device=device, dtype=dtype) < 1 - self.p
            c = torch.tensor(
                np.random.lognormal(size=(B, ), sigma=(0.2 * math.log(2)) ** 2),
                dtype=dtype,
                device=device)
            c[is_valid] = 1.
            C = scale3d(C, s=(c, c, c))

            v = torch.tensor([[1., 1., 1., 0.]], device=device, dtype=dtype) / math.sqrt(3)
            vTv = (v.T * v).expand(B, 4, 4)
            # luma flip
            is_valid = torch.rand((B, ), device=device, dtype=dtype) < 1 - self.p
            i = torch.tensor(random.choices([0., 1.], k=B), device=device, dtype=dtype)[:, None, None]
            i[is_valid] = 0.
            C = (torch.eye(n=4, device=device, dtype=dtype).repeat(B, 1, 1) - 2 * vTv * i).bmm(C)

            # hue rotation
            is_valid = torch.rand((B, ), device=device, dtype=dtype) < 1 - self.p
            theta = uniform(size=(B,), range=(-math.pi, math.pi), device=device, dtype=dtype)
            theta[is_valid] = 0.
            C = rotate3d(C, theta=theta, axis=v.expand(B, 4))

            # saturation rotation
            is_valid = torch.rand((B, ), device=device, dtype=dtype) < 1 - self.p
            s = torch.tensor(
                np.random.lognormal(size=(B, ), sigma=(0.2 * math.log(2)) ** 2),
                dtype=dtype,
                device=device)[:, None, None]
            s[is_valid] = 1.
            C = (vTv + (torch.eye(n=4, device=device, dtype=dtype).repeat(B, 1, 1) - vTv) * s).bmm(C)

        a_channel = torch.ones((B, 1, H, W), device=device, dtype=dtype)
        x = tuple(torch.cat((_x, a_channel), dim=1) for _x in x)  # [B, C, H, W]
        x = tuple(C.bmm(_x.reshape(B, 4, -1)).reshape(B, 4, H, W) for _x in x)  # [B, C, H, W]
        x = tuple(_x[:, :3] for _x in x)

        # if random.uniform(0., 1.) < 1 - self.p:  # band pass
        # g = torch.ones(B, 4, dtype=x.dtype, device=device)
        # lamb = torch.tensor([[10. / 13, 1. / 13, 1. / 13, 1 / 13]], dtype=x.dtype, device=device)
        # t = torch.tensor(
        #     np.random.lognormal(size=(B, 4), sigma=math.log(2) ** 2),
        #     dtype=x.dtype,
        #     device=device)
        # t = t / (t.square() * lamb).sum(dim=1, keepdim=True).sqrt()
        # is_valid = torch.rand((B, ), device=device, dtype=dtype) < self.p
        # t[is_valid] = 1.
        # g = g * t
        #
        # yl: torch.Tensor
        # yh: List[torch.Tensor]
        # yl, yh = self.dwt(y)
        # y = torch.cat((yl.unsqueeze(2), yh[0]), dim=2)  # [B, C, 4, H', W']
        # y = y * g[:, None, :, None, None]
        # yl, yh = torch.split(y, split_size_or_sections=(1, 3), dim=2)
        # y = self.idwt((yl.squeeze(2), (yh,)))

        # additive rgb noise
        is_valid = torch.rand((B, ), device=device, dtype=dtype) < 1 - self.p
        with torch.no_grad():
            std: torch.Tensor = self.halfnorm.sample(sample_shape=(B,)).to(device)
            std = std[:, None, None, None].expand_as(x[0])
            noise = torch.normal(mean=0., std=std)
        noise[is_valid] = 0.
        x = tuple(_x + noise for _x in x)

        # cutout
        is_valid = torch.rand((B, ), device=device, dtype=dtype) < 1 - self.p
        filter = self.cutout(torch.ones_like(x[0]))
        filter[is_valid] = 1.
        x = tuple(_x * filter for _x in x)

        return x



def scale2d(input: torch.Tensor, s: Tuple[torch.Tensor, torch.Tensor], from_right: bool = False) -> torch.Tensor:
    '''
    s: sx, sy
    '''
    B = input.shape[0]
    sx, sy = s
    operation = torch.eye(n=3, device=input.device, dtype=input.dtype).repeat(B, 1, 1)
    operation[:, 0, 0] = sx
    operation[:, 1, 1] = sy
    if from_right:
        return input.bmm(operation)
    else:
        return operation.bmm(input)

def scale2d_inv(input: torch.Tensor, s: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    sx, sy = s
    return scale2d(input, s=(1 / sx, 1 / sy), from_right=True)

def scale3d(input: torch.Tensor, s: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    '''
    s: sx, sy, sy
    '''
    B = input.shape[0]
    sx, sy, sz = s
    operation = torch.eye(n=4, device=input.device, dtype=input.dtype).repeat(B, 1, 1)
    operation[:, 0, 0] = sx
    operation[:, 1, 1] = sy
    operation[:, 2, 2] = sz
    return operation.bmm(input)

def rotate2d(input: torch.Tensor, theta: torch.Tensor, from_right: bool = False) -> torch.Tensor:
    B = input.shape[0]
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    operation = torch.eye(n=3, device=input.device, dtype=input.dtype).repeat(B, 1, 1)
    operation[:, 0, 0] = cos_theta
    operation[:, 0, 1] = -sin_theta
    operation[:, 1, 0] = sin_theta
    operation[:, 1, 1] = cos_theta
    if from_right:
        return input.bmm(operation)
    else:
        return operation.bmm(input)

def rotate2d_inv(input: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return rotate2d(input, -theta, from_right=True)

def rotate3d(input: torch.Tensor, theta: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    B = input.shape[0]
    n1, n2, n3, _ = torch.split(axis, split_size_or_sections=1, dim=1)
    n1.squeeze_(1)
    n2.squeeze_(1)
    n3.squeeze_(1)
    cos_theta = torch.cos(theta)
    one_minus_cos_theta = 1 - cos_theta
    sin_theta = torch.sin(theta)
    operation = torch.eye(n=4, device=input.device, dtype=input.dtype).repeat(B, 1, 1)
    operation[:, 0, 0] = n1 * n1 * one_minus_cos_theta + cos_theta
    operation[:, 0, 1] = n1 * n2 * (one_minus_cos_theta) - n3 * sin_theta
    operation[:, 0, 2] = n1 * n3 * one_minus_cos_theta + n2 * sin_theta

    operation[:, 1, 0] = n1 * n2 * one_minus_cos_theta + n3 * sin_theta
    operation[:, 1, 1] = n2 * n2 * one_minus_cos_theta + cos_theta
    operation[:, 1, 2] = n2 * n3 * one_minus_cos_theta - n1 * sin_theta

    operation[:, 2, 0] = n1 * n3 * one_minus_cos_theta - n2 * sin_theta
    operation[:, 2, 1] = n2 * n3 * one_minus_cos_theta + n1 * sin_theta
    operation[:, 2, 2] = n3 * n3 * one_minus_cos_theta + cos_theta
    return operation.bmm(input)

def translate2d(input: torch.Tensor, t: Tuple[torch.Tensor, torch.Tensor], from_right: bool = False) -> torch.Tensor:
    '''
    t: tx, ty
    '''
    B = input.shape[0]
    tx, ty = t
    operation = torch.eye(n=3, device=input.device, dtype=input.dtype).repeat(B, 1, 1)
    operation[:, 0, 2] = tx
    operation[:, 1, 2] = ty
    if from_right:
        return input.bmm(operation)
    else:
        return operation.bmm(input)

def translate2d_inv(input: torch.Tensor, t: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    tx, ty = t
    return translate2d(input, t=(-tx, -ty), from_right=True)

def translate3d(input: torch.Tensor, t: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    '''
    t: tx, ty, tz
    '''
    B = input.shape[0]
    tx, ty, tz = t
    operation = torch.eye(n=4, device=input.device, dtype=input.dtype).repeat(B, 1, 1)
    operation[:, 0, 3] = tx
    operation[:, 1, 3] = ty
    operation[:, 2, 3] = tz
    return operation.bmm(input)


class SimCLRAugmentation(nn.Module):
    def __init__(self, resolution: int):
        super(SimCLRAugmentation, self).__init__()
        self.transform = nn.Sequential(
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8)
                ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=resolution * 0.1)
                ], p=0.5),
            transforms.RandomErasing(p=0.5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.transform(input)

