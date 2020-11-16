
import os
import shutil
import torch
import csv
import numpy as np
from multiprocessing import Pool
import datetime
import re
from glob import glob
from tqdm import tqdm
from skimage import io
from torch import nn
from torch.utils import data
from torchvision import transforms
from PIL import Image
from typing import Optional, List


def weights_init(m):
    if type(m) is nn.Conv2d or type(m) is nn.ConvTranspose2d:
        nn.init.normal_(m.weight.data, 0.0)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) is nn.Linear:
        nn.init.normal_(m.weight.data, 0.0)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) is nn.Parameter:
        nn.init.normal_(m.weight.data, 0.0)


    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    #     nn.init.normal_(m.weight.data, 0.0, 0.02)
    # elif classname.find('BatchNorm') != -1:
    #     nn.init.normal_(m.weight.data, 1.0, 0.02)
    #     nn.init.constant_(m.bias.data, 0)


@torch.jit.script
def hypersphere(z: torch.Tensor, dim: int=1):
    return z.div(z.norm(p=2, dim=dim, keepdim=True)) * z.size(dim) ** 0.5


class Processor:
    def __init__(self, dest: str, max_level: int):
        self.dest = dest
        self.max_level = max_level

    def __call__(self, path):
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
               dest: Optional[str] = None):
    dest = dest if dest is not None else root
    pool = Pool()
    files = [p for p in glob(os.path.join(root, "**", "*"), recursive=True)
                if re.search("\.(png|jpg)", p)]
    processor = Processor(dest=dest, max_level=max_level)
    with tqdm(total=len(files)) as t:
        for _ in pool.imap_unordered(processor, files):
            t.update(1)


class FFHQDataset(data.Dataset):
    def __init__(self,
                 root: str,
                 level: int = 0,
                 use_fp16: bool = False,
                 transform = None):
        self.root = root
        self.level = level
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
                 use_fp16: bool = False,
                 transform = None):
        self.root = root
        self.level = level
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
        imgs = torch.load(fname)
        attr = None
        if self.attr is not None:
            attr = self.attr[last_fname]
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


def load_generator(generator, discriminator=None, path: str = None, ext: str = "model"):
    if path is not None:
        print(f"load {path}")
        state = torch.load(path, map_location=lambda storage, loc: storage)
        generator.load_state_dict(state["generator"])
        if discriminator is not None:
            discriminator.load_state_dict(state["discriminator"])
            return generator, discriminator
        return generator
    path = glob("model/*")
    path = list(map(lambda p: os.path.splitext(p)[0], path))
    path = list(map(lambda p: os.path.basename(p), path))
    date = list(map(lambda d: datetime.datetime.fromisoformat(d), path))
    latest = sorted(date)[-1]
    print(f"load latest {latest}")
    state = torch.load(f"model/{latest}.{ext}", map_location=lambda storage, loc: storage)
    generator.load_state_dict(state["generator"])
    if discriminator is not None:
        discriminator.load_state_dict(state["discriminator"])
        return generator, discriminator
    return generator

