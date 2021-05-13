
import sys
import argparse

import cv2
import torch
from torch import cuda, backends
from torchvision import utils

from hyperparam import HyperParam as hp
from network import GeneratorV2, LatentLayers
from utils import adjust_dynamic_range, load_generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    args = parser.parse_args()
    backends.cudnn.benchmark = True
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
    generator = GeneratorV2(
        style_channels=hp.latent_dim,
        activation=hp.activationG,
        activation_args=hp.activationG_args,
        channel_info=hp.channel_info,
        eps=hp.eps,
        use_scale=hp.use_scaleG,
        noise_mode=hp.noise_mode,
        mode=hp.Gmode,
        use_fp16=hp.use_fp16,
        ).to(hp.device, non_blocking=hp.non_blocking)

    style_mapper, generator = load_generator(
        style_mapper, generator, eval_mode=True, path=args.path, ext="model")
    style_mapper = style_mapper.to(hp.device).eval()
    generator = generator.to(hp.device).eval()

    with torch.no_grad():
        while True:
            # noise = torch.randn(16, hp.latent_dim, 1, 1)
            # if hp.use_fp16:
            #     noise = noise.half()
            # noise = noise.to(device=hp.device, memory_format=torch.channels_last)
            noise = torch.randn(
                    16, hp.latent_dim, 1, 1,
                    device=hp.device,
                    dtype=torch.float16 if hp.use_fp16 else torch.float)
            if hp.use_fp16:
                with cuda.amp.autocast():
                    styles = style_mapper(noise, truncation_trick_rate=hp.truncation_trick_rate)
                    imgs = generator(styles=styles)[-1].cpu().float()
            else:
                styles = style_mapper(noise, truncation_trick_rate=hp.truncation_trick_rate)
                imgs = generator(styles=styles)[-1].cpu()
            imgs = adjust_dynamic_range(imgs)
            imgs = utils.make_grid(imgs, nrow=4, padding=1, normalize=True)
            imgs = imgs.detach().numpy().transpose([1, 2, 0])
            imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
            cv2.imshow("generated images", imgs)
            cmd = cv2.waitKey(0)
            if cmd == ord("q"):
                cv2.destroyAllWindows()
                break
            elif cmd == ord("c"):
                pass
            else:
                print(f"cmd must be 'q' or 'c', invalid: {cmd}")
                print("default: continue")
            cv2.destroyAllWindows()

