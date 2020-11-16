
import sys
import torch
import torchvision
from torchvision import utils
import cv2

from network import Generator
from hyperparam import HyperParam as hp
from utils import hypersphere, load_generator, adjust_dynamic_range

if hp.use_fp16:
    scaler = torch.cuda.amp.GradScaler()

generator = Generator(
    leakiness=hp.leakiness,
    activation=hp.activationG,
    channel_info=hp.channel_info,
    use_tanh=hp.use_tanh)

path = None
if len(sys.argv) > 1:
    path = sys.argv[1]

generator = load_generator(generator, path=path, ext="model").to(hp.device).eval()

with torch.no_grad():
    while True:
        noise = hypersphere(torch.randn(16, hp.latent_dim, 1, 1))
        if hp.use_fp16:
            noise = noise.half()
        noise = noise.to(device=hp.device, memory_format=torch.channels_last)
        if hp.use_fp16:
            with torch.cuda.amp.autocast():
                imgs = generator(noise)[-1].cpu().float()
        else:
            imgs = generator(noise)[-1].cpu()
        imgs = adjust_dynamic_range(imgs)
        imgs = utils.make_grid(imgs, padding=1, normalize=True)
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

