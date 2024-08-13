import os
# MAR imports:
import torch
import numpy as np
from tqdm import trange
from models import mar
from models.vae import AutoencoderKL
from torchvision.utils import save_image
from util import download
import lightning as l
import argparse
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")

download.download_pretrained_vae()

def main(args):
  if args.tf32:
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')
    print(f"Fast inference mode is enabled🏎️🏎️🏎️. TF32: {tf32}")
  else:
    print("Fast inference mode is disabled🐢🐢🐢, you may enable it by passing the '--fast-inference' flag!")
  l.seed_everything(args.seed)
  if args.model_type == "mar_base":
    download.download_pretrained_marb(overwrite=False)
    diffloss_d = 6
    diffloss_w = 1024
  elif args.model_type == "mar_large":
    download.download_pretrained_marl(overwrite=False)
    diffloss_d = 8
    diffloss_w = 1280
  elif args.model_type == "mar_huge":
    download.download_pretrained_marh(overwrite=False)
    diffloss_d = 12
    diffloss_w = 1536
  else:
    raise NotImplementedError
  model = mar.__dict__[args.model_type](
    buffer_size=64,
    diffloss_d=diffloss_d,
    diffloss_w=diffloss_w,
    num_sampling_steps=str(args.num_sampling_steps_diffloss)
  ).to(device)
  state_dict = torch.load("pretrained_models/mar/{}/checkpoint-last.pth".format(args.model_type))["model_ema"]
  model.load_state_dict(state_dict)
  model = torch.compile(model)
  model.eval() # important!
  save_path = "generated_images"
  vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path="pretrained_models/vae/kl16.ckpt").cuda().eval()

  sampled_tokens = None
  total_generated_samples = 0
  iterator = args.samples_per_class // args.batch_size
  iterations = (iterator + 1) if (args.samples_per_class % args.batch_size != 0) else iterator
  with torch.cuda.amp.autocast():
    for i in trange(args.num_classes):
      for j in range(iterations):
        sampled_tokens = model.sample_tokens(
        bsz=args.batch_size, num_iter=args.num_ar_steps,
        cfg=args.cfg_scale, cfg_schedule=args.cfg_schedule,
        labels=torch.full((args.batch_size,), i, dtype=torch.long, device="cuda"),
        temperature=args.temperature, progress=False)
        sampled_images = vae.decode(sampled_tokens / 0.2325)
        # save the images into the folder
        for k in range(args.batch_size):
            os.makedirs(save_path, exist_ok=True)
            save_image(sampled_images[k], os.path.join(save_path, f"{k + total_generated_samples:6d}.png"), normalize=True, value_range=(-1, 1))
        total_generated_samples += args.batch_size

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-type", type=str, default="mar_huge", choices=["mar_base", "mar_large", "mar_huge"], help="The model type to use.")
  parser.add_argument("--num-sampling-steps-diffloss", type=int, default=115, help="The number of sampling steps for the diffloss.")
  parser.add_argument("--seed", type=int, default=0, help="The random seed")
  parser.add_argument("--num-ar-steps", type=int, default=64, help="The number of autoregressive steps.")
  parser.add_argument("--cfg-scale", type=float, default=1.5, help="The scale of the configuration.")
  parser.add_argument("--cfg-schedule", type=str, default="constant", choices=["linear", "constant"], help="The schedule of the configuration.")
  parser.add_argument("--temperature", type=float, default=1.0, help="The temperature.")
  parser.add_argument("--num-classes", type=int, default=1000, help="The number of classes.")
  parser.add_argument("--samples-per-class", type=int, default=50, help="The number of samples per class.")
  parser.add_argument("--batch-size", type=int, default=25, help="The batch size.")
  parser.add_argument("--tf32", type=bool, action="store_true", help="Whether to use TF32.")
  args = parser.parse_args()
  main(args)
