import os

import os
# MAR imports:
import torch
import numpy as np
from tqdm import trange
from models import mar
from models.vae import AutoencoderKL
from torchvision.utils import save_image
from util import download
from PIL import Image
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")

download.download_pretrained_vae()

model_type = "mar_huge" #@param ["mar_base", "mar_large", "mar_huge"]
num_sampling_steps_diffloss = 115 #@param {type:"slider", min:1, max:1000, step:1}
if model_type == "mar_base":
  download.download_pretrained_marb(overwrite=False)
  diffloss_d = 6
  diffloss_w = 1024
elif model_type == "mar_large":
  download.download_pretrained_marl(overwrite=False)
  diffloss_d = 8
  diffloss_w = 1280
elif model_type == "mar_huge":
  download.download_pretrained_marh(overwrite=False)
  diffloss_d = 12
  diffloss_w = 1536
else:
  raise NotImplementedError
model = mar.__dict__[model_type](
  buffer_size=64,
  diffloss_d=diffloss_d,
  diffloss_w=diffloss_w,
  num_sampling_steps=str(num_sampling_steps_diffloss)
).to(device)
state_dict = torch.load("pretrained_models/mar/{}/checkpoint-last.pth".format(model_type))["model_ema"]
model.load_state_dict(state_dict)
model.eval() # important!
save_path = "generated_images"
vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path="pretrained_models/vae/kl16.ckpt").cuda().eval()

# Set user inputs:
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
np.random.seed(seed)
num_ar_steps = 64 #@param {type:"slider", min:1, max:256, step:1}
cfg_scale = 1.5 #@param {type:"slider", min:1, max:10, step:0.1}
cfg_schedule = "constant" #@param ["linear", "constant"]
temperature = 1.0 #@param {type:"slider", min:0.9, max:1.1, step:0.01}
num_classes = 1000 #@param {type:"raw"}
samples_per_class = 50 #@param {type:"number"}
batch_size = 25 #@param {type:"number"}

sampled_tokens = None
total_generated_samples = 0
iterations = (samples_per_class // batch_size) + 1
with torch.cuda.amp.autocast():
  for i in trange(num_classes):
    for j in range(iterations):
      sampled_tokens = model.sample_tokens(
      bsz=batch_size, num_iter=num_ar_steps,
      cfg=cfg_scale, cfg_schedule=cfg_schedule,
      labels=torch.full((batch_size,), i, dtype=torch.long, device="cuda"),
      temperature=temperature, progress=False)
      sampled_images = vae.decode(sampled_tokens / 0.2325)
      # save the images into the folder
      for k in range(batch_size):
          os.makedirs(save_path, exist_ok=True)
          save_image(sampled_images[k], os.path.join(save_path, f"{k + total_generated_samples:6d}.png"), normalize=True, value_range=(-1, 1))
      total_generated_samples += batch_size
