# Autoregressive Image Generation without Vector Quantization <br><sub>Official PyTorch Implementation</sub>

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2406.11838-b31b1b.svg)](https://arxiv.org/abs/2406.11838)&nbsp;
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/autoregressive-image-generation-without/image-generation-on-imagenet-256x256)](https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?p=autoregressive-image-generation-without)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/LTH14/mar/blob/main/demo/run_mar.ipynb)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-mar-yellow)](https://huggingface.co/jadechoghari/mar)&nbsp;

<p align="center">
  <img src="demo/visual.png" width="720">
</p>

This is a PyTorch/GPU implementation of the paper [Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838):

```
@article{li2024autoregressive,
  title={Autoregressive Image Generation without Vector Quantization},
  author={Li, Tianhong and Tian, Yonglong and Li, He and Deng, Mingyang and He, Kaiming},
  journal={arXiv preprint arXiv:2406.11838},
  year={2024}
}
```

This repo contains:

* 🪐 A simple PyTorch implementation of [MAR](models/mar.py) and [DiffLoss](models/diffloss.py)
* ⚡️ Pre-trained class-conditional MAR models trained on ImageNet 256x256
* 💥 A self-contained [Colab notebook](http://colab.research.google.com/github/LTH14/mar/blob/main/demo/run_mar.ipynb) for running various pre-trained MAR models
* 🛸 An MAR+DiffLoss [training and evaluation script](main_mar.py) using PyTorch DDP
* 🎉 Also checkout our [Hugging Face model cards](https://huggingface.co/jadechoghari/mar) and [Gradio demo](https://huggingface.co/spaces/jadechoghari/mar) (thanks @jadechoghari).

## Preparation

### Dataset
Download [ImageNet](http://image-net.org/download) dataset, and place it in your `IMAGENET_PATH`.

### Installation

Download the code:
```
git clone https://github.com/LTH14/mar.git
cd mar
```

A suitable [conda](https://conda.io/) environment named `mar` can be created and activated with:

```
conda env create -f environment.yaml
conda activate mar
```

Download pre-trained VAE and MAR models:

```
python util/download.py
```

For convenience, our pre-trained MAR models can be downloaded directly here as well:

| MAR Model                                                              | FID-50K | Inception Score | #params | 
|------------------------------------------------------------------------|---------|-----------------|---------|
| [MAR-B](https://www.dropbox.com/scl/fi/f6dpuyjb7fudzxcyhvrhk/checkpoint-last.pth?rlkey=a6i4bo71vhfo4anp33n9ukujb&dl=0) | 2.31    | 281.7           | 208M    |
| [MAR-L](https://www.dropbox.com/scl/fi/pxacc5b2mrt3ifw4cah6k/checkpoint-last.pth?rlkey=m48ovo6g7ivcbosrbdaz0ehqt&dl=0) | 1.78    | 296.0           | 479M    |
| [MAR-H](https://www.dropbox.com/scl/fi/1qmfx6fpy3k7j9vcjjs3s/checkpoint-last.pth?rlkey=4lae281yzxb406atp32vzc83o&dl=0) | 1.55    | 303.7           | 943M    |

### (Optional) Caching VAE Latents

Given that our data augmentation consists of simple center cropping and random flipping, 
the VAE latents can be pre-computed and saved to `CACHED_PATH` to save computations during MAR training:

```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_cache.py \
--img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 \
--batch_size 128 \
--data_path ${IMAGENET_PATH} --cached_path ${CACHED_PATH}
```

## Usage

### Demo
Run our interactive visualization [demo](http://colab.research.google.com/github/LTH14/mar/blob/main/demo/run_mar.ipynb) using Colab notebook!

### Training
Script for the default setting (MAR-L, DiffLoss MLP with 3 blocks and a width of 1024 channels, 400 epochs):
```
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main_mar.py \
--img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model mar_large --diffloss_d 3 --diffloss_w 1024 \
--epochs 400 --warmup_epochs 100 --batch_size 64 --blr 1.0e-4 --diffusion_batch_mul 4 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH}
```
- Training time is ~1d7h on 32 H100 GPUs with `--batch_size 64`.
- Add `--online_eval` to evaluate FID during training (every 40 epochs).
- (Optional) To train with cached VAE latents, add `--use_cached --cached_path ${CACHED_PATH}` to the arguments. 
Training time with cached latents is ~1d11h on 16 H100 GPUs with `--batch_size 128` (nearly 2x faster than without caching).
- (Optional) To save GPU memory during training by using gradient checkpointing (thanks to @Jiawei-Yang), add `--grad_checkpointing` to the arguments. 
Note that this may slightly reduce training speed.

### Evaluation (ImageNet 256x256)

Evaluate MAR-B (DiffLoss MLP with 6 blocks and a width of 1024 channels, 800 epochs) with classifier-free guidance:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_mar.py \
--model mar_base --diffloss_d 6 --diffloss_w 1024 \
--eval_bsz 256 --num_images 50000 \
--num_iter 256 --num_sampling_steps 100 --cfg 2.9 --cfg_schedule linear --temperature 1.0 \
--output_dir pretrained_models/mar/mar_base \
--resume pretrained_models/mar/mar_base \
--data_path ${IMAGENET_PATH} --evaluate
```

Evaluate MAR-L (DiffLoss MLP with 8 blocks and a width of 1280 channels, 800 epochs) with classifier-free guidance:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_mar.py \
--model mar_large --diffloss_d 8 --diffloss_w 1280 \
--eval_bsz 256 --num_images 50000 \
--num_iter 256 --num_sampling_steps 100 --cfg 3.0 --cfg_schedule linear --temperature 1.0 \
--output_dir pretrained_models/mar/mar_large \
--resume pretrained_models/mar/mar_large \
--data_path ${IMAGENET_PATH} --evaluate
```

Evaluate MAR-H (DiffLoss MLP with 12 blocks and a width of 1536 channels, 800 epochs) with classifier-free guidance:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_mar.py \
--model mar_huge --diffloss_d 12 --diffloss_w 1536 \
--eval_bsz 128 --num_images 50000 \
--num_iter 256 --num_sampling_steps 100 --cfg 3.2 --cfg_schedule linear --temperature 1.0 \
--output_dir pretrained_models/mar/mar_huge \
--resume pretrained_models/mar/mar_huge \
--data_path ${IMAGENET_PATH} --evaluate
```

- Set `--cfg 1.0 --temperature 0.95` to evaluate without classifier-free guidance.
- Generation speed can be significantly increased by reducing the number of autoregressive iterations (e.g., `--num_iter 64`).

## Acknowledgements
We thank Congyue Deng and Xinlei Chen for helpful discussion. We thank
Google TPU Research Cloud (TRC) for granting us access to TPUs, and Google Cloud Platform for
supporting GPU resources.

A large portion of codes in this repo is based on [MAE](https://github.com/facebookresearch/mae), [MAGE](https://github.com/LTH14/mage) and [DiT](https://github.com/facebookresearch/DiT).

## Contact

If you have any questions, feel free to contact me through email (tianhong@mit.edu). Enjoy!