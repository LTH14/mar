import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

from diffusion import create_diffusion

import einops
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from torch.nn import functional as F

from typing import Optional
# import pytorch_lightning as L

class DiffLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, grad_checkpointing=False):
        super(DiffLoss, self).__init__()
        self.in_channels = target_channels
        # SimpleMLPAdaLN, HopfieldMLPAdaLN
        self.net = HopfieldMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2,  # for vlb loss
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing
        )
        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")

    def forward(self, target, z, mask=None, bsz=None, seq_len=None):
        t = torch.randint(0, self.train_diffusion.num_timesteps, target.shape[:-1], device=target.device)

        # print(f"Diffusion Loss - target: {target.shape}, z: {z.shape}, t: {t.shape}, mask: {mask.shape}")

        # bsz, seq_len, _ = z.shape
        # if t.dim() > 1:
        #     t = t.flatten(start_dim=0, end_dim=1)
        # if target.dim() > 2:
        #     target = target.flatten(start_dim=0, end_dim=1)
        # if z.dim() > 2:
        #     z = z.flatten(start_dim=0, end_dim=1)

        model_kwargs = dict(c=z, bsz=bsz, seq_len=seq_len)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            if mask.dim() > 1:
                mask = mask.flatten(start_dim=0, end_dim=1)
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z, temperature=1.0, cfg=1.0, mask_to_pred=None):

        # print(f"Before p Sample Loop - z: {z.shape}, in_channels: {self.in_channels}, cfg: {cfg}, temperature: {temperature}")

        bsz, seq_len, _ = z.shape
        z_to_pred = z[mask_to_pred.nonzero(as_tuple=True)]
        # z_full = z.flatten(start_dim=0, end_dim=1)  # (bsz*seq_len, d)
        c = z_to_pred

        # print(f"Diffusion Loss Sampling - z: {z.shape}, z_to_pred: {z_to_pred.shape}, mask_to_pred: {(mask_to_pred == True).sum().item()}")

        # diffusion loss sampling
        if not cfg == 1.0:
            noise = torch.randn(c.shape[0] // 2, self.in_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=c, bsz=bsz, seq_len=seq_len, cfg_scale=cfg, mask_to_pred=mask_to_pred)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(c.shape[0], self.in_channels).cuda()
            model_kwargs = dict(c=c, bsz=bsz, seq_len=seq_len, mask_to_pred=mask_to_pred)
            sample_fn = self.net.forward

        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            temperature=temperature
        )

        # sampled_token_latent = sampled_token_latent.reshape(bsz, seq_len, -1)
        # sampled_token_latent = sampled_token_latent[mask_to_pred.nonzero(as_tuple=True)]

        return sampled_token_latent


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    

class HopfieldMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        print(f"Hopfield Attention - in_channels: {in_channels}, model_channels: {model_channels}, out_channels: {out_channels}, z_channels: {z_channels}, num_res_blocks: {num_res_blocks}")

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))
        self.res_blocks = nn.ModuleList(res_blocks)

        # self.query_layer = FinalLayer(model_channels, model_channels)
        
        # self.mem_layer = FinalLayer(model_channels, out_channels)

        self.final_layer = FinalLayer(model_channels, out_channels)
        
        self.hopfield_layer = HopfieldAttention(dim_emb=model_channels, num_heads=1)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # nn.init.constant_(self.query_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.query_layer.adaLN_modulation[-1].bias, 0)
        # nn.init.constant_(self.query_layer.linear.weight, 0)
        # nn.init.constant_(self.query_layer.linear.bias, 0)

        # nn.init.constant_(self.mem_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.mem_layer.adaLN_modulation[-1].bias, 0)
        # nn.init.constant_(self.mem_layer.linear.weight, 0)
        # nn.init.constant_(self.mem_layer.linear.bias, 0)

    def forward(self, x, t, c, bsz, seq_len, mask_to_pred=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """

        # print(f"Diffusion Model (Before Embedding) - x: {x.shape}, t: {t.shape}, c: {c.shape}")

        # if t.dim() > 1:
        #     bsz, seq_len = t.shape
        #     t = t.flatten(start_dim=0, end_dim=1)

        t = self.time_embed(t)
        h = self.input_proj(x)
        c = self.cond_embed(c)

        # print(f"Diffusion Model (Before Reshape) - h: {h.shape}, t: {t.shape}, c: {c.shape}, bsz: {bsz}, seq_len: {seq_len}")

        t = t.reshape(bsz, -1, self.model_channels)
        h = h.reshape(bsz, -1, self.model_channels)
        c = c.reshape(bsz, -1, self.model_channels)

        # print(f"Diffusion Model (After Reshape) - h: {h.shape}, t: {t.shape}, c: {c.shape}, bsz: {bsz}, seq_len: {seq_len}")

        # assert t.shape == h.shape == c.shape == (bsz, seq_len, self.model_channels)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                h = checkpoint(block, h, y) # y, t
        else:
            for block in self.res_blocks:
                h = block(h, y) # y, t

        # print(f"Diffusion Model (Before Hopfield Layer) - h: {h.shape}, c: {c.shape}")

        h = self.hopfield_layer(h, c) # c

        # print(f"Diffusion Model (After Hopfield Layer) - gradient: {h.shape}")

        h = self.final_layer(h, y) # y, t

        # h = self.query_layer(h, t)

        # c = self.mem_layer(c, y)

        # if mask_to_pred is not None:
        #     output = h[mask_to_pred.nonzero(as_tuple=True)]
        # else:
        output = h.flatten(start_dim=0, end_dim=1)

        return output
    
    def forward_with_cfg(self, x, t, c, bsz, seq_len, cfg_scale, mask_to_pred=None):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c, bsz, seq_len, mask_to_pred)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class HopfieldAttention(nn.Module):
    r"""Z = softmax(β Q K^T) · V, with multi-head Hopfield energy computation"""
    def __init__(self,
                 dim_emb:    int,
                 dim_query:  Optional[int] = None,
                 dim_mem: Optional[int] = None,
                 num_heads:  int = 8,
                 qkv_bias:   bool = False,
                 qk_scale:   Optional[float] = None,
                 attn_drop:  float = 0.,
                 proj_drop:  float = 0.,
                 hetero:     bool = False,
                 **kwargs):
        super().__init__()

        dim_query = dim_query or dim_emb
        dim_mem = dim_mem or dim_emb
        self.num_heads = num_heads
        self.head_dim  = dim_emb // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.hetero = hetero

        # print(f"Hopfield Attention - dim_query: {dim_query}, dim_mem: {dim_mem}, dim_emb: {dim_emb}")

        # projections for Q, K and (K→V)
        self.W_Q = nn.Linear(dim_query, dim_emb, bias=qkv_bias) if dim_query != dim_emb else nn.Identity()
        self.W_K = nn.Linear(dim_mem, dim_emb, bias=qkv_bias) if dim_mem != dim_emb else nn.Identity()
        self.W_V = nn.Linear(dim_emb, dim_emb, bias=qkv_bias) if hetero else nn.Identity()

        # self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim_emb, dim_emb)
        # self.proj = nn.Identity()
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, memory=None, mode='attn'): # query ≡ R, memory ≡ Y

        # print(f"Hopfield Attention - query: {query.shape}, memory: {memory.shape}")

        # Full-dim projections
        Q = self.W_Q(query)  # (B,Lq,D)
        K = self.W_K(query if memory is None else memory)  # (B,Lk,D)
        # Split into multi-heads
        Qh = einops.rearrange(Q, 'B L (H D) -> B H L D', H=self.num_heads)  # (B,H,Lq,Dh)
        Kh = einops.rearrange(K, 'B L (H D) -> B H L D', H=self.num_heads)  # (B,H,Lk,Dh)

        # print(f"Hopfield Attention - Qh: {Qh.shape}, Kh: {Kh.shape}")

        # Compute multi-head attention
        logits = (Qh @ Kh.transpose(-2, -1)) * self.scale  # (B,H,Lq,Lk)
        attn = F.softmax(logits, dim=-1)
        # attn = self.attn_drop(attn)

        if mode == 'energy':
            # Compute Hopfield energy: log-sum-exp term + quadratic term E_h = -1/β * sum_i logsumexp_j (β * Qh · Kh) + 1/2 * ||Qh||^2
            lse = - torch.logsumexp(logits, dim=-1)  # (B,H,Lq)
            reg = 0.5 * (Qh ** 2).sum(dim=-1) 
            # reg += 0.5 * (Kh ** 2).sum(dim=-1)  # (B,H,Lq)
            Eh = (self.scale**-1) * lse + reg
            Eh = einops.rearrange(Eh, 'B H L -> B L H')  # (B,Lq,H)
            energy = Eh.sum(dim=-1, keepdim=True)  # (B,Lq,1)

            return energy
        
        elif mode == 'attn':
            if self.hetero:
                # Full-dim projections
                V_Q = self.W_V(Q)  # (B,Lk,D)
                V_K = self.W_V(K)  # (B,Lk,D)
                # Split into multi-heads
                V_Qh = einops.rearrange(V_Q, 'B L (H D) -> B H L D', H=self.num_heads)  # (B,H,Lq,Dh)
                V_Kh = einops.rearrange(V_K, 'B L (H D) -> B H L D', H=self.num_heads)  # (B,H,Lk,Dh)
            else:
                V_Qh = Qh; V_Kh = Kh
            # Compute Hopfield gradient: ∇_Q E_h = - softmax(β Q K^T) · V_K + V_Q
            grad_lse = -1.0 * (attn @ V_Kh)  # (B,H,Lq,Dh)
            grad_reg = 1.0 * V_Qh  # (B,H,Lq,Dh)
            # print(f"Hopfield Attention - grad_lse: {grad_lse.norm()}, grad_reg: {grad_reg.norm()}")
            grad_Qh = grad_lse + grad_reg  # (B,H,Lq,Dh)
            grad_Q = einops.rearrange(grad_Qh, 'B H L D -> B L (H D)')  # (B,Lq,D)
            # Project and drop
            # output = self.proj(grad_Q)
            # output = self.proj_drop(output)

            return grad_Q
        
        else:
            raise NotImplementedError(mode)


# class EBTBlock(L.LightningModule):
#     """
#     A EBT block with adaptive layer norm zero (adaLN-Zero) conditioning.
#     """
#     def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
#         self.attn = HopfieldAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
#         self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         mlp_hidden_dim = int(hidden_size * mlp_ratio)
#         approx_gelu = lambda: nn.GELU(approximate="tanh")
#         # approx_gelu = lambda: nn.Identity()
#         self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 6 * hidden_size, bias=True)
#         )

#     def forward(self, h, t, c):
#         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
#         h = self.norm1(h)
#         h = modulate(h, shift_msa, scale_msa)
#         with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False): #NOTE may want to turn this off for inference eventually
#             attn_output = self.attn(h, c)
#         h = h + gate_msa.unsqueeze(1) * attn_output
#         h = self.norm2(h)
#         h = modulate(h, shift_mlp, scale_mlp)
#         h = h + gate_mlp.unsqueeze(1) * self.mlp(h)
#         return h