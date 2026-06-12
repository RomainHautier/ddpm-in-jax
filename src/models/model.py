import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


def sinusoidal_time_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half) / half)
    args = t[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


class DDPMResnet(nn.Module):
    filters: int = 160
    dropout_p: float = 0.5

    @nn.compact
    def __call__(self, x, time_embed, train=True):
        residual = x
        h = nn.GroupNorm(8)(x)
        h = jax.nn.silu(h)
        h = nn.Conv(features=self.filters, kernel_size=(3, 3), padding="CIRCULAR")(h)
        t = nn.Dense(self.filters)(jax.nn.silu(time_embed))
        h = h + t[:, None, None, :]
        h = nn.GroupNorm(8)(h)
        h = jax.nn.silu(h)
        h_dropout = nn.Dropout(rate=self.dropout_p)(h, deterministic=not train)
        h = nn.Conv(features=self.filters, kernel_size=(3, 3), padding="CIRCULAR")(h_dropout)
        if x.shape[-1] != self.filters:
            residual = nn.Conv(self.filters, (1, 1), padding="CIRCULAR")(residual)
        return h + residual


class SelfAttention(nn.Module):
    num_groups: int = 32

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        h = nn.GroupNorm(self.num_groups)(x)
        h_flat = h.reshape(B, H * W, C)
        q = nn.Dense(C)(h_flat)
        k = nn.Dense(C)(h_flat)
        v = nn.Dense(C)(h_flat)
        scaling = jnp.sqrt(C)
        attn = jax.nn.softmax(q @ k.transpose(0, 2, 1) / scaling, axis=-1)
        out = attn @ v
        out = nn.Dense(C)(out)
        out = out.reshape(B, H, W, C)
        return out + x


class Unet(nn.Module):
    ch: int = 64
    ch_mult: tuple = (1, 1, 1, 2)
    out_ch: int = 3
    in_ch: int = 3
    n_resnet_blocks: int = 1
    dropout_p: float = 0.0
    freq_dim: int = 128

    @nn.compact
    def __call__(self, x, t, train=True):
        ch = self.ch
        temp_ch = ch * 4
        time_embed = sinusoidal_time_embedding(t, dim=self.freq_dim)
        time_embed = nn.Dense(temp_ch)(time_embed)
        time_embed = jax.nn.silu(time_embed)
        time_embed = nn.Dense(temp_ch)(time_embed)
        h = nn.Conv(ch, kernel_size=(3, 3), padding="CIRCULAR")(x)
        hs = [h]
        for _ in range(self.n_resnet_blocks):
            h = DDPMResnet(ch * self.ch_mult[0], self.dropout_p)(
                h, time_embed=time_embed, train=train
            )
            hs.append(h)
        h = nn.Conv(
            ch * self.ch_mult[0], kernel_size=(3, 3), strides=(2, 2), padding=((0, 1), (0, 1))
        )(h)
        hs.append(h)
        for _ in range(self.n_resnet_blocks):
            h = DDPMResnet(ch * self.ch_mult[1], self.dropout_p)(
                h, time_embed=time_embed, train=train
            )
            hs.append(h)
        h = nn.Conv(
            ch * self.ch_mult[1], kernel_size=(3, 3), strides=(2, 2), padding=((0, 1), (0, 1))
        )(h)
        hs.append(h)
        for _ in range(self.n_resnet_blocks):
            h = DDPMResnet(ch * self.ch_mult[2], self.dropout_p)(
                h, time_embed=time_embed, train=train
            )
            hs.append(h)
        h = nn.Conv(
            ch * self.ch_mult[2], kernel_size=(3, 3), strides=(2, 2), padding=((0, 1), (0, 1))
        )(h)
        hs.append(h)
        for _ in range(self.n_resnet_blocks):
            h = DDPMResnet(ch * self.ch_mult[3], self.dropout_p)(
                h, time_embed=time_embed, train=train
            )
            hs.append(h)
        h = DDPMResnet(ch * self.ch_mult[-1], self.dropout_p)(h, time_embed=time_embed, train=train)
        h = SelfAttention(num_groups=8)(h)
        h = DDPMResnet(ch * self.ch_mult[-1], self.dropout_p)(h, time_embed=time_embed, train=train)
        for _ in range(self.n_resnet_blocks + 1):
            h = DDPMResnet(ch * self.ch_mult[3], self.dropout_p)(
                jnp.concatenate([h, hs.pop()], axis=-1), time_embed=time_embed, train=train
            )
        B, H, W, C = h.shape
        h = jax.image.resize(h, (B, H * 2, W * 2, C), method="nearest")
        h = nn.Conv(ch * self.ch_mult[3], kernel_size=(3, 3), padding="CIRCULAR")(h)
        for _ in range(self.n_resnet_blocks + 1):
            h = DDPMResnet(ch * self.ch_mult[2], self.dropout_p)(
                jnp.concatenate([h, hs.pop()], axis=-1), time_embed=time_embed, train=train
            )
        B, H, W, C = h.shape
        h = jax.image.resize(h, (B, H * 2, W * 2, C), method="nearest")
        h = nn.Conv(ch * self.ch_mult[2], kernel_size=(3, 3), padding="CIRCULAR")(h)
        for _ in range(self.n_resnet_blocks + 1):
            h = DDPMResnet(ch * self.ch_mult[1], self.dropout_p)(
                jnp.concatenate([h, hs.pop()], axis=-1), time_embed=time_embed, train=train
            )
        B, H, W, C = h.shape
        h = jax.image.resize(h, (B, H * 2, W * 2, C), method="nearest")
        h = nn.Conv(ch * self.ch_mult[1], kernel_size=(3, 3), padding="CIRCULAR")(h)
        for _ in range(self.n_resnet_blocks + 1):
            h = DDPMResnet(ch * self.ch_mult[0], self.dropout_p)(
                jnp.concatenate([h, hs.pop()], axis=-1), time_embed=time_embed, train=train
            )
        h = nn.GroupNorm(8)(h)
        h = jax.nn.silu(h)
        return nn.Conv(self.out_ch, kernel_size=(3, 3), padding="CIRCULAR")(h)


class DDPM:
    def __init__(self, config: dict):
        self.config = config

        diffusion = config["diffusion"]
        self.beta_schedule = jnp.linspace(
            diffusion["beta_start"], diffusion["beta_end"], diffusion["T"]
        )
        self.alpha_bar = jnp.cumprod(1 - self.beta_schedule)

        model_cfg = config["model"]
        self.unet = Unet(
            ch=model_cfg["ch"],
            ch_mult=tuple(model_cfg["ch_mult"]),
            out_ch=model_cfg["out_ch"],
            in_ch=model_cfg["in_ch"],
            n_resnet_blocks=model_cfg["n_resnet_blocks"],
            dropout_p=model_cfg["dropout_p"],
            freq_dim=model_cfg["freq_dim"],
        )

    def forward_process(self, ims, t, eps):
        alpha = self.alpha_bar
        return (
            jnp.sqrt(alpha[t])[:, None, None, None] * ims
            + jnp.sqrt(1 - alpha[t])[:, None, None, None] * eps
        )

    def sample(self, params, dims, key, x_g=None, t_start=None):
        T_max = self.config["diffusion"]["T"]
        if t_start is not None:
            assert t_start <= T_max, f"t_start={t_start} exceeds T={T_max}"
            T = t_start
        else:
            T = T_max

        alpha_bar = self.alpha_bar
        beta_schedule = self.beta_schedule
        model = self.unet
        
        key, init_key = jax.random.split(key)
        keys = jax.random.split(key, T + 1)

        if x_g is not None:
            eps = jax.random.normal(init_key, dims)
            xT = jnp.sqrt(alpha_bar[t_start])*x_g + jnp.sqrt(1- alpha_bar[t_start])*eps
        else:
            xT = jax.random.normal(keys[0], dims)

        xT = xT[None]  # (H, W, C) -> (1, H, W, C): model expects a batch dimension

        for t in range(T, 0, -1):
            if t % 10 == 0 or t == T:
                print(f"    denoising t={t}/{T}", flush=True)
            z = jax.random.normal(keys[t], xT.shape) if t > 1 else 0
            eps_pred = model.apply({"params": params}, xT, jnp.array([t]), train=False)
            alpha_bar_t = alpha_bar[t]
            alpha_t = 1 - beta_schedule[t]
            xt_bwd = (1 / jnp.sqrt(alpha_t)) * (
                xT - (1 - alpha_t) / jnp.sqrt(1 - alpha_bar_t) * eps_pred
            ) + jnp.sqrt(beta_schedule[t]) * z
            xT = xt_bwd
        return xt_bwd[0]  # (1, H, W, C) -> (H, W, C)
