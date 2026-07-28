import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from src.physics_guidance import make_cond_func
from jax.scipy import stats


def sinusoidal_time_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half) / half)
    args = t[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)

def identity_combine_init(ch):
    """Init for the cond_combine 1x1 conv (kernel-shape (1, 1, 2*ch, ch)):
    idendity on the conv_in half, zero on the cond_combine half -> 
    combine([h, cond]) == h at initialisation. """
    def init(key, shape, dtype=jnp.float32):
        k = jnp.zeros(shape, dtype)
        return k.at[0,0,:ch, :].set(jnp.eye(ch, dtype=dtype))
    return init

def calc_log_probs(mean, std, xt):
    return jnp.sum(stats.norm.logpdf(xt, loc=mean, scale=std**2))


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


class ConditionalUnet(nn.Module):
    ch: int = 64
    ch_mult: tuple = (1, 1, 1, 2)
    out_ch: int = 3
    in_ch: int = 3
    n_resnet_blocks: int = 1
    dropout_p: float = 0.0
    freq_dim: int = 128

    @nn.compact
    def __call__(self, x, t, train=True, condRes = None):
        ch = self.ch
        temp_ch = ch * 4
        time_embed = sinusoidal_time_embedding(t, dim=self.freq_dim)
        time_embed = nn.Dense(temp_ch)(time_embed)
        time_embed = jax.nn.silu(time_embed)
        time_embed = nn.Dense(temp_ch)(time_embed)

        

        h = nn.Conv(ch, kernel_size=(3, 3), padding="CIRCULAR")(x)
        
        if condRes is not None:
            cond_h = nn.Conv(ch, kernel_size=(1,1), strides=1, padding=0, name="cond_in")(condRes)
            cond_h = jax.nn.gelu(cond_h)
            cond_h = nn.Conv(self.ch, kernel_size=(3,3), strides=1, padding="CIRCULAR", name="cond_hidden")(cond_h)

        else:
            cond_h = jnp.zeros_like(h)

        # Concatenating the conv input to the embedded residual information.
        h_concat = jnp.concat([h, cond_h], axis = -1)
        # Combining the input and residual embeddings to recover 3 channels.
        # Kernel init is needed to make sure that upon init the original h is 
        # return and the part of the kernel applied to the conditioning is init to 0.
        h = nn.Conv(ch, kernel_size=(1,1), strides=1, padding=0, name="cond_combine",
                                kernel_init=identity_combine_init(ch))(h_concat)

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
        model_kwargs = {"ch": model_cfg["ch"],
            "ch_mult": tuple(model_cfg["ch_mult"]),
            "out_ch": model_cfg["out_ch"],
            "in_ch": model_cfg["in_ch"],
            "n_resnet_blocks": model_cfg["n_resnet_blocks"],
            "dropout_p": model_cfg["dropout_p"],
            "freq_dim": model_cfg["freq_dim"],}

        if config['conditioning']['train']['enabled'] or config['conditioning']['inference']['enabled']:
            self.unet = ConditionalUnet(**model_kwargs)
        else:
            self.unet = Unet(**model_kwargs)
            print(f'Type of the model constructed with no cond {type(self.unet)}')

    @functools.partial(jax.jit, static_argnums=(0,))
    def forward_process(self, ims, t, noise_key):
        
        eps = jax.random.normal(noise_key, ims.shape)
        alpha = self.alpha_bar
        return (
            jnp.sqrt(alpha[t])[..., None, None, None] * ims
            + jnp.sqrt(1 - alpha[t])[..., None, None, None] * eps
        )


    def sample(self, params, dims, key, std, mean, x_g=None, t_start=None):
        T_max = self.config["diffusion"]["T"]
        if t_start is not None:
            assert t_start <= T_max, f"t_start={t_start} exceeds T={T_max}"
            T = t_start
        else:
            T = T_max

        alpha_bar = self.alpha_bar
        beta_schedule = self.beta_schedule
        model = self.unet

        # Defining arguments needed for conditioning
        if self.config['conditioning']['train']["enabled"]:
            n = dims[0] if x_g is None else x_g.shape[-2]
            re = self.config["conditioning"]['train']['re']
            cond_strength = self.config["conditioning"]['inference']['cond_strength']
            cond_signal = self.config['conditioning']['train'].get('cond_signal', 'gradient')
            cond_func = make_cond_func(cond_signal, n=n, re=re, std=std, mean=mean)
        
        key, init_key = jax.random.split(key)
        keys = jax.random.split(key, T + 1)

        if x_g is not None:
            batched = x_g.ndim == 4
            if not batched:
                x_g = x_g[None]  # (H, W, C) -> (1, H, W, C)
            eps = jax.random.normal(init_key, x_g.shape)
            xT = jnp.sqrt(alpha_bar[t_start])*x_g + jnp.sqrt(1- alpha_bar[t_start])*eps
        else:
            batched = False
            xT = jax.random.normal(keys[0], dims)
            xT = xT[None]  # (H, W, C) -> (1, H, W, C): model expects a batch dimension

        @jax.jit
        def jit_denoise_step(params, xT, t, z):

            if self.config['conditioning']['train']['enabled']:
                condRes = cond_func(xT)
                eps_pred_cond = model.apply({"params": params}, xT, jnp.array([t]), train=False, condRes=condRes)
                eps_pred_uncond = model.apply({"params": params}, xT, jnp.array([t]), train=False)
                eps_pred = (1 + cond_strength) * eps_pred_cond - cond_strength * eps_pred_uncond
            else:
                eps_pred = model.apply({"params": params}, xT, jnp.array([t]), train=False)
            
            alpha_bar_t = alpha_bar[t]
            alpha_t = 1 - beta_schedule[t]
            mean = (1 / jnp.sqrt(alpha_t)) * (
                xT - (1 - alpha_t) / jnp.sqrt(1 - alpha_bar_t) * eps_pred)
            std = jnp.sqrt(beta_schedule[t])

            xt_bwd = mean + std * z

            
            log_prob = calc_log_probs(mean, std, xt_bwd)


            return xt_bwd, log_prob

        for t in range(T, 0, -1):
            if t % 10 == 0 or t == T:
                print(f"    denoising t={t}/{T}", flush=True)
            z = jax.random.normal(keys[t], xT.shape) if t > 1 else jnp.zeros(xT.shape)
            xt_bwd = jit_denoise_step(params, xT, t, z)
            xT = xt_bwd
        if x_g is not None:
            return xT if batched else xT[0]  # strip batch dim for single-image callers
        return xt_bwd[0]  # unconditional: (1, H, W, C) -> (H, W, C)
