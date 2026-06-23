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


class Diffusion:
    def __init__(self, config: dict):
        
        self.config = config
        
        self.diffusion = config["diffusion"]
        
        # define sampling method
        self.method = self.diffusion["method"]

        # define noise schedule
        self.beta_schedule = jnp.linspace(
            self.diffusion["beta_start"], self.diffusion["beta_end"], self.diffusion["T"]
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
        a = alpha[t]                                          # scalar () or batched (B,)
        a = a.reshape(a.shape + (1,) * (ims.ndim - a.ndim))   # -> (1,1,1,1) or (B,1,1,1)
        return jnp.sqrt(a) * ims + jnp.sqrt(1 - a) * eps
    
    def init_sampling(self, key, dims, x_g=None, t_start=None):
        
        T_max = self.config["diffusion"]["T"]
        
        if t_start is not None:
            assert t_start <= T_max, f"t_start={t_start} exceeds T={T_max}"
            T = t_start
        else:
            T = T_max
        
        key, init_key = jax.random.split(key)
        keys = jax.random.split(key, T + 1)


        if x_g is not None:
            batched = x_g.ndim == 4
            if not batched:
                x_g = x_g[None]  # (H, W, C) -> (1, H, W, C)
            eps = jax.random.normal(init_key, x_g.shape)
            #xT = jnp.sqrt(alpha_bar[t_start])*x_g + jnp.sqrt(1- alpha_bar[t_start])*eps
            xT = self.forward_process(x_g, t_start, eps)
        else:
            batched = False
            xT = jax.random.normal(keys[0], dims)
            xT = xT[None]  # (H, W, C) -> (1, H, W, C): model expects a batch dimension

        return xT, T, keys, batched



class DDPM(Diffusion):
    def __init__(self, config:dict):
        super().__init__(config=config)        
    
    def sample(self, params, dims, key, x_g=None, t_start=None):
        
        model = self.unet
        alpha_bar = self.alpha_bar
        beta_schedule = self.beta_schedule

        xT, T, keys, batched = self.init_sampling(key, dims, x_g, t_start)

        @jax.jit
        def ddpm_denoise_step(params, xT, t, z):
            eps_pred = model.apply({"params": params}, xT, jnp.array([t]), train=False)
            alpha_bar_t = alpha_bar[t]
            alpha_t = 1 - beta_schedule[t]
            xt_bwd = (1 / jnp.sqrt(alpha_t)) * (
                xT - (1 - alpha_t) / jnp.sqrt(1 - alpha_bar_t) * eps_pred
            ) + jnp.sqrt(beta_schedule[t]) * z

            return xt_bwd

        for t in range(T, 0, -1):
            if t % 10 == 0 or t == T:
                print(f"    denoising t={t}/{T}", flush=True)
            z = jax.random.normal(keys[t], xT.shape) if t > 1 else jnp.zeros(xT.shape)
            xt_bwd = ddpm_denoise_step(params, xT, t, z)
            
            xT = xt_bwd
        
        if x_g is not None:
            return xT if batched else xT[0]  # strip batch dim for single-image callers
        
        return xt_bwd[0]  # unconditional: (1, H, W, C) -> (H, W, C)


class DDIM(Diffusion):

    def __init__(self, config:dict):
        super().__init__(config=config)
    
    def sample(self, params, dims, key, x_g=None, t_start=None):
        
        model = self.unet
        alpha_bar = self.alpha_bar
        eta = self.diffusion["eta"]
        stride = self.diffusion["stride"]

        xT, T, keys, batched = self.init_sampling(key, dims, x_g, t_start)

        def ddim_x1(x0, post_mean, t, sigma, z):
            return x0 + sigma*z

        def ddim_xt(x0, post_mean, t, sigma, z):
            return post_mean + sigma*z
        
        @jax.jit
        def ddim_denoise_step(condition, params, xT, t, t_prev, key):
            eps_pred = model.apply({"params": params}, xT, jnp.array([t]), train=False)
            
            alpha_bar_t = alpha_bar[t]
            alpha_bar_t_prev = alpha_bar[t_prev]

            sigma_t = eta * jnp.sqrt((1-alpha_bar_t_prev) / (1-alpha_bar_t) ) * jnp.sqrt(1-alpha_bar_t / alpha_bar_t_prev)

            x0_pred = (xT - jnp.sqrt(1-alpha_bar_t) * eps_pred) / jnp.sqrt(alpha_bar_t)
            z = jax.random.normal(key, shape = xT.shape)
            post_mean = jnp.sqrt(alpha_bar_t_prev)*x0_pred + jnp.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * (xT - jnp.sqrt(alpha_bar_t)*x0_pred)/jnp.sqrt(1-alpha_bar_t)
            xt_bwd = jax.lax.cond(condition, ddim_x1, ddim_xt, x0_pred, post_mean, t, sigma_t, z)
            
            return xt_bwd


        for t in range(T, 0, -stride):
            
            t_prev = max(t - stride, 0)
            xt_bwd = ddim_denoise_step(t_prev < stride, params, xT, t, t_prev, keys[t])
            xT = xt_bwd

        if x_g is not None:
            return xT if batched else xT[0]  # strip batch dim for single-image callers

        return xt_bwd[0]  # unconditional: (1, H, W, C) -> (H, W, C)


_DIFFUSION_REGISTRY = {"ddpm": DDPM, "ddim": DDIM}


def build_diffusion(config: dict):
    """Instantiate the sampler selected by config['diffusion']['method'] ('ddpm'|'ddim').

    DDPM and DDIM share the same sample() signature, so callers only swap the
    constructor — nothing else downstream changes.
    """
    method = config["diffusion"]["method"].lower()
    if method not in _DIFFUSION_REGISTRY:
        raise ValueError(
            f"Unknown diffusion method '{method}'; expected one of {list(_DIFFUSION_REGISTRY)}"
        )
    return _DIFFUSION_REGISTRY[method](config)
