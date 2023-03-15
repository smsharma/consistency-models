import dataclasses

from typing import Optional, Tuple

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp


class MlpBlock(nn.Module):
    mlp_dim: int

    @nn.compact
    def __call__(self, x):
        y = nn.Dense(self.mlp_dim)(x)
        y = nn.gelu(y)
        return nn.Dense(x.shape[-1])(y)


class MixerBlock(nn.Module):
    """Mixer block layer."""

    tokens_mlp_dim: int
    channels_mlp_dim: int

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm()(x)
        y = jnp.swapaxes(y, 1, 2)
        y = MlpBlock(self.tokens_mlp_dim)(y)
        y = jnp.swapaxes(y, 1, 2)
        x = x + y
        y = nn.LayerNorm()(x)
        y = MlpBlock(self.channels_mlp_dim)(y)
        return x + y


class MLPMixer(nn.Module):
    """Mixer architecture."""

    patch_size: int
    num_blocks: int
    hidden_dim: int
    tokens_mlp_dim: int
    channels_mlp_dim: int

    @nn.compact
    def __call__(self, x, context):
        b, h, w, c = x.shape

        # Repeat time context across spatial dimensions
        t = einops.repeat(context, "b t -> b (h p1) (w p2) t", h=h // self.patch_size, w=w // self.patch_size, p1=self.patch_size, p2=self.patch_size)

        # Concatenate time context to each patch
        x = jnp.concatenate([x, t], axis=-1)

        x = nn.Conv(self.hidden_dim, [self.patch_size, self.patch_size], strides=[self.patch_size, self.patch_size])(x)
        x = einops.rearrange(x, "n h w c -> n (h w) c")
        for i in range(self.num_blocks):
            x = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.patch_size * self.patch_size * c)(x)
        x = einops.rearrange(x, "B (Hp Wp) (pH pW C) -> B (Hp pH) (Wp pW) C", Hp=h // self.patch_size, Wp=w // self.patch_size, pH=self.patch_size, pW=self.patch_size, C=c)
        return x
