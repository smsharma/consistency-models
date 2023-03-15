import einops
import flax.linen as nn
import jax.numpy as np


class MLPBlock(nn.Module):
    """MLP block layer."""

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
        y = np.swapaxes(y, 1, 2)
        y = MLPBlock(self.tokens_mlp_dim)(y)
        y = np.swapaxes(y, 1, 2)
        x = x + y
        y = nn.LayerNorm()(x)
        y = MLPBlock(self.channels_mlp_dim)(y)
        return x + y


class MLPMixer(nn.Module):
    """MLP-Mixer architecture from https://arxiv.org/abs/2105.01601."""

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

        # Concatenate time context to image (alt.: concat to patch?)
        x = np.concatenate([x, t], axis=-1)

        # Create patches
        x = nn.Conv(self.hidden_dim, [self.patch_size, self.patch_size], strides=[self.patch_size, self.patch_size])(x)
        x = einops.rearrange(x, "n h w c -> n (h w) c")

        # Apply mixer blocks
        for _ in range(self.num_blocks):
            x = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)(x)
        x = nn.LayerNorm()(x)

        # Project back to image
        x = nn.Dense(self.patch_size * self.patch_size * c)(x)
        x = einops.rearrange(x, "b (hp wp) (ph pw c) -> b (hp ph) (wp pw) c", hp=h // self.patch_size, wp=w // self.patch_size, ph=self.patch_size, pw=self.patch_size, c=c)

        return x
