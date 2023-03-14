import jax
import jax.numpy as jnp
from flax import linen as nn
from einops import repeat


class ChannelMixingBlock(nn.Module):
    """A channel mixing block."""

    channels: int
    tokens_dim: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x):
        # Layer normalization
        x = nn.LayerNorm()(x)

        # Token mixing
        y = nn.Dense(features=self.mlp_dim)(x)
        y = nn.gelu(y)
        y = nn.Dense(features=self.tokens_dim)(y)
        y = nn.softmax(y, axis=1)
        z = jnp.einsum("bij,bkj->bik", y, x)

        # Channel mixing
        z = nn.Conv(features=self.channels, kernel_size=(1, 1))(z)

        # Residual connection
        x = x + z

        return x


class MLPMixer(nn.Module):
    """An img2img MLP-Mixer."""

    patch_size: int
    channels: int
    tokens_dim: int
    mlp_dim: int
    num_blocks: int

    @nn.compact
    def __call__(self, x, t):

        b, h, w, c = x.shape

        # Repeat time context across spatial dimensions
        t = repeat(t, "b t -> b (h p1) (w p2) t", h=x.shape[1] // self.patch_size, w=x.shape[2] // self.patch_size, p1=self.patch_size, p2=self.patch_size)

        # Concatenate time context to each patch
        x = jnp.concatenate([x, t], axis=-1)

        # Split the image into patches
        x = nn.Conv(features=self.channels, kernel_size=(self.patch_size, self.patch_size), strides=(self.patch_size, self.patch_size))(x)

        # Reshape patches to 1D tokens
        x = jnp.reshape(x, (x.shape[0], -1, x.shape[-1]))

        # Apply channel mixing blocks
        for _ in range(self.num_blocks):
            x = ChannelMixingBlock(self.channels, self.tokens_dim, self.mlp_dim)(x)

        # Combine tokens into image
        output_size = h
        x = jnp.reshape(x, (x.shape[0], output_size // self.patch_size, output_size // self.patch_size, self.channels))
        x = nn.ConvTranspose(features=self.channels, kernel_size=(self.patch_size, self.patch_size), strides=(self.patch_size, self.patch_size))(x)

        return x
