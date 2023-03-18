import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from einops import repeat


class UNet(nn.Module):
    hidden_channels: int
    num_layers: int
    num_classes: int

    @nn.compact
    def __call__(self, x, t, context):
        class EncoderBlock(nn.Module):
            channels: int

            @nn.compact
            def __call__(self, x, conditioning):
                x = jnp.concatenate([x, conditioning], axis=-1)
                x = nn.Conv(features=self.channels, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
                x = nn.gelu(x)
                x = nn.LayerNorm()(x)
                return x

        class DecoderBlock(nn.Module):
            channels: int

            @nn.compact
            def __call__(self, x, conditioning):
                x = jnp.concatenate([x, conditioning], axis=-1)
                x = nn.ConvTranspose(features=self.channels, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
                x = nn.gelu(x)
                x = nn.LayerNorm()(x)
                return x

        b, h, w, c = x.shape

        # Embed context to same dim as time embedding and repeat across spatial dimensions
        d_t_emb = t.shape[-1]
        context = nn.Embed(self.num_classes, d_t_emb)(context)
        conditioning = jnp.concatenate([context, t], axis=-1)

        # Small element-wise MLP to process context
        conditioning = nn.gelu(nn.Dense(4 * conditioning.shape[-1])(conditioning))
        conditioning = nn.Dense(d_t_emb)(conditioning)

        # Encoder
        encodings = []  # Save encodings for skip connections
        hidden_channels = self.hidden_channels
        for i in range(self.num_layers):

            curr_conditioning = nn.Dense(features=hidden_channels)(conditioning)
            curr_conditioning = nn.LayerNorm()(curr_conditioning)
            curr_conditioning = repeat(curr_conditioning, "b f -> b h w f", b=b, h=h, w=w)

            enc = EncoderBlock(hidden_channels)(x, curr_conditioning)
            encodings.append(enc)
            hidden_channels *= 2  # Dilate
            x = enc
            h, w = h // 2, w // 2

        # Bottleneck
        bottleneck = nn.Conv(features=hidden_channels, kernel_size=(3, 3), padding="SAME")(x)
        bottleneck = nn.gelu(bottleneck)
        bottleneck = nn.LayerNorm()(bottleneck)

        curr_conditioning = nn.Dense(features=hidden_channels)(conditioning)
        curr_conditioning = nn.LayerNorm()(curr_conditioning)
        curr_conditioning = repeat(curr_conditioning, "b f -> b h w f", b=b, h=h, w=w)

        bottleneck = jnp.concatenate([bottleneck, curr_conditioning], axis=-1)

        # Decoder
        for i in range(self.num_layers - 1, -1, -1):
            hidden_channels //= 2  # Contract

            curr_conditioning = nn.Dense(features=hidden_channels)(conditioning)
            curr_conditioning = nn.LayerNorm()(curr_conditioning)
            curr_conditioning = repeat(curr_conditioning, "b f -> b h w f", b=b, h=h, w=w)

            dec = DecoderBlock(hidden_channels)(bottleneck, curr_conditioning)
            if i != 0:
                dec = jnp.concatenate([dec, encodings[i - 1]], axis=-1)
            bottleneck = dec
            h, w = h * 2, w * 2

        # Output
        output = nn.ConvTranspose(features=c, kernel_size=(1, 1), strides=(1, 1), padding="SAME")(dec)

        return output
