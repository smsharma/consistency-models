import flax.linen as nn
from einops import rearrange


class AttentionBlock(nn.Module):
    embed_dim: int  # Dimensionality of input and attention feature vectors
    hidden_dim: int  # Dimensionality of hidden layer in feed-forward network
    num_heads: int  # Number of heads to use in the Multi-Head Attention block
    dropout_prob: float = 0.0  # Amount of dropout to apply in the feed-forward network

    def setup(self):
        self.attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.linear = [nn.Dense(self.hidden_dim), nn.gelu, nn.Dense(self.embed_dim)]
        self.layer_norm_1 = nn.LayerNorm()
        self.layer_norm_2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x):
        inp_x = self.layer_norm_1(x)
        attn_out = self.attn(inputs_q=inp_x, inputs_kv=inp_x)
        x = x + attn_out

        linear_out = self.layer_norm_2(x)
        for l in self.linear:
            linear_out = l(linear_out)
        x = x + linear_out
        return x


class VisionTransformer(nn.Module):
    embed_dim: int = 256  # Dimensionality of input and attention feature vectors
    hidden_dim: int = 512  # Dimensionality of hidden layer in feed-forward network
    num_heads: int = 2  # Number of heads to use in the Multi-Head Attention block
    num_channels: int = 1  # Number of channels of the input (3 for RGB)
    num_layers: int = 4  # Number of layers to use in the Transformer
    patch_size: int = 4  # Number of pixels that the patches have per dimension
    num_patches: int = 128  # Maximum number of patches an image can have

    def setup(self):

        # Layers/Networks
        self.input_layer = nn.Dense(self.embed_dim)
        self.context_embedding = nn.Dense(self.embed_dim)
        self.transformer = [AttentionBlock(self.embed_dim, self.hidden_dim, self.num_heads) for _ in range(self.num_layers)]
        self.patchify = nn.Dense(self.patch_size * self.patch_size * self.num_channels)

        # Parameters/Embeddings
        self.pos_embedding = self.param("pos_embedding", nn.initializers.normal(stddev=1.0), (1, 1 + self.num_patches, self.embed_dim))

    def __call__(self, x, context=None):

        B, H, W, C = x.shape

        # Preprocess input
        x = rearrange(x, "B (Hp pH) (Wp pW) C -> B (Hp Wp) (pH pW C)", Hp=H // self.patch_size, Wp=W // self.patch_size, pH=self.patch_size, pW=self.patch_size, C=self.num_channels)

        _, num_tokens, _ = x.shape

        x = self.input_layer(x)

        # Add positional embedding
        x = x + self.pos_embedding[:, :num_tokens]

        # Add context embedding
        if context is not None:
            x = x + self.context_embedding(context)[:, None, :]

        # Apply transformer layers
        for attn_block in self.transformer:
            x = attn_block(x)

        x = self.patchify(x)
        x = rearrange(x, "B (Hp Wp) (pH pW C) -> B (Hp pH) (Wp pW) C", Hp=H // self.patch_size, Wp=W // self.patch_size, pH=self.patch_size, pW=self.patch_size, C=self.num_channels)
        return x
