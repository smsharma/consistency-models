import jax
import jax.numpy as np
from flax.training import train_state
from typing import Any


def karras_boundaries(sigma, eps, N, T):
    """Boundaries for the time discretization."""
    idx = np.arange(N)
    return (eps ** (1 / sigma) + idx / (N - 1) * (T ** (1 / sigma) - eps ** (1 / sigma))) ** sigma


def get_timestep_embedding(timesteps, embedding_dim: int, dtype=np.float32):
    """Sinusoidal embeddings (from Fairseq)."""

    assert len(timesteps.shape) == 1
    timesteps *= 1000

    half_dim = embedding_dim // 2
    emb = np.log(10_000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim, dtype=dtype) * -emb)
    emb = timesteps.astype(dtype)[:, None] * emb[None, :]
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # Zero pad
        emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def apply_ema_decay(state, ema_decay):
    """Apply exponential moving average (EMA) decay to the model parameters and return the updated state."""
    params_ema = jax.tree_map(lambda p_ema, p: p_ema * ema_decay + p * (1.0 - ema_decay), state.params_ema, state.params)
    state = state.replace(params_ema=params_ema)
    return state


class TrainState(train_state.TrainState):
    """TrainState container class with an extra attribute for the EMA parameters."""

    params_ema: Any = None
