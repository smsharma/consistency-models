from absl import logging
from functools import partial
import jax
import jax.numpy as np
import math


def f_theta(params, score, x, t, y, sigma_data, eps, d_t_embed):
    """The consistency model."""

    c_skip = sigma_data**2 / ((t - eps) ** 2 + sigma_data**2)
    c_out = sigma_data * (t - eps) / np.sqrt(sigma_data**2 + t**2)

    t = t[..., 0]
    t = timestep_embedding(t, d_t_embed)
    x_out = score.apply(params, x, t, y)

    return x * c_skip[:, :, None, None] + x_out * c_out[:, :, None, None]


def timestep_discretization(sigma, eps, N, T):
    """Boundaries for the time discretization (from Karras et al, 2022)."""
    idx = np.arange(N)
    return (eps ** (1 / sigma) + idx / (N - 1) * (T ** (1 / sigma) - eps ** (1 / sigma))) ** sigma


def timestep_embedding(timesteps, embedding_dim: int, dtype=np.float32):
    """Sinusoidal embeddings (from Fairseq)."""

    assert len(timesteps.shape) == 1
    timesteps *= 1000

    half_dim = embedding_dim // 2
    emb = math.log(10_000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim, dtype=dtype) * -emb)
    emb = timesteps.astype(dtype)[:, None] * emb[None, :]
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # Zero pad
        emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def sample(params, score, config, y, timesteps, key, shape=(16, 28, 28, 1)):
    """Draw samples from consistency model."""
    x0 = jax.random.normal(key, shape) * timesteps[0]
    x = f_theta(params, score, x0, np.repeat(timesteps[0], x0.shape[0])[:, None], y, config.sigma_data, config.eps, config.d_t_embed)
    for t in timesteps[1:]:
        key, _ = jax.random.split(key)
        z = jax.random.normal(key, shape=x0.shape)
        x = x + math.sqrt(t**2 - config.eps**2) * z
        x = f_theta(params, score, x, np.repeat(t, x0.shape[0])[:, None], y, config.sigma_data, config.eps, config.d_t_embed)
    return x


@partial(jax.jit, static_argnums=(5, 8, 9, 10))
def loss_fn_discrete(params, params_ema, x, t1, t2, score, key, y, sigma_data, eps, d_t_embed):
    """Discrete consistency loss function."""

    z = jax.random.normal(key, shape=x.shape)

    x2 = x + z * t2[:, :, None, None]
    x2 = f_theta(params, score, x2, t2, y, sigma_data, eps, d_t_embed)

    x1 = x + z * t1[:, :, None, None]
    x1 = f_theta(params_ema, score, x1, t1, y, sigma_data, eps, d_t_embed)

    return np.mean((x1 - x2) ** 2)


# @partial(jax.jit, static_argnums=(3,))
# def loss_fn_continuous(params, x, t, score, key):
#     """Continous consistency loss function."""

#     z = jax.random.normal(key, shape=x.shape)
#     xt = x + z * t[:, :, None, None]

#     params_min = jax.lax.stop_gradient(params)

#     f_theta_vmapped = jax.vmap(f_theta, in_axes=(None, None, 0, 0))(params, score, x, t)
#     d_f_theta_dx, d_f_theta_dt = jax.vmap(jax.jacfwd(f_theta, argnums=(2, 3)), in_axes=(None, None, 0, 0))(params_min, score, x, t)

#     loss2 = d_f_theta_dt[..., 0] - np.einsum("bhwcijk,bijk->bhwc", d_f_theta_dx, (xt - x) / t[:, None, None])

#     return 2 * np.mean(np.einsum("bijk,bijk->b", f_theta_vmapped, loss2))
