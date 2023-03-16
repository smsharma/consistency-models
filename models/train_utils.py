from absl import logging
import jax
import jax.numpy as np
from flax.training import train_state
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
from einops import rearrange
import wandb

from functools import partial
from typing import Any

from models.consistency_utils import sample


def apply_ema_decay(state, ema_decay):
    """Apply exponential moving average (EMA) decay to the model parameters and return the updated state."""
    params_ema = jax.tree_map(lambda p_ema, p: p_ema * ema_decay + p * (1.0 - ema_decay), state.params_ema, state.params)
    state = state.replace(params_ema=params_ema)
    return state


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(5, 6, 7, 8, 9, 10))
def train_step(state, batch, t1, t2, key, model, loss_fn, mu, sigma_data, eps, d_t_embed):
    """Single train step."""
    x_batch, y_batch = batch

    loss, grads = jax.value_and_grad(loss_fn)(state.params, state.params_ema, x_batch, t1, t2, model, key, y_batch, sigma_data, eps, d_t_embed)

    grads = jax.lax.pmean(grads, "batch")
    loss = jax.lax.pmean(loss, "batch")

    state = state.apply_gradients(grads=grads)
    state = apply_ema_decay(state, mu)

    metrics = {"loss": loss}

    return state, metrics


def to_wandb_config(d: ConfigDict, parent_key: str = "", sep: str = "."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, ConfigDict):
            items.extend(to_wandb_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def param_count(pytree):
    """Count the number of parameters in a pytree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(pytree))


class TrainState(train_state.TrainState):
    """TrainState container class with an extra attribute for the EMA parameters."""

    params_ema: Any = None


def log_eval_grid(state, score, key, config, shape):

    y = np.arange(16) % 10
    T = config.T

    # Generate samples with 5 and 2 steps
    x_samples_2 = sample(state.params, score, config, y, list(reversed([1.0, T])), key, shape)
    x_samples_5 = sample(state.params, score, config, y, list(reversed([1.0, 5.0, T / 4.0, T / 2.0, T])), key, shape)

    x_samples_2_grid = rearrange(x_samples_2, "(n1 n2) h w c -> (n1 h) (n2 w) c", n1=4)
    x_samples_5_grid = rearrange(x_samples_5, "(n1 n2) h w c -> (n1 h) (n2 w) c", n1=4)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow((x_samples_5_grid + 1) / 2.0)
    ax1.axis("off")
    ax1.set_title("5-step samples", fontsize=16)

    ax2.imshow((x_samples_2_grid + 1) / 2.0)
    ax2.axis("off")
    ax2.set_title("2-step samples", fontsize=16)

    plt.tight_layout()

    wandb.log({"eval/grid": fig})
