import sys
import os
import yaml
import math

from absl import flags, logging
from absl import logging
import ml_collections
from ml_collections import config_flags
from clu import metric_writers
import wandb

sys.path.append("./")
sys.path.append("../")

from tqdm import trange

import jax
import jax.numpy as np
import optax
import flax
from flax.core import FrozenDict
from flax.training import checkpoints, common_utils, train_state

import tensorflow as tf

# Ensure TF does not see GPU and grab all GPU memory
tf.config.experimental.set_visible_devices([], "GPU")

from models.train_utils import to_wandb_config, TrainState, train_step, param_count, log_eval_grid
from models.consistency_utils import timestep_discretization, loss_fn_discrete
from dataset import Dataset

from models.mlp_mixer import MLPMixer
from models.unet import UNet

replicate = flax.jax_utils.replicate
unreplicate = flax.jax_utils.unreplicate

logging.set_verbosity(logging.INFO)


def train(config: ml_collections.ConfigDict, workdir: str = "./logging/") -> train_state.TrainState:

    # Set up wandb run
    if config.wandb.log_train and jax.process_index() == 0:
        wandb_config = to_wandb_config(config)
        run = wandb.init(entity=config.wandb.entity, project=config.wandb.project, job_type=config.wandb.job_type, group=config.wandb.group, config=wandb_config)
        wandb.define_metric("*", step_metric="train/step")
        workdir = os.path.join(workdir, run.group, run.name)

        # Recursively create workdir
        os.makedirs(workdir, exist_ok=True)

    # Dump a yaml config file in the output directory
    with open(os.path.join(workdir, "config.yaml"), "w") as f:
        yaml.dump(config.to_dict(), f)

    writer = metric_writers.create_default_writer(logdir=workdir, just_logging=jax.process_index() != 0)

    # Load the dataset
    dataset = Dataset(dataset_name=config.data.dataset, batch_size=config.training.batch_size)
    train_ds = dataset.create_dataset()
    batches = dataset.create_input_iter(train_ds)

    logging.info("Loaded the %s dataset", config.data.dataset)

    ## Model configuration and instantiation
    score_dict = FrozenDict(config.score)
    if config.score.score == "unet":
        score_dict.pop("score", None)
        score = UNet(num_classes=config.data.num_classes, **score_dict)
    elif config.score.score == "mlp_mixer":
        score_dict.pop("score", None)
        score = MLPMixer(num_classes=config.data.num_classes, **score_dict)
    else:
        raise NotImplementedError

    x_batch, y_batch = next(batches)
    rng = jax.random.PRNGKey(42)

    x = x_batch[0]
    t = np.ones((x_batch.shape[1], config.consistency.d_t_embed))
    context = y_batch[0]

    params = score.init(rng, x, t, context)

    logging.info("Instantiated the model")
    logging.info("Number of parameters: %d", param_count(params))

    ## Training config and loop

    tx = optax.adamw(learning_rate=config.optim.learning_rate, weight_decay=config.optim.weight_decay)
    state = TrainState.create(apply_fn=score.apply, params=params, tx=tx, params_ema=params)
    pstate = replicate(state)

    logging.info("Starting training...")

    train_metrics = []
    with trange(config.training.n_train_steps) as steps:
        for step in steps:

            rng, *train_step_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
            train_step_rng = np.asarray(train_step_rng)

            # Timestep discretization and EMA schedule from paper
            N = math.ceil(math.sqrt((step * ((config.consistency.s1 + 1) ** 2 - config.consistency.s0**2) / config.training.n_train_steps) + config.consistency.s0**2) - 1) + 1
            mu = math.exp(config.consistency.s0 * math.log(config.consistency.mu0) / N)
            boundaries = timestep_discretization(config.consistency.sigma, config.consistency.eps, N, config.consistency.T)

            # Draw timesteps from discretized schedule
            n_batch = jax.random.randint(rng, minval=0, maxval=N - 1, shape=(*x_batch.shape[:2], 1))

            pstate, metrics = train_step(pstate, next(batches), boundaries[n_batch], boundaries[n_batch + 1], train_step_rng, score, loss_fn_discrete, mu, config.consistency.sigma_data, config.consistency.eps, config.consistency.d_t_embed)

            steps.set_postfix(val=unreplicate(metrics["loss"]))
            train_metrics.append(metrics)

            # Log periodically
            if (step % config.training.log_every_steps == 0) and (step != 0) and (jax.process_index() == 0):

                train_metrics = common_utils.get_metrics(train_metrics)
                summary = {f"train/{k}": v for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()}

                writer.write_scalars(step, summary)
                train_metrics = []

                if config.wandb.log_train:
                    wandb.log({"train/step": step, **summary})

            # Eval periodically
            if (step % config.training.eval_every_steps == 0) and (step != 0) and (jax.process_index() == 0) and (config.wandb.log_train):
                rng, _ = jax.random.split(rng)
                log_eval_grid(unreplicate(pstate), score, rng, config.consistency, (16, *x.shape[1:]))

            # Save checkpoints periodically
            if (step % config.training.save_every_steps == 0) and (step != 0) and (jax.process_index() == 0):
                state_ckpt = unreplicate(pstate)
                checkpoints.save_checkpoint(ckpt_dir=workdir, target=state_ckpt, step=step, overwrite=True, keep=np.inf)

    logging.info("All done! Have a great day.")

    return unreplicate(pstate)


if __name__ == "__main__":

    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file("config", None, "File path to the training or sampling hyperparameter configuration.", lock_config=True)
    FLAGS(sys.argv)  # Parse flags

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())
    logging.info("JAX total visible devices: %r", jax.device_count())

    train(FLAGS.config)
