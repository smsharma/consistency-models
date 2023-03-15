# Consistency Models
_Work in progress_

Implemention of Consistency Models ([Song et al 2023](https://arxiv.org/abs/2303.01469)) in Jax. When used as standalone generative models, consistency models achieve state of the art performance in one- and few-step generation, outperforming existing techniques for distilling diffusion models.

A simple implementation of the discrete-steps version is in [notebooks/consistency-mnist.ipynb](notebooks/consistency-mnist.ipynb). This example uses a simple MLP-Mixer as the backbone for the consistency function $f_\theta(x_t, t)$.