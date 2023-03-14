# Consistency Models

Implemention of Consistency Models ([Song et al 2023](https://arxiv.org/abs/2303.01469)) in Jax. When used as standalone generative models, consistency models achieve state of the art performance in one- and few-step generation, outperforming existing techniques for distilling diffusion models.

A "minified", more pedagogical version of the discrete-steps version is in [notebooks/consistency-minified.ipynb](notebooks/consistency-minified.ipynb). This example uses a simple MLP as the consistency function $f_\theta(x_t, t)$ so as not to obfuscate the details of the method itself.