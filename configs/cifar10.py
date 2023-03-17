import ml_collections


def get_config():

    config = ml_collections.ConfigDict()

    # Wandb logging
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.entity = None
    wandb.project = "consistency-models"
    wandb.group = "cifar10"
    wandb.job_type = "training"
    wandb.name = None
    wandb.log_train = True

    # Vartiational diffusion model
    config.consistency = consistency = ml_collections.ConfigDict()
    consistency.d_t_embed = 16
    consistency.sigma_data = 0.5
    consistency.s0 = 2
    consistency.s1 = 150
    consistency.sigma = 7.0
    consistency.mu0 = 0.9
    consistency.eps = 0.002
    consistency.T = 80.0

    # # Score model (MLP-Mixer)
    # config.score = score = ml_collections.ConfigDict()
    # score.score = "mlp_mixer"
    # score.patch_size = 2
    # score.num_blocks = 12
    # score.hidden_dim = 256
    # score.tokens_mlp_dim = 512
    # score.channels_mlp_dim = 512

    # Score model (UNet)
    config.score = score = ml_collections.ConfigDict()
    score.score = "unet"
    score.hidden_channels = 64
    score.num_layers = 4

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.half_precision = False
    training.batch_size = 256  # Must be divisible by number of devices; this is the total batch size, not per-device
    training.n_train_steps = 1_001_000
    training.log_every_steps = 100
    training.eval_every_steps = 1000  # Eval not yet supported
    training.save_every_steps = 100_000

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "cifar10"
    data.num_classes = 10

    # Optimizer (AdamW)
    config.optim = optim = ml_collections.ConfigDict()
    optim.learning_rate = 3e-4
    optim.weight_decay = 1e-4

    config.seed = 42

    return config
