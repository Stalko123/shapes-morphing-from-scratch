import configargparse

p = configargparse.ArgParser(
    description="Training and evaluation config for DDPM denoisers (MLP/CNN/U-Net)."
)

# --------------------------------------------------------------------------------------
# General / experiment
# --------------------------------------------------------------------------------------
p.add_argument(
    '--t_max',
    default=1000,
    type=int,
    help='Number of diffusion timesteps T (forward noising steps).'
)
p.add_argument(
    '--n_epochs',
    required=True,
    type=int,
    help='Number of training epochs (set 0 to run inference only).'
)
p.add_argument(
    '--verbose',
    action='store_true',
    help='Enable verbose logging during training.'
)

# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
p.add_argument(
    '--dataset_name',
    default="CIFAR10",
    choices=["CIFAR10", "MNIST", "CelebA", "STL10"],
    type=str,
    help='Dataset to use (torchvision).'
)
p.add_argument(
    '--batch_size',
    default=32,
    type=int,
    help='Batch size.'
)
p.add_argument(
    '--num_workers',
    default=4,
    type=int,
    help='Number of worker processes for the DataLoader.'
)

# --------------------------------------------------------------------------------------
# Model selection (high level)
# --------------------------------------------------------------------------------------
p.add_argument(
    '--model',
    required=True,
    type=str,
    help='Denoiser backbone.'
)
p.add_argument(
    '--path_to_weights',
    default=None,
    type=str,
    help='Path to pretrained denoiser weights (if loading).'
)

# --------------------------------------------------------------------------------------
# Diffusion schedule / Monte Carlo
# --------------------------------------------------------------------------------------
p.add_argument(
    '--num_trials',
    default=100,
    type=int,
    help="Number of Monte Carlo samples used inside the loss (if applicable)."
)
p.add_argument(
    '--alpha_min',
    default=0.9,
    type=float,
    help="Minimum per-step α used to build the (linear) alpha schedule."
)
p.add_argument(
    '--alpha_max',
    default=1.0,
    type=float,
    help="Maximum per-step α used to build the (linear) alpha schedule."
)
p.add_argument(
    '--alpha_interp',
    default="linear",
    choices=["linear", "cosine", "quadratic"],
    type=str,
    help="Interpolation used to generate the per-step α schedule over t (default: linear)."
)

# --------------------------------------------------------------------------------------
# Training hyperparameters
# --------------------------------------------------------------------------------------
p.add_argument(
    '--learning_rate',
    default=1e-4,
    type=float,
    help='Learning rate.'
)
p.add_argument(
    '--dropout',
    default=0.0,
    type=float,
    help='Dropout probability used within the denoiser (applies to blocks that support it).'
)
p.add_argument(
    '--optimizer_name',
    default="Adam",
    type=str,
    help='Optimizer used for training'
)
p.add_argument(
    '--validation',
    action='store_true',
    help='Proceed to a validation step at each epoch.'
)
p.add_argument(
    '--val_ratio',
    default=0.05,
    type=float,
    help='Ratio between training and validation samples.'
)
p.add_argument(
    '--seed',
    default=42,
    type=int,
    help='Seed used to perform the split between the training and the validation sets.'
)
p.add_argument(
    '--test',
    action='store_true',
    help='Proceed to a test step at the end of training.'
)
p.add_argument(
    '--use_amp',
    action='store_true',
    help='Use AMP during training.'
)
p.add_argument(
    '--grad_clip',
    default=0.0,
    type=float,
    help='Maximal gradient norm (0.0 disables).'
)
p.add_argument(
    '--patience',
    default=0,
    type=int,
    help="Number of epochs without improvement before early-stopping (0 disables)."
)

# --------------------------------------------------------------------------------------
# MLP-specific (ignored if model != MLP)
# --------------------------------------------------------------------------------------
p.add_argument(
    '--hidden_sizes',
    default=(2048, 1024),
    type=lambda s: tuple(map(int, s.split(","))),
    help="Width of the MLP's hidden layers (comma-separated), e.g. '1,2,4'."
)
p.add_argument(
    '--norm_1d',
    default="layer",
    choices=["none", "layer", "batch"],
    type=str,
    help='Normalization type: for MLP use {"none","layer","batch"}; for CNN/U-Net use the 2D variants.'
)

# --------------------------------------------------------------------------------------
# U-Net / CNN-specific (ignored if model == MLP)
# --------------------------------------------------------------------------------------
p.add_argument(
    '--init_scheme',
    default="auto",
    choices=["auto", "kaiming_uniform", "kaiming_normal", "xavier_uniform", "xavier_normal", "orthogonal"],
    type=str,
    help="Initialization scheme for model weights."
)
p.add_argument(
    '--base_channels',
    default=64,
    type=int,
    help="Base number of channels at the first U-Net/CNN stage."
)
p.add_argument(
    '--channel_mults',
    default=(1, 2, 4),
    type=lambda s: tuple(map(int, s.split(","))),
    help="Per-stage channel multipliers (comma-separated), e.g. '1,2,4'."
)
p.add_argument(
    '--num_res_blocks',
    default=2,
    type=int,
    help="Residual blocks per stage (int). For per-stage lists, handle in code or extend this arg."
)
p.add_argument(
    '--num_res_blocks_in_bottleneck',
    default=2,
    type=int,
    help="Number of residual blocks in the bottleneck stage."
)
p.add_argument(
    '--downsample',
    default="stride",
    choices=["stride", "pool", "avgpool"],
    type=str,
    help="Downsampling method in encoder stages."
)
p.add_argument(
    '--upsample',
    default="nearest_conv",
    choices=["convtranspose", "nearest_conv"],
    type=str,
    help="Upsampling method in decoder stages."
)
p.add_argument(
    '--groups',
    default=32,
    type=int,
    help="Number of groups for GroupNorm (CNN/U-Net)."
)
p.add_argument(
    '--norm_2d',
    default="group",
    choices=["none","batch2d","group","instance","layer2d"],
    type=str,
    help="Normalization type: for CNN/U-Net use {'none','batch2d','group','instance','layer2d'}; for MLP use the 1D variants."
)
p.add_argument(
    '--stem_kernel',
    default=5,
    type=int,
    help="Convolution kernel size of the first hidden-layer."
)
p.add_argument(
    '--head_kernel',
    default=5,
    type=int,
    help="Convolution kernel size of the last hidden-layer."
)

# --------------------------------------------------------------------------------------
# Shared knobs
# --------------------------------------------------------------------------------------
p.add_argument(
    '--activation',
    default="silu",
    choices=["silu", "relu", "gelu", "tanh"],
    type=str,
    help='Activation function.'
)
p.add_argument(
    '--time_base_dim',
    default=128,
    type=int,
    help='Dimension of the time embedding'
)
p.add_argument(
    '--time_hidden',
    default=512,
    type=int,
    help='Hidden dimension of the time embedder MLP'
)
p.add_argument(
    '--time_output_dim',
    default=256,
    type=int,
    help='Output dimension of the time embedder MLP'
)

# --------------------------------------------------------------------------------------
# Logging / checkpoints / outputs
# --------------------------------------------------------------------------------------
p.add_argument(
    '--log_dir',
    default='./logs',
    type=str,
    help='Directory for TensorBoard logs.'
)
p.add_argument(
    '--checkpoint_dir',
    default='./checkpoints',
    type=str,
    help='Directory for model checkpoints.'
)
p.add_argument(
    '--save_frequency',
    default=10,
    type=int,
    help='Save a checkpoint every N epochs.'
)
p.add_argument(
    '--output_dir',
    default='./outputs',
    type=str,
    help='Directory for generated samples/visualizations.'
)
p.add_argument(
    '--fps',
    default=5,
    type=int,
    help='Frames per second for generated GIFs/videos.'
)

args_parsed = p.parse_args()