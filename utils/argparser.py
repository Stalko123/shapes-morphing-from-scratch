import configargparse

p = configargparse.ArgParser(default_config_files=['todo', 'todo'])

p.add(
    '--t_max',
    default=1000,
    type=int,
    help='maximum number of time steps of the blurring process'
)

p.add(
    '--dataset',
    default="CIFAR10",
    choices=["CIFAR10", "MNIST", "CelebA", "STL10"],
    type=str,
    help='name of the dataset that will be used for training (amongst torchvision datasets)'
)

p.add(
    '--model',
    default="MLP",
    choices=["MLP", "CNN", "U-Net"],
    type=str,
    help="model type for the denoiser"

)
p.add(
    '--num_trials',
    default=100,
    type=int,
    help="number of trials in the loss's Monte-Carlo simulation"
)

p.add(
    '--alphas',
    default="linear",
    type=str,
    help='interpolation type between 0 and 1 defining the alpha coefficients'
)

p.add(
    '--n_epochs',
    required=True,
    type = int,
    help='number of epochs (0 is inference)'
)

p.add(
    '-v',
    default=False,

    action='store_true',
    help='verbose'
)

# Training arguments
p.add(
    '--learning_rate',
    default=1e-4,
    type=float,
    help='learning rate for training'
)

p.add(
    '--batch_size',
    default=32,
    type=int,
    help='batch size for training'
)

p.add(
    '--n_workers',
    default=4,
    type=int,
    help='number of workers for data loading'
)

p.add(
    '--name_dataset',
    default="MNIST",
    type=str,
    help='name of the dataset (used by dataloader)'
)

# Model arguments for MLP
p.add(
    '--hidden_sizes',
    default=[1024, 1024],
    type=int,
    nargs='+',
    help='hidden layer sizes for MLP denoiser'
)

p.add(
    '--time_dim',
    default=128,
    type=int,
    help='time embedding dimension'
)

p.add(
    '--activation',
    default="silu",
    type=str,
    help='activation function for MLP'
)

p.add(
    '--norm',
    default="layer",
    type=str,
    help='normalization type for MLP'
)

p.add(
    '--dropout',
    default=0.0,
    type=float,
    help='dropout probability for MLP'
)

# Logging and checkpointing
p.add(
    '--log_dir',
    default='./logs',
    type=str,
    help='directory for tensorboard logs'
)

p.add(
    '--checkpoint_dir',
    default='./checkpoints',
    type=str,
    help='directory for model checkpoints'
)

p.add(
    '--save_frequency',
    default=10,
    type=int,
    help='frequency of saving checkpoints (in epochs)'
)

p.add(
    '--verbose',
    action='store_true',
    help='verbose logging during training'
)

# Visualization arguments
p.add(
    '--fps',
    default=5,
    type=int,
    help='frames per second for generated GIFs'
)

p.add(
    '--output_dir',
    default='./outputs',
    type=str,
    help='directory for output files'
)

p.add(
    '--denoiser_weights',
    default=None,
    type=str,
    help='path to denoiser weights'
)
args = p.parse_args()
