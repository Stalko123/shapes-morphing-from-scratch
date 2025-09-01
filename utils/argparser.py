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
    type=bool,
    action='store_true',
    help='verbose'
)

args = p.parse_args()