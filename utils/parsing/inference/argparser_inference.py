import configargparse

p = configargparse.ArgParser(
    description="Inference config for DDPM inference."
)

p.add_argument(
    '--model',
    required=True,
    type=str,
    help='Denoiser backbone used.'
)
p.add_argument(
    '--dataset_name',
    required=True,
    choices=["CIFAR10", "MNIST", "CelebA", "STL10"],
    type=str,
    help='Dataset used.'
)
p.add_argument(
    '--version',
    default=0,
    type=int,
    help='Version to use for inference.'
)
p.add_argument(
    '--output_dir',
    default=None,
    type=str,
    help='Directory for generated samples/visualizations.'
)
p.add_argument(
    '--path_to_weights',
    default=None,
    type=str,
    help='Path to the weights to load.'
)
p.add_argument(
    '--path_to_yaml',
    default=None,
    type=str,
    help='Path to the yaml config file to use.'
)
p.add_argument(
    '--fps',
    default=5,
    type=int,
    help='Frames per second for generated GIFs/videos.'
)
p.add_argument(
    '--seed',
    default=42,
    type=int,
    help='Seed for the randomness of the Visualizer.'
)
p.add_argument(
    '--viz_noising',
    action='store_true',
    help='Flag to visualize a noising process.'
)
p.add_argument(
    '--viz_progressive_denoising',
    action='store_true',
    help='Flag to visualize a denoising process.'
)
p.add_argument(
    '--viz_denoising_from_t',
    default=0,
    type=lambda s: tuple(map(int, s.split(","))),
    help="Instants from which denoising is to be visualized (0 disables)."
)
p.add_argument(
    '--generate_gifs',
    default=0,
    type=int,
    help='Number of gifs to generate (0 disables).'
)

args_parsed = p.parse_args()