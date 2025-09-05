from argparser import args_parsed
from DDPM.denoisers.denoisermlp.denoisermlp import DenoiserMLP
from DDPM.denoisers.denoiserunet.denoiserunet import DenoiserUNet
from loaders.dataloader import DataLoader
import torch

class Args:
    def __init__(self, args_parsed):

        self.args_parsed = args_parsed
        
        # ---------------------------
        # General / run control
        # ---------------------------
        self.t_max: int = args.t_max
        self.n_epochs: int = args.n_epochs
        self.verbose: bool = args.verbose
        self.model_name: str = args.model  

        # ---------------------------
        # Data
        # ---------------------------
        self.dataset_name: str = args.dataset
        self.batch_size: int = args.batch_size
        self.n_workers: int = args.n_workers
        self.dataset = DataLoader(self.dataset_name, self.batch_size, self.n_workers, self.n_epochs)
        self.image_shape = self.dataset.image_shape  # (C, H, W)

        # ---------------------------
        # Diffusion schedule / MC
        # ---------------------------
        self.num_trials: int = args.num_trials
        self.alpha_min: float = args.alpha_min
        self.alpha_max: float = args.alpha_max
        self.alpha_interp: str = args.alpha_interp

        # precompute alphas 
        if self.alpha_interp == "linear":
            self.alphas = torch.linspace(self.alpha_max, self.alpha_min, self.t_max)
        else:
            raise ValueError(f"Argument error : interpolation method {args.alpha_interp} not implemented")

        # ---------------------------
        # Training hyperparameters
        # ---------------------------
        self.learning_rate: float = args.learning_rate
        self.dropout: float = args.dropout
        self.optimizer_name = args.optimizer
        if self.optimizer_name.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # ---------------------------
        # Model-shared knobs
        # ---------------------------
        self.time_dim: int = args.time_dim
        self.activation: str = args.activation
        self.norm: str = args.norm
        self.init_scheme: str = args.init_scheme
        self.hidden_sizes = args.hidden_sizes

        # ---------------------------
        # U-Net / CNN-specific
        # ---------------------------
        self.base_channels: int = args.base_channels
        self.channel_mults = args.channel_mults              # tuple[int,...]
        self.num_res_blocks = args.num_res_blocks            # int
        self.upsample: str = args.upsample                   # "convtranspose" | "nearest_conv"
        self.groups: int = args.groups                       # GroupNorm groups
        self.num_res_blocks_in_bottleneck: int = args.num_res_blocks_in_bottleneck
        self.kernel_size: int = args.kernel_size
        self.downsample: str = args.downsample               # "stride" | "pool" | "avgpool"
        self.time_hidden: int = args.time_hidden             # time MLP hidden

        # ---------------------------
        # Logging / checkpoints / outputs
        # ---------------------------
        self.log_dir: str = args.log_dir
        self.checkpoint_dir: str = args.checkpoint_dir
        self.save_frequency: int = args.save_frequency
        self.output_dir: str = args.output_dir
        self.fps: int = args.fps
        self.path_to_weights = args.path_to_weights

        # ---------------------------
        # Model construction
        # ---------------------------
        if args.model.lower() == "mlp":
            self.model = DenoiserMLP(
                self.dataset.image_shape,
                self.hidden_sizes,
                self.time_dim,
                self.activation,
                self.norm,
                self.dropout,
                self.init_scheme
            )
        elif args.model.lower() == "unet":
            self.model = DenoiserUNet(
                self.dataset.image_shape,
                self.base_channels,
                self.channel_mults,
                self.num_res_blocks,
                self.upsample,
                self.norm,
                self.groups,  # <- was args.group; using the parsed 'groups'
                self.num_res_blocks_in_bottleneck,
                self.kernel_size,
                self.downsample,
                self.activation,
                self.time_dim,
                self.time_hidden,
                self.dropout
            )
        else:
            raise ValueError(f"Argument error : model {args.model} not implemented")


args = Args(args_parsed)