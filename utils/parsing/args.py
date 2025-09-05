from .argparser import args_parsed
from DDPM.denoisers.denoisermlp.denoisermlp import DenoiserMLP
from DDPM.denoisers.denoiserunet.denoiserunet import DenoiserUNet
from loaders.dataloader import Dataloader
import torch

class Args:
    def __init__(self, args_parsed):

        self.args_parsed = args_parsed
        
        # ---------------------------
        # General / run control
        # ---------------------------
        self.t_max: int = args_parsed.t_max
        self.n_epochs: int = args_parsed.n_epochs
        self.verbose: bool = args_parsed.verbose
        self.model_name: str = args_parsed.model  

        # ---------------------------
        # Data
        # ---------------------------
        self.dataset_name: str = args_parsed.dataset
        self.batch_size: int = args_parsed.batch_size
        self.n_workers: int = args_parsed.n_workers
        self.dataset = Dataloader(self.dataset_name, self.batch_size, self.n_workers, self.n_epochs)
        self.image_shape = self.dataset.image_shape  # (C, H, W)

        # ---------------------------
        # Diffusion schedule / MC
        # ---------------------------
        self.num_trials: int = args_parsed.num_trials
        self.alpha_min: float = args_parsed.alpha_min
        self.alpha_max: float = args_parsed.alpha_max
        self.alpha_interp: str = args_parsed.alpha_interp

        # precompute alphas 
        if self.alpha_interp == "linear":
            self.alphas = torch.linspace(self.alpha_max, self.alpha_min, self.t_max)
        else:
            raise ValueError(f"Argument error : interpolation method {args_parsed.alpha_interp} not implemented")

        # ---------------------------
        # Training hyperparameters
        # ---------------------------
        self.learning_rate: float = args_parsed.learning_rate
        self.dropout: float = args_parsed.dropout
        self.optimizer_name = args_parsed.optimizer

        # ---------------------------
        # Model-shared knobs
        # ---------------------------
        self.time_dim: int = args_parsed.time_dim
        self.activation: str = args_parsed.activation
        self.norm: str = args_parsed.norm
        self.init_scheme: str = args_parsed.init_scheme
        self.hidden_sizes = args_parsed.hidden_sizes

        # ---------------------------
        # U-Net / CNN-specific
        # ---------------------------
        self.base_channels: int = args_parsed.base_channels
        self.channel_mults = args_parsed.channel_mults              # tuple[int,...]
        self.num_res_blocks = args_parsed.num_res_blocks            # int
        self.upsample: str = args_parsed.upsample                   # "convtranspose" | "nearest_conv"
        self.groups: int = args_parsed.groups                       # GroupNorm groups
        self.num_res_blocks_in_bottleneck: int = args_parsed.num_res_blocks_in_bottleneck
        self.kernel_size: int = args_parsed.kernel_size
        self.downsample: str = args_parsed.downsample               # "stride" | "pool" | "avgpool"
        self.time_hidden: int = args_parsed.time_hidden             # time MLP hidden

        # ---------------------------
        # Logging / checkpoints / outputs
        # ---------------------------
        self.log_dir: str = args_parsed.log_dir
        self.checkpoint_dir: str = args_parsed.checkpoint_dir
        self.save_frequency: int = args_parsed.save_frequency
        self.output_dir: str = args_parsed.output_dir
        self.fps: int = args_parsed.fps
        self.path_to_weights = args_parsed.path_to_weights

        # ---------------------------
        # Model construction
        # ---------------------------
        if args_parsed.model.lower() == "mlp":
            self.model = DenoiserMLP(
                self.dataset.image_shape,
                self.hidden_sizes,
                self.time_dim,
                self.activation,
                self.norm,
                self.dropout,
                self.init_scheme
            )
        elif args_parsed.model.lower() == "unet":
            self.model = DenoiserUNet(
                self.dataset.image_shape,
                self.base_channels,
                self.channel_mults,
                self.num_res_blocks,
                self.upsample,
                self.norm,
                self.groups,
                self.num_res_blocks_in_bottleneck,
                self.kernel_size,
                self.downsample,
                self.activation,
                self.time_dim,
                self.time_hidden,
                self.dropout
            )
        else:
            raise ValueError(f"Argument error : model {args_parsed.model} not implemented")

        # Create optimizer after model is initialized
        if self.optimizer_name.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


args = Args(args_parsed)
