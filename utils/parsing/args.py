from .argparser import args_parsed
from DDPM.denoisers.denoisermlp.denoisermlp import DenoiserMLP
from DDPM.denoisers.denoiserunet.denoiserunet import DenoiserUNet
from loaders.dataloader import Loader
import torch
import os
import yaml
import datetime
import math

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
        self.dataset_name: str = args_parsed.dataset_name
        self.batch_size: int = args_parsed.batch_size
        self.num_workers: int = args_parsed.num_workers
        self.validation = args_parsed.validation
        self.test = args_parsed.test
        self.val_ratio = args_parsed.val_ratio
        self.seed = args_parsed.seed
        if self.validation:
            loader = Loader(name=self.dataset_name,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            split="train+val",
                            val_ratio=self.val_ratio,
                            seed = self.seed
                            )
            self.image_shape = loader.image_shape
            self.dataloader_train = loader.dataloader
            self.dataloader_val = loader.dataloader_val
        else :
            self.dataloader_train = Loader(name=self.dataset_name,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            split="train",
                            )
            self.image_shape = self.dataloader_train.image_shape
        if self.test :
            self.dataloader_test = Loader(name=self.dataset_name,
                                       batch_size=self.batch_size,
                                       num_workers=self.num_workers,
                                       split="test").dataloader
        # ---------------------------
        # Diffusion schedule / MC
        # ---------------------------
        self.num_trials: int = args_parsed.num_trials
        self.alpha_min: float = args_parsed.alpha_min
        self.alpha_max: float = args_parsed.alpha_max
        self.alpha_interp: str = args_parsed.alpha_interp

        if self.alpha_interp == "linear":
            self.alphas = torch.linspace(self.alpha_max, self.alpha_min, self.t_max)
            self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        elif self.alpha_interp == "cosine":
            self.alphas, self.alphas_bar = self.cosine_alpha_bar()
        else:
            raise ValueError(f"Argument error : interpolation method {args_parsed.alpha_interp} not implemented")
        
        if self.verbose:
            print(f"Diffusion process info : last ᾱ is {self.alphas_bar[-1]}")

        # ---------------------------
        # Training hyperparameters
        # ---------------------------
        self.learning_rate: float = args_parsed.learning_rate
        self.dropout: float = args_parsed.dropout
        self.optimizer_name: str = args_parsed.optimizer_name
        self.grad_clip: float = args_parsed.grad_clip
        self.patience: int = args_parsed.patience
        self.use_amp: bool = args_parsed.use_amp

        # ---------------------------
        # Model-shared knobs
        # ---------------------------
        self.activation: str = args_parsed.activation
        self.time_base_dim: int = args_parsed.time_base_dim
        self.time_hidden: int = args_parsed.time_hidden
        self.time_output_dim: int = args_parsed.time_output_dim
        self.init_scheme: str = args_parsed.init_scheme

        # ---------------------------
        # MLP-specific
        # ---------------------------
        self.hidden_sizes = args_parsed.hidden_sizes
        self.norm_1d: str = args_parsed.norm_1d                     # normalization method used

        # ---------------------------
        # U-Net / CNN-specific
        # ---------------------------
        self.base_channels: int = args_parsed.base_channels
        self.channel_mults = args_parsed.channel_mults              # tuple[int,...]
        self.num_res_blocks = args_parsed.num_res_blocks            # int
        self.upsample: str = args_parsed.upsample                   # "convtranspose" | "nearest_conv"
        self.groups: int = args_parsed.groups                       # GroupNorm groups
        self.num_res_blocks_in_bottleneck: int = args_parsed.num_res_blocks_in_bottleneck
        self.norm_2d: str = args_parsed.norm_2d                     # normalization method used
        self.stem_kernel: int = args_parsed.stem_kernel             # kernel size of the stem layer
        self.head_kernel: int = args_parsed.head_kernel             # kernel size of the head layer
        self.downsample: str = args_parsed.downsample               # "stride" | "pool" | "avgpool"

        # ---------------------------
        # Logging / checkpoints / outputs
        # ---------------------------
        self.exp_name: str = f"{self.dataset_name}_{self.model_name}_experiment"
        version_dir = self._get_next_version_dir(self.exp_name, args_parsed.log_dir)
        self.log_dir: str = os.path.join(args_parsed.log_dir, self.exp_name, version_dir)
        self.checkpoint_dir: str = os.path.join(args_parsed.checkpoint_dir, self.exp_name, version_dir)
        self.save_frequency: int = args_parsed.save_frequency
        self.output_dir: str = os.path.join(args_parsed.output_dir, self.exp_name, version_dir)
        self.fps: int = args_parsed.fps
        self.path_to_weights: str = args_parsed.path_to_weights
        
        # Create directories if they don't exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Experiment: {self.exp_name}")
        print(f"Version: {version_dir}")
        print(f"Logs will be saved to: {self.log_dir}")

        # ---------------------------
        # Model construction
        # ---------------------------
        if args_parsed.model.lower() == "mlp":
            self.model = DenoiserMLP(
                img_shape=self.image_shape,
                hidden_sizes=self.hidden_sizes,
                time_base_dim=self.time_base_dim,
                time_output_dim=self.time_output_dim,
                time_hidden=self.time_hidden,
                activation=self.activation,
                norm=self.norm_1d,
                dropout=self.dropout,
                init_scheme=self.init_scheme
            )
        elif args_parsed.model.lower() == "unet" or args_parsed.model.lower() == "u-net":
            self.model = DenoiserUNet(
                img_shape=self.image_shape,
                base_channels=self.base_channels,
                channel_mults=self.channel_mults,
                num_res_blocks=self.num_res_blocks,
                upsample=self.upsample,
                norm=self.norm_2d,
                groups=self.groups,
                num_res_blocks_in_bottleneck=self.num_res_blocks_in_bottleneck,
                stem_kernel=self.stem_kernel,
                head_kernel=self.head_kernel,
                downsample=self.downsample,
                activation=self.activation,
                time_base_dim=self.time_base_dim,
                time_output_dim=self.time_output_dim,
                time_hidden=self.time_hidden,
                dropout=self.dropout,
            )

        else:
            raise ValueError(f"Argument error : model {args_parsed.model} not implemented")

        # Create optimizer after model is initialized
        if self.optimizer_name.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Save hyperparameters to YAML file
        self._save_hyperparameters(args_parsed)


    def cosine_alpha_bar(self, s: float = 0.08):
        """
        Nichol & Dhariwal alphas : enable linear decay of signal to noise ratio
        """
        T = self.t_max
        t = torch.linspace(0, T, T+1, dtype=torch.float64) / T
        f = torch.cos(( (t + s) / (1 + s) ) * math.pi / 2) ** 2
        alpha_bar = (f / f[0]).clamp(min=1e-12, max=1.0)
        alphas = (alpha_bar[1:] / alpha_bar[:-1]).clamp(min=1e-6, max=1-1e-6)
        return alphas.float(), alpha_bar.float()

    def _get_next_version_dir(self, exp_name, base_log_dir):
        """Find the next available version directory for this experiment."""
        exp_base_dir = os.path.join(base_log_dir, exp_name)
        
        # If experiment directory doesn't exist, start with version_0
        if not os.path.exists(exp_base_dir):
            return "version_0"
        
        # Find existing version directories
        existing_versions = []
        for item in os.listdir(exp_base_dir):
            item_path = os.path.join(exp_base_dir, item)
            if os.path.isdir(item_path) and item.startswith("version_"):
                try:
                    version_num = int(item.split("_")[1])
                    existing_versions.append(version_num)
                except (ValueError, IndexError):
                    # Skip directories that don't follow version_N format
                    continue
        
        # Get the next version number
        if existing_versions:
            next_version = max(existing_versions) + 1
        else:
            next_version = 0
        
        return f"version_{next_version}"

    def _save_hyperparameters(self, args_parsed):
        """Save all hyperparameters to a YAML file in the experiment directory."""
        # Convert args_parsed to dictionary
        hyperparams = vars(args_parsed)
        
        # Add some additional computed parameters
        hyperparams['image_shape'] = list(self.image_shape)
        hyperparams['total_parameters'] = sum(p.numel() for p in self.model.parameters())
        hyperparams['trainable_parameters'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Add experiment versioning info
        hyperparams['experiment_version'] = os.path.basename(self.log_dir)  # e.g., "version_1"
        hyperparams['timestamp'] = datetime.datetime.now().isoformat()
        hyperparams['full_log_dir'] = self.log_dir
        hyperparams['full_checkpoint_dir'] = self.checkpoint_dir
        hyperparams['full_output_dir'] = self.output_dir
        
        # Save to YAML file
        config_path = os.path.join(self.log_dir, 'config.yml')
        with open(config_path, 'w') as f:
            yaml.dump(hyperparams, f, default_flow_style=False, sort_keys=True)
        
        print(f"Hyperparameters saved to: {config_path}")


args = Args(args_parsed)
