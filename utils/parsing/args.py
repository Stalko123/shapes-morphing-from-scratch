from .argparser import args_parsed
from DDPM.denoisers.denoisermlp.denoisermlp import DenoiserMLP
from DDPM.denoisers.denoiserunet.denoiserunet import DenoiserUNet
from loaders.dataloader import Dataloader
import torch
import os
import yaml

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

        # ---------------------------
        # MLP-specific
        # ---------------------------
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
        self.exp_name: str = args_parsed.exp_name
        
        # Get the next version number for this experiment
        version_dir = self._get_next_version_dir(args_parsed.exp_name, args_parsed.log_dir)
        
        self.log_dir: str = os.path.join(args_parsed.log_dir, self.exp_name, version_dir)
        self.checkpoint_dir: str = os.path.join(args_parsed.checkpoint_dir, self.exp_name, version_dir)
        self.save_frequency: int = args_parsed.save_frequency
        self.output_dir: str = os.path.join(args_parsed.output_dir, self.exp_name, version_dir)
        self.fps: int = args_parsed.fps
        self.path_to_weights = args_parsed.path_to_weights
        
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
                self.dataset.image_shape,
                self.hidden_sizes,
                self.time_dim,
                self.activation,
                self.norm,
                self.dropout,
                self.init_scheme
            )
        elif args_parsed.model.lower() == "unet" or args_parsed.model.lower() == "u-net":
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

        # Save hyperparameters to YAML file
        self._save_hyperparameters(args_parsed)

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
        import datetime
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
