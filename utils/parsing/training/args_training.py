from .argparser_training import args_parsed
from DDPM.denoisers.denoisermlp.denoisermlp import DenoiserMLP
from DDPM.denoisers.denoiserunet.denoiserunet import DenoiserUNet
from loaders.dataloader import Loader
import torch
import os
import yaml
import datetime
import math


class StepLRWithMinLR(torch.optim.lr_scheduler._LRScheduler):
    """
    StepLR scheduler with minimum learning rate threshold.
    Prevents learning rate from going below a specified minimum value.
    """
    
    def __init__(self, optimizer, step_size, gamma=0.1, min_lr=1e-7, last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        super(StepLRWithMinLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        # Closed form calculation respecting minimum LR
        return [max(base_lr * self.gamma ** (self.last_epoch // self.step_size), self.min_lr)
                for base_lr in self.base_lrs]

class TrainingArgs:
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
        
        # ---------------------------
        # Training hyperparameters
        # ---------------------------
        self.learning_rate: float = args_parsed.learning_rate
        self.dropout: float = args_parsed.dropout
        self.optimizer_name: str = args_parsed.optimizer_name
        self.grad_clip: float = args_parsed.grad_clip
        self.patience: int = args_parsed.patience
        self.use_amp: bool = args_parsed.use_amp
        self.grad_accum: int = args_parsed.grad_accum

        # ---------------------------
        # Learning rate scheduler
        # ---------------------------
        self.scheduler_name: str = args_parsed.scheduler_name
        self.scheduler_step_size: int = args_parsed.scheduler_step_size
        self.scheduler_gamma: float = args_parsed.scheduler_gamma
        self.scheduler_t_max: int = args_parsed.scheduler_t_max if args_parsed.scheduler_t_max is not None else self.n_epochs
        self.scheduler_eta_min: float = args_parsed.scheduler_eta_min
        self.scheduler_t_0: int = args_parsed.scheduler_t_0
        self.scheduler_t_mult: int = args_parsed.scheduler_t_mult
        self.scheduler_min_lr: float = args_parsed.scheduler_min_lr

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
        self.channel_mults = args_parsed.channel_mults                  # tuple[int,...]
        self.num_res_blocks = args_parsed.num_res_blocks                # int
        self.upsample: str = args_parsed.upsample                       # "convtranspose" | "nearest_conv"
        self.groups: int = args_parsed.groups                           # GroupNorm groups
        self.num_res_blocks_in_bottleneck: int = args_parsed.num_res_blocks_in_bottleneck
        self.norm_2d: str = args_parsed.norm_2d                         # normalization method used
        self.stem_kernel: int = args_parsed.stem_kernel                 # kernel size of the stem layer
        self.head_kernel: int = args_parsed.head_kernel                 # kernel size of the head layer
        self.downsample: str = args_parsed.downsample                   # "stride" | "pool" | "avgpool"
        self.attn_stages = args_parsed.attn_stages                      # stages to use attention (list/tuple of booleans)
        self.attn_num_heads = args_parsed.attn_num_heads                # number of attention heads per attention block
        self.attn_in_bottleneck: bool = args_parsed.attn_in_bottleneck  # use attention in bottleneck ?

        # ---------------------------
        # Logging / checkpoints / outputs
        # ---------------------------
        self.exp_name: str = f"{self.dataset_name}_{self.model_name}_experiment"
        version_dir = self._get_next_version_dir(self.exp_name, args_parsed.checkpoint_dir)
        # use checkpoint_dir (aka experiments root) as the base for versioning
        self.checkpoint_dir: str = os.path.join(args_parsed.checkpoint_dir, self.exp_name, version_dir)
        self.log_dir: str = self.checkpoint_dir
        self.save_frequency: int = args_parsed.save_frequency
        self.output_dir: str = os.path.join(args_parsed.output_dir, self.exp_name, version_dir)
        self.path_to_weights: str = args_parsed.path_to_weights
        
        # Create directories if they don't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(self.checkpoint_dir, "tb_logs"), exist_ok=True)
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
                attn_stages=self.attn_stages,
                attn_num_heads=self.attn_num_heads,
                attn_in_bottleneck=self.attn_in_bottleneck
            )

        else:
            raise ValueError(f"Argument error : model {args_parsed.model} not implemented")

        if self.path_to_weights:
            try:
                self.model.load_state_dict(torch.load(self.path_to_weights))
            except:
                try:
                    self.model.load_state_dict(torch.load(self.path_to_weights)["model_state_dict"])
                except:
                    raise f"Invalid path to model weights {self.path_to_weights}"

        # Create optimizer after model is initialized
        if self.optimizer_name.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Create learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Save hyperparameters to YAML file
        self._save_hyperparameters(args_parsed)

    def _create_scheduler(self):
        """Create learning rate scheduler based on the specified scheduler name."""
        if self.scheduler_name.lower() == "none":
            return None
        elif self.scheduler_name.lower() == "step":
            # Use custom StepLR with minimum LR limit
            return StepLRWithMinLR(
                self.optimizer, 
                step_size=self.scheduler_step_size, 
                gamma=self.scheduler_gamma,
                min_lr=self.scheduler_min_lr
            )
        elif self.scheduler_name.lower() == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, 
                gamma=self.scheduler_gamma
            )
        elif self.scheduler_name.lower() == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.scheduler_t_max, 
                eta_min=self.scheduler_eta_min
            )
        elif self.scheduler_name.lower() == "cosine_with_restarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=self.scheduler_t_0, 
                T_mult=self.scheduler_t_mult, 
                eta_min=self.scheduler_eta_min
            )
        else:
            raise ValueError(f"Unknown scheduler name: {self.scheduler_name}")

    def _get_next_version_dir(self, exp_name, base_dir):
        """Find the next available version directory for this experiment under base_dir/exp_name."""
        exp_base_dir = os.path.join(base_dir, exp_name)
        os.makedirs(exp_base_dir, exist_ok=True)
        entries = [d for d in os.listdir(exp_base_dir) if os.path.isdir(os.path.join(exp_base_dir, d)) and d.startswith("version_")]
        if not entries:
            return "version_0"
        max_idx = -1
        for d in entries:
            try:
                idx = int(d.split("_")[1])
                max_idx = max(max_idx, idx)
            except Exception:
                continue
        return f"version_{max_idx + 1}"
    

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
        
        # Save to YAML file at the root of version_i folder (same as checkpoints)
        config_path = os.path.join(self.checkpoint_dir, 'config.yml')
        with open(config_path, 'w') as f:
            yaml.dump(hyperparams, f, default_flow_style=False, sort_keys=True)
        
        print(f"Hyperparameters saved to: {config_path}")


args = TrainingArgs(args_parsed)