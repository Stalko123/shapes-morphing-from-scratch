from .argparser_inference import args_parsed
from loaders.dataloader import Loader
import torch
import os
import yaml
from DDPM.denoisers.denoisermlp.denoisermlp import DenoiserMLP
from DDPM.denoisers.denoiserunet.denoiserunet import DenoiserUNet

class InferenceArgs:
    def __init__(self, args_parsed):
        
        # ---------------------------
        # General / run control
        # ---------------------------
        self.model_name: str = args_parsed.model
        if self.model_name.lower() == "unet":
            self.model_name = "u-net"  

        # ---------------------------
        # Data
        # ---------------------------
        self.dataset_name: str = args_parsed.dataset_name
        self.dataset = Loader(name=self.dataset_name, batch_size=1).dataset

        # ---------------------------
        # Logging / checkpoints / outputs
        # ---------------------------
        self.version: int = args_parsed.version
        self.output_dir: str = args_parsed.output_dir or f'./outputs/{self.dataset_name}_{self.model_name}_experiment/version_{self.version}'

        # ---------------------------
        # Viz
        # ---------------------------
        self.fps = args_parsed.fps
        self.seed = args_parsed.seed
        self.viz_noising = args_parsed.viz_noising
        self.viz_progressive_denoising = args_parsed.viz_progressive_denoising
        self.viz_denoising_from_t = args_parsed.viz_denoising_from_t
        self.generate_gifs = args_parsed.generate_gifs

        # Paths :
        self.path_to_weights = args_parsed.path_to_weights or f"./checkpoints/{self.dataset_name}_{self.model_name}_experiment/version_{self.version}/best.pth"
        self.path_to_yaml = args_parsed.path_to_yaml or f"./checkpoints/{self.dataset_name}_{self.model_name}_experiment/version_{self.version}/config.yml"
        assert os.path.exists(self.path_to_weights), f"InferenceArgs error : path {self.path_to_weights} doesn't exist."
        assert os.path.exists(self.path_to_yaml), f"InferenceArgs error : path {self.path_to_yaml} doesn't exist."

        # Others :
        with open(self.path_to_yaml, 'r') as f:

            def tuple_constructor(loader, node):
                return tuple(loader.construct_sequence(node))
            yaml.SafeLoader.add_constructor(
            u'tag:yaml.org,2002:python/tuple',
            tuple_constructor)  

            cfg = yaml.safe_load(f)
            self.t_max = cfg['t_max']
            self.image_shape = cfg['image_shape']
            
            # Load beta schedule parameters for proper DDPM initialization
            # Use command line overrides if provided, otherwise use YAML values, with fallback defaults
            self.beta_schedule = args_parsed.beta_schedule or cfg.get('beta_schedule', 'cosine')
            self.beta_start = args_parsed.beta_start or cfg.get('beta_start', 1e-4)
            self.beta_end = args_parsed.beta_end or cfg.get('beta_end', 0.02)
            
            self.model = self.load_model_from_yaml(cfg)
        self.model.load_state_dict(torch.load(self.path_to_weights)['model_state_dict'])



    def load_model_from_yaml(self, cfg):
        if self.model_name.lower() == "mlp":
            return DenoiserMLP(
                img_shape=cfg['image_shape'],
                hidden_sizes=cfg['hidden_sizes'],
                time_base_dim=cfg['time_base_dim'],
                time_output_dim=cfg['time_output_dim'],
                time_hidden=cfg['time_hidden'],
                activation=cfg['activation'],
                norm=cfg['norm_1d'],
                dropout=cfg['dropout'],
                init_scheme=cfg['init_scheme']
            )
        elif self.model_name.lower() == "u-net":
            return DenoiserUNet(
                img_shape=cfg['image_shape'],
                base_channels=cfg['base_channels'],
                channel_mults=cfg['channel_mults'],
                num_res_blocks=cfg['num_res_blocks'],
                upsample=cfg['upsample'],
                norm=cfg['norm_2d'],
                groups=cfg['groups'],
                num_res_blocks_in_bottleneck=cfg['num_res_blocks_in_bottleneck'],
                stem_kernel=cfg['stem_kernel'],
                head_kernel=cfg['head_kernel'],
                downsample=cfg['downsample'],
                activation=cfg['activation'],
                time_base_dim=cfg['time_base_dim'],
                time_output_dim=cfg['time_output_dim'],
                time_hidden=cfg['time_hidden'],
                dropout=cfg['dropout'],
                attn_stages=cfg['attn_stages'],
                attn_num_heads=cfg['attn_num_heads'],
                attn_in_bottleneck=cfg['attn_in_bottleneck'],
            )

args = InferenceArgs(args_parsed)