import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from typing import Optional
import numpy as np

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import Dataloader
from DDPM.ddpm import DDPM
from DDPM.denoisers.denoisermlp import DenoiserMLP


class Trainer:
    """
    Training class that uses all existing components without modifying them.
    Handles model initialization, training loop, and tensorboard logging.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize dataloader
        self.dataloader = Dataloader(args)
        
        # Get dataset dimensions from first batch
        sample_batch, _ = next(iter(self.dataloader.dataloader))
        self.img_shape = sample_batch.shape[1:]  # (C, H, W)
        
        # Create dataset attribute object for DDPM
        self.dataset = type('Dataset', (), {
            'C': self.img_shape[0],
            'H': self.img_shape[1], 
            'W': self.img_shape[2]
        })()
        
        # Initialize alphas
        self.alphas = self._create_alphas_schedule(args.alphas, args.t_max)
        self.alphas = self.alphas.to(self.device)
        
        # Initialize denoiser model
        self.denoiser = self._create_denoiser(args.model, self.img_shape)
        self.denoiser.to(self.device)
        
        # Create modified args for DDPM
        ddpm_args = type('Args', (), {
            'dataset': self.dataset,
            'denoiser': self.denoiser,
            'alphas': self.alphas,
            'num_trials': args.num_trials,
            't_max': args.t_max
        })()
        
        # Initialize DDPM
        self.ddpm = DDPM(ddpm_args)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.denoiser.parameters(), lr=args.learning_rate)
        
        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'tb_logs'))
        
        # Training state
        self.global_step = 0
        
    def _create_alphas_schedule(self, schedule_type: str, t_max: int) -> torch.Tensor:
        """Create alphas schedule based on type."""
        if schedule_type == "linear":
            beta_start = 1e-4
            beta_end = 0.02
            betas = torch.linspace(beta_start, beta_end, t_max)
            alphas = 1.0 - betas
            return alphas
        elif schedule_type == "cosine":
            # Cosine schedule as in improved DDPM
            steps = t_max + 1
            s = 0.008
            x = torch.linspace(0, t_max, steps)
            alphas_cumprod = torch.cos((x / t_max + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            alphas = 1.0 - torch.clamp(betas, 0.0001, 0.9999)
            return alphas
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    def _create_denoiser(self, model_type: str, img_shape: tuple) -> nn.Module:
        """Create denoiser model based on type."""
        if model_type == "MLP":
            return DenoiserMLP(
                img_shape=img_shape,
                hidden_sizes=getattr(self.args, 'hidden_sizes', [1024, 1024]),
                time_dim=getattr(self.args, 'time_dim', 128),
                activation=getattr(self.args, 'activation', 'silu'),
                norm=getattr(self.args, 'norm', 'layer'),
                dropout=getattr(self.args, 'dropout', 0.0)
            )
        elif model_type == "CNN":
            # CNN denoiser would be implemented here
            raise NotImplementedError("CNN denoiser not implemented yet")
        elif model_type == "U-Net":
            # U-Net denoiser would be implemented here
            raise NotImplementedError("U-Net denoiser not implemented yet")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_step(self, batch: torch.Tensor) -> float:
        """Single training step."""
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = batch.to(self.device)
        
        # Compute loss using DDPM
        loss = self.ddpm.computeLoss(batch)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self):
        """Main training loop."""
        self.denoiser.train()
        
        for epoch in range(self.args.n_epochs):
            epoch_losses = []
            
            for batch_idx, (data, _) in enumerate(self.dataloader.dataloader):
                # Training step
                loss = self.train_step(data)
                epoch_losses.append(loss)
                
                # Log to tensorboard
                self.writer.add_scalar('Loss/Train_Step', loss, self.global_step)
                
                # Verbose logging
                if getattr(self.args, 'verbose', False) and batch_idx % 100 == 0:
                    print(f'Epoch: {epoch+1}/{self.args.n_epochs}, '
                          f'Batch: {batch_idx}/{len(self.dataloader.dataloader)}, '
                          f'Loss: {loss:.6f}')
                
                self.global_step += 1
            
            # Log epoch statistics
            avg_loss = np.mean(epoch_losses)
            self.writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)
            
            print(f'Epoch {epoch+1}/{self.args.n_epochs} completed. Average Loss: {avg_loss:.6f}')
            
            # Save checkpoint
            if (epoch + 1) % getattr(self.args, 'save_frequency', 10) == 0:
                self.save_checkpoint(epoch + 1)
        
        # Save final model
        self.save_checkpoint(self.args.n_epochs, final=True)
        self.writer.close()
        
    def save_checkpoint(self, epoch: int, final: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = getattr(self.args, 'checkpoint_dir', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.denoiser.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'args': self.args,
            'img_shape': self.img_shape
        }
        
        if final:
            path = os.path.join(checkpoint_dir, 'final_model.pth')
        else:
            path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, path)
        print(f'Checkpoint saved: {path}')


def main():
    """Main training function."""
    # Import args here to avoid circular imports
    from utils.argparser import args
    
    print("Starting training...")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Arguments: {vars(args)}")
    
    # Create trainer and start training
    trainer = Trainer(args)
    trainer.train()
    
    print("Training completed!")


if __name__ == '__main__':
    main()
