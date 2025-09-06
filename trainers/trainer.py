import torch
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import numpy as np
from tqdm import tqdm

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Comment bapt : maybe avoid this for final version

from DDPM.ddpm import DDPM

class Trainer:
    """
    Training class that uses all existing components without modifying them.
    Handles model initialization, training loop, and tensorboard logging.
    """
    
    def __init__(self, args):

        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize dataloader
        self.dataloader = args.dataset
        
        self.n_epochs = args.n_epochs
        
        self.alphas = args.alphas
        self.alphas = self.alphas.to(self.device)
        
        # Initialize denoiser model
        self.denoiser = args.model
        self.denoiser.to(self.device)
        
        self.ddpm = DDPM(args)
        
        # Initialize optimizer
        self.optimizer = args.optimizer
        
        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'tb_logs'))
        
        # Training state
        self.global_step = 0
        
    
    def train_step(self, batch: torch.Tensor) -> float:
        """
        Single training step.
        """
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
        """
        Main training loop.
        """
        self.denoiser.train()

        for epoch in range(self.n_epochs):
            epoch_losses = []

            # create tqdm progress bar for the batches
            pbar = tqdm(self.dataloader.dataloader, desc=f"Epoch {epoch+1}/{self.n_epochs}")
            
            for batch_idx, (data, _) in enumerate(pbar):
                # Training step
                loss = self.train_step(data)
                epoch_losses.append(loss)

                # Log to tensorboard
                self.writer.add_scalar('Loss/Train_Step', loss, self.global_step)

                # Verbose logging 
                # if getattr(self.args, 'verbose', False) and batch_idx % 100 == 0: 
                #   print(f'Epoch: {epoch+1}/{self.n_epochs}, Batch: {batch_idx}/{len(self.dataloader.dataloader)}, Loss: {loss:.6f}')

                # update tqdm bar with current loss
                pbar.set_postfix(loss=f"{loss:.6f}")

                self.global_step += 1

            # Log epoch statistics
            avg_loss = np.mean(epoch_losses)
            self.writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)

            print(f"Epoch {epoch+1}/{self.n_epochs} completed. Average Loss: {avg_loss:.6f}")

            # Save checkpoint
            if (epoch + 1) % getattr(self.args, 'save_frequency', 10) == 0:
                self.save_checkpoint(epoch + 1)

        # Save final model
        self.save_checkpoint(self.n_epochs, final=True)
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
            'img_shape': self.args.dataset.image_shape
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
    from utils.parsing.args import args
    
    print("Starting training...")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Arguments: {vars(args.args_parsed)}")
    
    # Create trainer and start training
    trainer = Trainer(args)
    trainer.train()
    
    print("Training completed!")


if __name__ == '__main__':
    main()