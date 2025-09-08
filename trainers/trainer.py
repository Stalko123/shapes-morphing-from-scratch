import os
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from DDPM.ddpm import DDPM


def _to_device(batch, device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Handle datasets that return either (image, label) or just image.
    """
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        x, y = batch
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    # fallback if dataset returns only images
    x = batch if torch.is_tensor(batch) else batch[0]
    return x.to(device, non_blocking=True), None


class Trainer:
    """
    Handles model init, training/validation/test loops, logging, checkpoints, early stopping.
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data
        self.train_loader: DataLoader = args.dataloader_train
        self.has_val = hasattr(args, "dataloader_val") and args.dataloader_val is not None
        self.val_loader: Optional[DataLoader] = args.dataloader_val if self.has_val else None
        self.has_test = hasattr(args, "dataloader_test") and args.dataloader_test is not None
        self.test_loader: Optional[DataLoader] = args.dataloader_test if self.has_test else None

        # Model & DDPM wrapper
        self.alphas = args.alphas.to(self.device)
        self.denoiser: nn.Module = args.model.to(self.device)
        self.ddpm = DDPM(args)

        # Optimizer
        self.optimizer = args.optimizer

        # Logging
        self.verbose = int(getattr(args, "verbose", 1))  # 0: quiet, 1: epoch logs + tqdm
        log_dir = os.path.join(getattr(args, "log_dir", "./logs"), "tb_logs")
        self.writer = SummaryWriter(log_dir=log_dir)

        # Train cfg
        self.n_epochs = int(args.n_epochs)
        self.global_step = 0
        self.use_amp = bool(getattr(args, "use_amp", False))
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.grad_clip = float(getattr(args, "grad_clip", 0.0))  # 0 disables

        # Early stopping
        self.patience = int(getattr(args, "patience", 0))  # 0 disables
        self.use_early_stopping = self.patience > 0 and self.has_val
        self.best_val = float("inf")
        self.bad_epochs = 0
        self.best_state = None  # dict of model params
        self.best_epoch = 0

        # Checkpoints
        self.checkpoint_dir = getattr(args, "checkpoint_dir", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_frequency = getattr(args, "save_frequency", 10)

        if self.verbose:
            print(f"[Trainer] Device: {self.device}")
            print(f"[Trainer] Train batches: {len(self.train_loader)}"
                  + (f" | Val batches: {len(self.val_loader)}" if self.has_val else "")
                  + (f" | Test batches: {len(self.test_loader)}" if self.has_test else ""))
            print(f"[Trainer] AMP: {self.use_amp} | EarlyStopping: {self.use_early_stopping} (patience={self.patience})")

    # Steps

    def train_step(self, batch) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        x, _ = _to_device(batch, self.device)
        with torch.amp.autocast(enabled=self.use_amp, device_type=x.device.type):
            loss = self.ddpm.computeLoss(x)
        self.scaler.scale(loss).backward()
        if self.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return float(loss.detach().cpu().item())

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> float:
        self.denoiser.eval()
        losses = []
        for batch in loader:
            x, _ = _to_device(batch, self.device)
            with torch.amp.autocast(enabled=self.use_amp, device_type=x.device.type):
                loss = self.ddpm.computeLoss(x)
            losses.append(float(loss.detach().cpu().item()))
        self.denoiser.train()
        return float(np.mean(losses)) if len(losses) else float("nan")

    # Loop

    def train(self):
        self.denoiser.train()
        stop_early = False

        for epoch in range(1, self.n_epochs + 1):
            # train epoch
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.n_epochs}", disable=(self.verbose == 0))
            epoch_losses = []

            for batch in pbar:
                loss = self.train_step(batch)
                epoch_losses.append(loss)

                if self.writer:
                    self.writer.add_scalar("loss/train_step", loss, self.global_step)

                pbar.set_postfix(train_loss=f"{loss:.6f}")
                self.global_step += 1

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            if self.writer:
                self.writer.add_scalar("loss/train_epoch", train_loss, epoch)

            # validation
            if self.has_val:
                val_loss = self.eval_epoch(self.val_loader)
                if self.writer:
                    self.writer.add_scalar("loss/val_epoch", val_loss, epoch)

                if self.verbose:
                    print(f"Epoch {epoch}: train={train_loss:.6f} | val={val_loss:.6f}")

                if self.use_early_stopping:
                    improved = (val_loss < self.best_val)
                    if improved:
                        self.best_val = val_loss
                        self.bad_epochs = 0
                        self.best_epoch = epoch
                        self.best_state = {k: v.detach().cpu().clone() for k, v in self.denoiser.state_dict().items()}
                        self._save_checkpoint(epoch, is_best=True)
                    else:
                        self.bad_epochs += 1
                        if self.bad_epochs >= self.patience:
                            if self.verbose:
                                print(f"[EarlyStopping] Stopping at epoch {epoch}. "
                                      f"Best val={self.best_val:.6f} @ epoch {self.best_epoch}.")
                            if self.best_state is not None:
                                self.denoiser.load_state_dict(self.best_state)
                            stop_early = True
            else:
                if self.verbose:
                    print(f"Epoch {epoch}: train={train_loss:.6f}")

            # periodic checkpoint
            if (epoch % self.save_frequency) == 0:
                self._save_checkpoint(epoch)

            if stop_early:
                break

        # final save (last weights)
        self._save_checkpoint(epoch="final")

        # test
        if self.has_test:
            test_loss = self.eval_epoch(self.test_loader)
            if self.writer:
                self.writer.add_scalar("loss/test", test_loss, getattr(self, "best_epoch", self.n_epochs))
            print(f"[Test] loss={test_loss:.6f}")

        if self.writer:
            self.writer.close()

    # checkpoints

    def _save_checkpoint(self, epoch, is_best: bool = False):
        """
        Memory-light checkpointing:
        - Always overwrite 'last.pth' (latest state).
        - If is_best=True, also overwrite 'best.pth'.
        - On final save, overwrite 'final.pth'.
        No epoch-tagged files, so disk stays small.
        """
        payload = {
            "epoch": epoch,
            "model_state_dict": self.denoiser.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "img_shape": self.args.image_shape,
            "best_val": getattr(self, "best_val", None),
            "best_epoch": getattr(self, "best_epoch", None),
            "t_max": self.args.t_max,
            "alpha_min": self.args.alpha_min,
            "alpha_max": self.args.alpha_max,
            "alpha_interp": self.args.alpha_interp,
        }
        if epoch == "final":
            path = os.path.join(self.checkpoint_dir, "final.pth")
        else:
            path = os.path.join(self.checkpoint_dir, "last.pth")  # single rolling file
        torch.save(payload, path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pth")
            torch.save(payload, best_path)
        if self.verbose:
            msg = f"[Checkpoint] saved → {path}"
            if is_best:
                msg += " | best → best.pth"
            print(msg)


def main():
    from utils.parsing.args import args
    print("Starting training...")
    trainer = Trainer(args)
    trainer.train()
    print("Training completed!")

if __name__ == "__main__":
    main()