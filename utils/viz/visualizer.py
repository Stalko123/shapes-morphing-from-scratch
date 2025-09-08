"""
Utilities for visualizing DDPM forward/reverse processes and saving them as GIFs.
"""

from PIL import Image
from DDPM.ddpm import DDPM
import numpy as np
from tqdm import tqdm
import torch
import random
from typing import List, Optional


class Visualizer:
    """Create and save GIF visualizations for DDPM noising/denoising.

    This helper wraps a `DDPM` instance and your training args to:
      - save tensor frames as an animated GIF
      - visualize forward diffusion (noising) on a random training image
      - (stub) visualize reverse diffusion (denoising) starting from an arbitrary timestep

    Attributes:
        args: Namespace-like object containing runtime configuration (e.g., dataloaders,
            `t_max`, `alphas_bar`, `fps`, `output_dir`, etc.).
        ddpm: `DDPM` wrapper instantiated from `args`.
        fps: Frame rate (frames per second) used when writing GIFs.
        output_dir: Directory path where GIFs are saved.
    """

    def __init__(self, args):
        """Initialize the visualizer and underlying DDPM helper.

        Args:
            args: Configuration object with at least:
                - dataloader_train: torch.utils.data.DataLoader for training data.
                - t_max: int, number of diffusion steps to visualize.
                - alphas_bar: 1D tensor of shape [t_max], cumulative noise schedule.
                - fps: int, target GIF frame rate.
                - output_dir: str, directory for saved GIFs.
                - (used by DDPM): whatever `DDPM(args)` expects.
        """
        self.args = args
        self.ddpm = DDPM(args)
        self.fps = args.fps
        self.output_dir = args.output_dir

    def save_gif(self, frames: List[torch.Tensor], path: str) -> None:
        """Save a list of image tensors as an animated GIF.

        Each frame should be a single image tensor with shape [C, H, W] and values
        either in [-1, 1] or [0, 1]. Channels must be 1 (grayscale) or 3 (RGB).

        The function:
          - rescales from [-1, 1] to [0, 1] if needed,
          - clamps to [0, 1],
          - converts to uint8,
          - writes an infinite-looping GIF with a per-frame duration derived from `fps`.

        Args:
            frames: List of image tensors of shape [C, H, W]. They may be on CPU or GPU.
            path: Output filepath for the GIF (e.g., "out/noising.gif").

        Raises:
            ValueError: If the number of channels is not 1 or 3.
        """
        pil_frames = []
        for f in frames:
            # ensure CPU tensor for numpy conversion
            f = f.detach().cpu()

            # map [-1, 1] -> [0, 1] if needed
            if float(f.min()) < 0:
                f = 0.5 * f + 0.5

            f = f.clamp(0, 1)
            C, H, W = f.shape
            arr = f.numpy()

            if C == 1:
                arr = (arr[0] * 255).astype(np.uint8)                 # [H, W]
                pil = Image.fromarray(arr)                            # 'L'
            elif C == 3:
                arr = (arr.transpose(1, 2, 0) * 255).astype(np.uint8) # [H, W, 3]
                pil = Image.fromarray(arr)                            # 'RGB'
            else:
                raise ValueError(f"Unsupported channels: {C}")

            pil_frames.append(pil)

        # GIF viewers often clamp minimum frame duration to ~20ms
        duration_ms = max(20, int(round(1000.0 / float(self.fps))))
        pil_frames[0].save(
            path,
            format="GIF",
            save_all=True,
            append_images=pil_frames[1:],
            duration=[duration_ms] * len(pil_frames),  # per-frame durations
            loop=0,                                    # 0 = infinite loop
            disposal=2,                                # replace each frame
            optimize=False,                            # avoid frame dropping
        )

    def generate_gif(self, n_gifs: int) -> None:
        """Sample from the DDPM and save each sample's intermediates as a GIF.

        Calls `self.ddpm.generate_one_sample(return_intermediates=True)` repeatedly
        to obtain a sequence of intermediate frames for each generated sample, then
        saves a GIF to `self.output_dir`.

        Args:
            n_gifs: Number of sample GIFs to generate.

        Side Effects:
            Writes files named `generated_sample_{i}.gif` to `self.output_dir`.
        """
        for i in tqdm(range(n_gifs)):
            frames = self.ddpm.generate_one_sample(return_intermediates=True)
            self.save_gif(frames, f"{self.output_dir}/generated_sample_{i}.gif")
        print(f"Saved the gifs to {self.output_dir}")


    def visualize_noising(self, save: bool = True, return_images: bool = False) -> Optional[List[torch.Tensor]]:
        """Visualize forward diffusion (x_t) on a random training image as a GIF.

        Picks a random image from the next batch of the training dataloader, then
        constructs frames:
            x_t = sqrt(alphas_bar[t]) * x_0 + sqrt(1 - alphas_bar[t]) * ε
        using a single fixed ε ~ N(0, I) so the noise progressively increases in the
        same direction—this yields a smooth "gradual destruction" visualization.

        Args:
            save: If True, writes the GIF to `{output_dir}/noising.gif`.
            return_images: If True, returns the list of frames (x_0, x_1, ..., x_T).

        Returns:
            frames: Optional list of tensors [C, H, W] for each timestep (including the
                clean image as the first frame). Returned only if `return_images=True`.
        """
        batch = next(iter(self.args.dataloader_train))
        images = batch[0]  # [B, C, H, W]
        idx = random.randint(0, images.size(0) - 1)
        image = images[idx]  # [C, H, W]

        frames: List[torch.Tensor] = [image]
        white_noise = torch.randn_like(image)

        for t in tqdm(range(self.args.t_max)):
            image_noisy = (
                torch.sqrt(self.args.alphas_bar[t]) * image
                + torch.sqrt(1 - self.args.alphas_bar[t]) * white_noise
            )
            frames.append(image_noisy)

        if save:
            self.save_gif(frames=frames, path=f"{self.output_dir}/noising.gif")
        if return_images:
            return frames
        return None

    def visualize_denoising_from_t(self, t: int) -> None:
        
        pass


def main():
    from utils.parsing.args import args
    viz = Visualizer(args)
    viz.visualize_noising()


if __name__ == "__main__":
    main()
