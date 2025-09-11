"""
Utilities for visualizing DDPM forward/reverse processes and saving them as GIFs.
"""

from PIL import Image
from DDPM.ddpm import DDPM
import numpy as np
from tqdm import tqdm
import torch
import random
from typing import List, Optional, Sequence, Union
import matplotlib.pyplot as plt


class Visualizer:

    def __init__(self, args):
        self.ddpm = DDPM(args)
        self.fps = args.fps
        self.output_dir = args.output_dir
        self.dataset = args.dataset
        self.t_max = args.t_max
        self.generate_gifs = args.generate_gifs
        self.viz_noising = getattr(args, "viz_noising", False)
        self.viz_denoising_from_t = getattr(args, "viz_denoising_from_t", 0)
        self.main()

    @staticmethod
    def to_vis(img: torch.Tensor) -> np.ndarray:
        # Converts to CPU [H,W,C] in [0,255]
        img = img.detach().float().cpu()
        if img.dim() == 4:
            img = img[0]
        if img.min() < 0:
            img = (img.clamp(-1, 1) + 1) / 2
        else:
            img = img.clamp(0, 1)
        if img.size(0) == 3:                  # CHW -> HWC
            img = img.permute(1, 2, 0)
        elif img.size(0) == 1:
            img = img[0] # HW
        img = img.numpy()
        img = (img * 255).astype(np.uint8)
        return img

    def get_random_image(self):
        idx = random.randint(0, len(self.dataset) - 1)
        sample = self.dataset[idx]

        if isinstance(sample, tuple):
            img, *_ = sample
            return img
        return sample


    def save_gif(self, frames: Sequence[torch.Tensor], path: str, scale: int = 10) -> None:
        pil_frames = []
        for f in frames:
            img = Image.fromarray(Visualizer.to_vis(f))
            img = img.resize(
                (img.width * scale, img.height * scale), 
                resample=Image.NEAREST
            )
            pil_frames.append(img)
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
        image = self.get_random_image()

        frames: List[torch.Tensor] = [image]
        white_noise = torch.randn_like(image)

        for t in tqdm(range(self.t_max)):
            image_noisy = (
                torch.sqrt(self.ddpm.alphas_bar[t]) * image
                + torch.sqrt(1 - self.ddpm.alphas_bar[t]) * white_noise
            )
            frames.append(image_noisy)

        if save:
            self.save_gif(frames=frames, path=f"{self.output_dir}/noising.gif")
        if return_images:
            return frames
        return None
    

    def visualize_denoising_from_t(self, t: Union[int, Sequence[int]]) -> None:
        """
        Loads a random image from the training set, applies t noising steps to it,
        runs the model to predict the noise, reconstructs x0, and shows the result.
        """
        if type(t) == int:
            t = [t]

        for time in t: 
            try:
                device = next(self.ddpm.denoiser.parameters()).device  # type: ignore[attr-defined]
            except Exception:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if time > self.t_max:
                print(f"Warning: time {time} is greater than t_max {self.t_max} and will be clamped.")
            if time < 0:
                print(f"Warning: time {time} is negative and will be clamped to 0.")
            time = max(0, min(int(time), int(self.t_max)))

            image = self.get_random_image()                 # tensor [C,H,W]
            x = image.unsqueeze(0).to(device)
            
            t_tensor = torch.full((x.size(0),), time, dtype=torch.long, device=device)

            self.ddpm.denoiser.eval()
            with torch.no_grad():
                x_noisy, _ = self.ddpm.blurData(x, t_tensor)  # both [1,C,H,W]
                eps_pred = self.ddpm.denoiser(x_noisy, t_tensor) # [1,C,H,W]
                alpha_bar_t = self.ddpm.alphas_bar[time].view(1, *([1] * (x.ndim - 1)))
                x0_hat = (x_noisy - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

            orig_vis = Visualizer.to_vis(x)
            noisy_vis = Visualizer.to_vis(x_noisy)
            denoised_vis = Visualizer.to_vis(x0_hat)

            fig, axes = plt.subplots(ncols=3, figsize=(15, 5), constrained_layout=True)

            for ax, (title, im) in zip(
                axes,
                [
                    ("Original", orig_vis),
                    (f"Noisy (t={time})", noisy_vis),
                    ("Denoised (x̂₀)", denoised_vis),
                ],
            ):
                ax.imshow(im.squeeze() if im.shape[-1] == 1 else im)
                ax.set_title(title)
                ax.axis("off")

            fig.savefig(f"{self.output_dir}/denoising_from_time{time}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            

    def main(self):
        if self.viz_noising:
            self.visualize_noising()
        if self.generate_gifs != 0:
            self.generate_gif(self.generate_gifs)
        if self.viz_denoising_from_t != 0:
            self.visualize_denoising_from_t(self.viz_denoising_from_t)


if __name__ == "__main__":
    from utils.parsing.inference.args_inference import args
    viz = Visualizer(args)
