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

    def __init__(self, args):
        self.ddpm = DDPM(args)
        self.fps = args.fps
        self.output_dir = args.output_dir
        self.dataset = args.dataset
        self.t_max = args.t_max
        self.generate_gifs = args.generate_gifs
        self.viz_noising = getattr(args, "visualize_noising", False)
        self.main()


    def get_random_image(self):
        idx = random.randint(0, len(self.dataset) - 1)
        sample = self.dataset[idx]

        if isinstance(sample, tuple):
            img, *_ = sample
            return img
        return sample


    def save_gif(self, frames: List[torch.Tensor], path: str) -> None:
        pil_frames = []
        for f in frames:
            # ensure CPU tensor for numpy conversion
            f = f.detach().cpu()

            # map [-1, 1] -> [0, 1] if needed
            if float(f.min()) < 0:
                f = 0.5 * f + 0.5

            f = f.clamp(0, 1)
            C, _, _ = f.shape
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
    

    def visualize_denoising_from_t(self, t: int) -> None:
        pass


    def main(self):
        if self.viz_noising:
            self.visualize_noising()
        if self.generate_gifs != 0:
            self.generate_gif(self.generate_gifs)


if __name__ == "__main__":
    from utils.parsing.inference.argparser_inference import p as inference_parser
    from utils.parsing.inference.args_inference import InferenceArgs
    parsed = inference_parser.parse_args()
    viz = Visualizer(InferenceArgs(parsed))
