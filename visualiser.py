import matplotlib.pyplot as plt
import PIL
import numpy as np
import torch
from PIL import Image

class Visualiser : 

    def __init__(self, args, DDPM):

        self.fps = args.fps
        self.output_dir = args.output_dir
        self.visualise_every_n_iters = args.visualise_every_n_iters



    def save_gif(self, frames, path, fps=5):
        """
        Save a list of torch tensors as a GIF.
        
        Args:
            frames: List of torch tensors of size [Channels, Height, Width]
            path: Output path for the GIF file
            fps: Frames per second for the GIF
        """

        
        num_channels = frames[0].shape[0]
        if num_channels not in [1, 3]:
            raise ValueError(f"Unsupported number of channels: {num_channels}")
        
        is_grayscale = (num_channels == 1)
        
        pil_frames = []
        
        for frame in frames:

            frame_np = frame.cpu().numpy()
            
            if is_grayscale:  
                # Remove channel dimension and convert to uint8
                frame_np = frame_np.squeeze(0)
                # Ensure values are in range [0, 255]
                if frame_np.max() <= 1.0:
                    frame_np = (frame_np * 255).astype('uint8')
                else:
                    frame_np = frame_np.astype('uint8')
                pil_frame = Image.fromarray(frame_np, mode='L')
                
            else:
                # Transpose from [C, H, W] to [H, W, C]
                frame_np = frame_np.transpose(1, 2, 0)
                # Ensure values are in range [0, 255]
                if frame_np.max() <= 1.0:
                    frame_np = (frame_np * 255).astype('uint8')
                else:
                    frame_np = frame_np.astype('uint8')
                pil_frame = Image.fromarray(frame_np, mode='RGB')
            
            pil_frames.append(pil_frame)
        
        # Calculate duration in milliseconds
        duration = int(1000 / fps)
        
        # Save as GIF
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0  # 0 means infinite loop
        )



                   