import numpy as np
from PIL import Image

class Visualizer : 

    def __init__(self, args):

        self.fps = args.fps
        self.output_dir = args.output_dir
        self.denoiser_weights = args.denoiser_weights


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

            frame_np = frame.detach().cpu().numpy()  # Use detach() to remove gradients
            
            if is_grayscale:  
                # Remove channel dimension and keep values in [0, 1] range
                frame_np = frame_np.squeeze(0)
                # Clip values to [0, 1] range and convert to uint8 [0, 255] for PIL
                frame_np = np.clip(frame_np, 0, 1)
                frame_np = (frame_np * 255).astype('uint8')
                pil_frame = Image.fromarray(frame_np, mode='L')
                
            else:
                # Transpose from [C, H, W] to [H, W, C]
                frame_np = frame_np.transpose(1, 2, 0)
                # Clip values to [0, 1] range and convert to uint8 [0, 255] for PIL
                frame_np = np.clip(frame_np, 0, 1)
                frame_np = (frame_np * 255).astype('uint8')
                pil_frame = Image.fromarray(frame_np, mode='RGB')
            
            pil_frames.append(pil_frame)
        
        duration = int(1000 / fps) #(ms)
        
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0  # 0 means infinite loop
        )

    def generate_gif(self, path, n_gifs):
        #load the denoiser weights, and visualise n_gifs random images from the dataset

        pass
