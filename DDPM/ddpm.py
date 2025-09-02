import torch
import torch.nn.functional as F
from ..utils.argparser import args

class DDPM:

    def __init__(self, args):

        # data
        self.dataset = args.dataset

        # model
        self.denoiser = args.denoiser
    
        # loss
        self.alphas = args.alphas # torch tensor
        self.num_trials = args.num_trials
        self.T_max = args.T_max

        
    def blurData(self, x, t_tensor):
        """
        x: [B*num_trials, C, H, W]
        t_tensor: [B*num_trials] (long), timesteps in [0, T-1]
        """
        device = x.device
        t_tensor = t_tensor.to(device=device, dtype=torch.long)
 
        alpha = self.alphas.index_select(0, t_tensor)
        alpha = alpha.view(-1, *([1] * (x.ndim - 1)))

        white_noise = torch.randn_like(x)
        x_noisy = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * white_noise
        return x_noisy, white_noise


    def computeLoss(self, x):
        B = x.size(0)
        t = torch.randint(0, self.T_max, (B * self.num_trials,), device=x.device)
        x_rep = x.repeat_interleave(self.num_trials, dim=0)  # [B*num_trials, C, H, W]
        x_blurred, white_noise = self.blurData(x_rep, t)
        preds = self.denoiser(x_blurred, t)
        loss = F.mse_loss(preds, white_noise, reduction='mean')
        return loss
    
    
    def generate(self, n_samples):
        #TODO
        pass



def main(args):
    pass

if __name__ == '__main__':
    main(args)
