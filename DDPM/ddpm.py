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
        self.alphas_bar = DDPM.compute_alphas_bar(self.alphas) # torch tensor
        self.num_trials = args.num_trials
        self.t_max = args.t_max
        
    @staticmethod
    def compute_alphas_bar(alphas):
        return torch.cumprod(alphas, dim=0)
    
    def blurData(self, x, t_tensor):
        """
        x: [B*num_trials, C, H, W]
        t_tensor: [B*num_trials] (long), timesteps in [0, T-1]
        """
        device = x.device
        t_tensor = t_tensor.to(device=device, dtype=torch.long)
 
        alpha_bar_t = self.alphas_bar.index_select(0, t_tensor)
        alpha_bar_t = alpha_bar_t.view(-1, *([1] * (x.ndim - 1)))

        white_noise = torch.randn_like(x)
        x_noisy = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * white_noise
        return x_noisy, white_noise


    def computeLoss(self, x):
        B = x.size(0)
        t = torch.randint(0, self.t_max, (B * self.num_trials,), device=x.device)
        x_rep = x.repeat_interleave(self.num_trials, dim=0)  # [B*num_trials, C, H, W]
        x_blurred, white_noise = self.blurData(x_rep, t)
        preds = self.denoiser(x_blurred, t)
        loss = F.mse_loss(preds, white_noise, reduction='mean')
        return loss
    
    
    @torch.no_grad()
    def generate(self, n_samples, device=None):
        """
        DDPM reverse process (ε-prediction).
        Returns x0 samples: [n_samples, C, H, W]
        """
        device = device or self.alphas_bar.device
        dtype = self.alphas_bar.dtype

        beta_t  = 1.0 - self.alphas
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alphas_bar)
        inv_sqrt_alpha = torch.rsqrt(self.alphas)

        x_t = torch.randn(n_samples, 
                          self.dataset.C, 
                          self.dataset.H, 
                          self.dataset.W, 
                          device=device, 
                          dtype=dtype)

        for t in reversed(range(self.T)):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)

            # (eq 11)
            eps_theta = self.denoiser(x_t, t_batch)
            c1 = inv_sqrt_alpha[t]
            c2 = beta_t[t] / (sqrt_one_minus_alpha_bar[t] + 1e-12)
            mean = c1 * (x_t - c2 * eps_theta)

            if t > 0:
                sigma = torch.sqrt(beta_t[t])
                z = torch.randn_like(x_t)
                x_t = mean + sigma * z
            else:
                x_t = mean

        return x_t



def main(args):
    pass

if __name__ == '__main__':
    main(args)
