

class DDPM:

    def __init__(self, args):

        # data
        self.dataset = args.dataset

        # model
        self.denoiser = args.denoiser
    
        # loss
        self.alphas = args.alphas
        self.num_trials = args.num_trials
        self.T_max = args.T_max
        
    def blurData(self, x, t):
        """
        Applies t blurring steps on the batch x. 
        """
        white_noise = torch.random.gaussian(0, self.dataset.dim)
        alpha = self.alphas[t]
        return torch.sqrt(alpha)*x + torch.sqrt(1 - alpha)*white_noise, white_noise


    def computeLoss(self, x):
        
        total = 0
        for _ in range(self.num_trials):
            t = torch.random.randint(0, self.T_max)
            x_blurred, white_noise = self.blurData(x, t)
            preds = self.denoiser(x_blurred)
            total += torch.nn.MSE(preds, white_noise)
        return total/self.num_trials
            


    def __call__(self, x):
        return self.denoiser(x)
    

    def generate(self):
        noise = torch.random(nfjkoi)
        return DDPM(noise)

