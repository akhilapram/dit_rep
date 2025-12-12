import torch
from torch import nn, Tensor
from tqdm import trange
from typing import Optional

class DiffusionUtils:
    def __init__(self, config):
        self.num_timesteps = config.num_timesteps # (nT,)
        self.beta_range = config.var_range
        self.device = config.device
        self.get_alpha()

        self.H, self.W = config.H, config.W
        self.in_channels = config.in_channels

    def get_alpha(self):
        self.beta = torch.linspace(start=self.beta_range[0], end=self.beta_range[1], steps=self.num_timesteps, device=self.device) # (nT,)
        self.alpha = (1-self.beta) # (nT,)
        self.alpha_bar = torch.concatenate(
            (torch.tensor([1.], device=self.device), self.alpha.cumprod(axis=0)),
            axis=0
        ) # (nT,)
    
    def noisy_it(self, X:Tensor, t:Tensor): # (B, H, W, C), (B,)
        noise = torch.normal(mean=0.0, std=1.0, size=X.shape, device=self.device) # (B,)

        alpha_bar_t = self.alpha_bar[t][:, None, None, None] # (B, 1, 1, 1) <= (B,) <= (nT,)
        return {
            "noisy_images": torch.sqrt(alpha_bar_t)*X + torch.sqrt(1 - alpha_bar_t) * noise,
            "timesteps": t,
            "alpha_bar_t": alpha_bar_t
        }, noise
    
    def one_step_ddpm(self, xt:Tensor, pred_noise:Tensor, t:int):
        alpha_t, alpha_bar_t = self.alpha[t, None, None, None], self.alpha_bar[t, None, None, None]
        z = torch.normal(mean=0.0, std=1.0, size=xt.shape, device=self.device) if t>0 else 0.0
        xt_minus_1 = (
            (1/torch.sqrt(alpha_t))
            *
            (xt - (1-alpha_t)*pred_noise/torch.sqrt(1-alpha_bar_t)
            ) + torch.sqrt(self.beta[t])*z
        )
        return xt_minus_1
    
    def one_step_ddim(self, xt:Tensor, pred_noise:Tensor, t:int) -> Tensor:
        raise NotImplementedError
    
    @torch.no_grad()
    def generate(
        self, *,
        model: nn.Module,
        labels: Optional[int] = None,
        use_ddim: bool = False,
    ):
        """
        Sampling when the model predicts the CLEAN IMAGE x0.
        We convert x0_pred -> eps_pred and reuse the DDPM step.
        """
        sample_func = self.one_step_ddim if use_ddim else self.one_step_ddpm

        print(f"Generating images", "" if labels is None else "of " + str(labels))
        labels = torch.tensor(labels, device=self.device) if labels is not None else None

        # start from pure noise
        x = torch.normal(
            mean=0.0,
            std=1.0,
            size=(1, self.in_channels, self.H, self.W),
            device=self.device,
        )

        # t goes from T-1 down to 1
        for i in trange(0, self.num_timesteps - 1):
            t_scalar = self.num_timesteps - i - 1            # int: T-1, T-2, ..., 1
            t = torch.tensor([t_scalar], device=self.device) # (1,)

            # model now predicts x0, not noise
            x0_pred = model(x, t, labels)  # (1, C, H, W)
            
            # get alpha_bar_t for this timestep
            alpha_bar_t = self.alpha_bar[t][:, None, None, None]  # (1,1,1,1)

            # convert x0_pred -> eps_pred using:
            # x_t = sqrt(a_bar_t) * x0 + sqrt(1 - a_bar_t) * eps  =>
            # eps = (x_t - sqrt(a_bar_t) * x0) / sqrt(1 - a_bar_t)
            eps_pred = (x - torch.sqrt(alpha_bar_t) * x0_pred) / torch.sqrt(1 - alpha_bar_t)

            # DDPM step using eps_pred
            x = sample_func(x, eps_pred, t_scalar)

        return x