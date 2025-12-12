import math
import typing as tp
import os
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from copy import deepcopy

class TimeEmbedding(nn.Module):
    def __init__(self, config:tp.Any, dim:int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LazyLinear(config.d_model),
            nn.SELU(),
            nn.LazyLinear(config.d_model),
        )
        self.half_dim = dim // 2

    def _init_weights(self, module:nn.Module):
        if isinstance(module, nn.LazyLinear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def sinusoidal_embeddings(self, x:Tensor):
        """https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py"""
        embeddings = torch.exp(-math.log(10000) * torch.arange(0, self.half_dim, device=x.device) / self.half_dim)
        embeddings = x[:, None] * embeddings[None]
        # in link implementation, concat(cos, sin, -1) is done
        embeddings = torch.concatenate([torch.cos(embeddings), torch.sin(embeddings)], -1)
        return embeddings

    def forward(self, x:Tensor): # (B,)
        x = self.sinusoidal_embeddings(x) # (B,) => (B, dim)
        x = self.mlp(x) # (B, d_model)
        return x # (B, d_model)
    

class PositionalEmbedding:
    def __init__(self, maxlen:int, dim:int):
        p, i = torch.meshgrid(torch.arange(float(maxlen)), torch.arange(dim/2)*2, indexing="xy")
        theta = (p/1e4**(i/dim)).T

        self.pos_emb = torch.stack([torch.sin(theta), torch.cos(theta)], axis=-1)
        self.pos_emb = self.pos_emb.reshape((maxlen, dim))[None] # (1, maxlen, dim)

    def sinusoidal_embeddings(self):
        return self.pos_emb # (1, maxlen, dim)
    

class Attention(nn.Module):
    def __init__(self, config:tp.Any):
        super().__init__()
        self.wq = nn.LazyLinear(config.d_model) # commented a warning in the source code for Lazy Modules
        self.wk = nn.LazyLinear(config.d_model)
        self.wv = nn.LazyLinear(config.d_model)

        self.w = nn.Linear(config.d_model, config.d_model)
        self.w.RESIDUAL_CONNECTION_SPECIAL_INIT = config.num_layers**-0.5

        self.num_heads = config.num_heads
        self.hdim = config.d_model // config.num_heads
        self.dropout_rate = config.dropout_rate

    def _init_weights(self, module:nn.Module):
        if isinstance(module, nn.LazyLinear):
            if hasattr(module, "RESIDUAL_CONNECTION_SPECIAL_INIT"):
                nn.init.xavier_uniform_(module.weight)
                module.weight.data *= module.RESIDUAL_CONNECTION_SPECIAL_INIT
            else:
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x:Tensor):
        T = x.shape[1]
        q, k, v = self.wq(x), self.wk(x), self.wv(x) # (B, T, d_model), (B, N, d_model), (B, N, d_model)

        q = q.view(-1, T, self.num_heads, self.hdim).transpose(1, 2) # (B, num_heads, T, hdim)
        k = k.view(-1, T, self.num_heads, self.hdim).transpose(1, 2) # (B, num_heads, N, hdim)
        v = v.view(-1, T, self.num_heads, self.hdim).transpose(1, 2) # (B, num_heads, N, hdim)

        # Flash Attn Shapes: q: (B, ..., T, hdim); k: (B, ..., N, hdim); v: (B, ..., N, hdim) => (B, ..., T, hdim)
        att_out = F.scaled_dot_product_attention(
            query=q, key=k, value=v,
            attn_mask=None,
            dropout_p=self.dropout_rate if self.training else 0.0
        ) # (B, num_heads, T, hdim)
        att_out = att_out.transpose(1, 2).contiguous().view(-1, T, self.hdim*self.num_heads) # (B, num_heads, T, hdim) => (B, T, d_model)
        
        linear_att_out = self.w(att_out) # (B, T, d_model)
        return linear_att_out
    

class FFN(nn.Module):
    """https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py#L13"""
    def __init__(self, config:tp.Any):
        super().__init__()
        hidden_dim = config.d_model*4
        self.layers = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.LazyLinear(config.d_model),
            nn.Dropout(config.dropout_rate)
        )

    def _init_weights(self, module:nn.Module):
        if isinstance(module, nn.LazyLinear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x:Tensor):
        return self.layers(x)
    

class PatchOps(nn.Module):
    """See https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L112"""
    def __init__(self, config:tp.Any):
        super().__init__()
        self.p = config.patch_size
        self.patch_proj = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True
        )
        w = self.patch_proj.weight.data
        nn.init.xavier_uniform_(
            w.view([w.shape[0], -1]) # init like linear layer
        )
        self.H, self.W = config.H, config.W
        self.c = config.out_channels

    def patchify(self, images:Tensor):
        B, C, H, W = images.shape
        patches = self.patch_proj(images) # (B, dim, H//P, W//P)
        patches = patches.flatten(2).transpose(1, 2)
        return patches # (B, N = (H*W)/P**2, embed_dim or dim)
    
    def unpacthify(self, x:Tensor): # (B, N = H*W//P**2, D = (P**2)*C)
        h, w = self.H//self.p, self.W//self.p # int(x.shape[1]**0.5) # h = H//P
        x = x.reshape(-1, h, w, self.p, self.p, self.c) # (B, C, H//P, W//P, P, P)
        x = torch.einsum('bhwpqc->bchpwq', x) # (B, C, H//P, P, W//P, P)
        x = x.reshape(-1, self.c, h*self.p, w*self.p) # (B, C, H, W)
        return x
    
    def forward(self, images:Tensor):
        return self.patchify(images)
    

class LabelEmbedding(nn.Module):
    def __init__(self, config:tp.Any):
        super().__init__()
        self.use_cfg = config.cfg_dropout_rate > 0
        self.label_embed = nn.Embedding(
            num_embeddings=config.num_classes + int(self.use_cfg),
            embedding_dim=config.d_model,
        )
        nn.init.normal_(self.label_embed.weight, mean=0.0, std=0.02)
        self.cfg_dropout_rate = config.cfg_dropout_rate
        self.num_classes = config.num_classes
    
    def drop_labels(self, labels:Tensor):
        ids_to_drop = torch.rand(size=(labels.shape[0],), device=labels.device) < self.cfg_dropout_rate
        labels = torch.where(ids_to_drop, self.num_classes, labels)
        return labels

    def forward(self, labels:Tensor):
        if self.use_cfg and self.training:
            labels = self.drop_labels(labels)
        return self.label_embed(labels)
    

def shift_scale(x:Tensor, gamma:Tensor, beta:Tensor):
    """ https://github.com/facebookresearch/DiT/blob/main/models.py#L19
    ```
    def modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    ```
    why +1 while scaling?
    """
    return x*(1 + gamma) + beta


class Block(nn.Module):
    def __init__(self, config:tp.Any):
        super().__init__()
        self.scale_shift_scale = nn.Sequential(
            nn.SiLU(),
            nn.LazyLinear(config.d_model*6),
        )
        self.norm1 = nn.LayerNorm(config.d_model, eps=1e-6, elementwise_affine=False)
        self.att = Attention(config=config)

        self.norm2 = nn.LayerNorm(config.d_model, eps=1e-6, elementwise_affine=False)
        self.pointwise_ffn = FFN(config)

    def _init_weights(self, module:nn.Module):
        m = self.scale_shift_scale[-1]
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)
        self.pointwise_ffn.apply(self.pointwise_ffn._init_weights)

    def forward(self, x:Tensor, cond:Tensor): # x: (B, T, d_model), cond: (B, d_model)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = torch.chunk(self.scale_shift_scale(cond)[:, None], 6, dim=-1) # 6*(B, 1, d_model) <= (B, 1, 6*d_model) <= (B, d_model)
        
        x = x + alpha1*self.att(shift_scale(self.norm1(x), gamma1, beta1)) # (B, T, d_model)
        x = x + alpha2*self.pointwise_ffn(shift_scale(self.norm2(x), gamma2, beta2)) # (B, T, d_model)
        return x # (B, T, d_model)
    
    # def forward(self, x: Tensor, cond:Tensor) -> Tensor:  # x: (B, T, d_model)
    #     gamma1, beta1, alpha1, gamma2, beta2, alpha2 = torch.chunk(self.scale_shift_scale(cond)[:, None], 6, dim=-1) # 6*(B, 1, d_model) <= (B, 1, 6*d_model) <= (B, d_model)
        
    #     y = x + alpha1*self.att(shift_scale(self.norm1(x), gamma1, beta1)) # (B, T, d_model)
    #     y = x + alpha2*self.pointwise_ffn(shift_scale(self.norm2(x), gamma2, beta2)) # (B, T, d_model)
        
    #     x = x + self.att(self.norm1(x))        
    #     x = x + self.pointwise_ffn(self.norm2(x))
    #     return x # (B, T, d_model)
    

class NormLinear(nn.Module):
    def __init__(self, config:tp.Any):
        super().__init__()
        self.shift_scale = nn.Sequential(
            nn.SELU(),
            nn.LazyLinear(config.d_model*2),
        )
        self.norm = nn.LayerNorm(config.d_model, eps=1e-6, elementwise_affine=False)
        self.linear = nn.LazyLinear((config.patch_size**2)*config.out_channels)

    def _init_weights(self, module:nn.Module):
        m = self.shift_scale[-1]
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x:Tensor, c:Tensor): # (B, T, D), # (B, D)
        gamma, beta = torch.chunk(self.shift_scale(c)[:, None], 2, dim=-1) # (B, 1, D) <= (B, 1, 2*D) <= (B, D)
        x = shift_scale(self.norm(x), gamma, beta)
        x = self.linear(x)
        return x # (B, T, (P**2)*C)
    
    # def forward(self, x:Tensor, c:Tensor): # (B, T, D), # (B, D)
    #     gamma, beta = torch.chunk(self.shift_scale(c)[:, None], 2, dim=-1) # (B, 1, D) <= (B, 1, 2*D) <= (B, D)
    #     y = shift_scale(self.norm(x), gamma, beta)
        
    #     x = self.norm(x)
    #     x = self.linear(x)
    #     return x # (B, T, (P**2)*C)
    

class DiT(nn.Module):
    def __init__(self, config:tp.Any):
        super().__init__()
        self.patch_ops = PatchOps(config)
        self.register_buffer(
            "positional_embeddings", 
            PositionalEmbedding(config.maxlen, config.d_model).sinusoidal_embeddings()
        ) # (1, maxlen, d_model)
        self.time_embed = TimeEmbedding(config, dim=config.d_model//2)
        self.label_embed = LabelEmbedding(config)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.norm_linear = NormLinear(config)
        
    def initialize_weights(self, module):
        self.time_embed.apply(self.time_embed._init_weights)
        for block in self.blocks:
            block.apply(block._init_weights)
        self.norm_linear.apply(self.norm_linear._init_weights)

    def forward(
        self,
        x:Tensor, # (B, C = 3 or 1, H, W)
        t:Tensor, # (B,)
        y:tp.Optional[Tensor], # (B,)
    ):
        x = self.patch_ops(x) + self.positional_embeddings  # (B, N = H*W//P**2, d_model)

        cond = self.time_embed(t) # (B, d_model)
        # if y is not None:
        #     cond += self.label_embed(y) # (B, d_model)
        
        for block in self.blocks:
            x = block(x, cond) # (B, N, d_model)
        
        x = self.norm_linear(x, cond) # (B, N = H*W//P**2, D = (P**2)*C)
        x = self.patch_ops.unpacthify(x) # (B, H, W, C)
        return x
    
    
    def forward_until_layer(
        self,
        x: Tensor,   # (B, C, H, W)
        t: Tensor,   # (B,)
        y: tp.Optional[Tensor],  # (B,)
        num_blocks: int          # how many transformer blocks to run
    ) -> Tensor:
        """
        Run the DiT up to (and including) num_blocks blocks and return token features.

        num_blocks = 0  -> just patch + pos emb
        num_blocks = 1  -> after first Block
        ...
        num_blocks = len(self.blocks) -> after last Block
        """
        assert 0 <= num_blocks <= len(self.blocks), \
            f"num_blocks must be in [0, {len(self.blocks)}], got {num_blocks}"

        x = self.patch_ops(x) + self.positional_embeddings  # (B, N, d_model)

        cond = self.time_embed(t)  # (B, d_model)
        
        # if y is not None:
        #     cond += self.label_embed(y)

        for i, block in enumerate(self.blocks):
            x = block(x, cond)
            if i + 1 == num_blocks:
                break

        return x  # (B, N, d_model)
    
    # TODO
    @torch.inference_mode()
    def cfg_forward(
        self, 
        x:Tensor, 
        t:Tensor,
        y:Tensor,
        cfg_num:float=4.0
    ):
        raise NotImplementedError("Not Implemented")

    
    def configure_optimizers(
        self,
        weight_decay:float,
        learning_rate:float,
        betas:tuple[float, float],
        device_type:str
    ):
        import inspect
        params_dict = {pname:p for pname, p in self.named_parameters() if p.requires_grad}

        # all weights except layernorms and biases, embeddings and linears
        decay_params = [p for pname, p in params_dict.items() if p.dim() >= 2]
        # layernorms and biases
        non_decay_params = [p for pname, p in params_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": non_decay_params, "weight_decay": 0.}
        ]

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        # other_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            params=optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=False # getting error with fused=True
        )
        return optimizer
    

def get_diffution_transformer(
        config:tp.Any, 
        input_shape:tuple[int, int, int, int],
        compile:bool
    ):
    x = torch.randn(input_shape).float() # (B, C, H, W)
    t = torch.randint(0, config.num_timesteps, size=(input_shape[0],)) # (B,)
    y = torch.randint(0, config.num_classes, size=(input_shape[0],)) # (B,)

    model = DiT(config)
    out:Tensor = model(x, t, y)
    assert out.shape == input_shape, f"Expected {input_shape} but got {out.shape}"

    model.apply(model.initialize_weights)
    with torch.no_grad():
        out:Tensor = model(x, t, y)
    assert (out == 0.0).all()

    if compile:
        model.compile()
    return model


class RepProjectionHead(nn.Module):
    """
    j_psi: two-layer MLP with SiLU, operating token-wise on (B, N, D).
    """
    def __init__(self, d_model: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, N, D)
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)        # <-- instead of view
        x_proj = self.mlp(x_flat)
        return x_proj.reshape(B, N, D)      # <-- instead of view


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, decay: float):
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data.mul_(decay).add_(p_s.data, alpha=1.0 - decay)
            

# Test run on MNIST
if __name__ == "__main__":
    import random
    import time
    import matplotlib.pyplot as plt; plt.style.use('dark_background')  

    from torchvision import datasets, transforms
    import torch.utils.data

    from diffusion_utils import DiffusionUtils
    from config import MNIST_config as config
    from utils import CosineDecayWithWarmup

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = get_diffution_transformer(
        config=config,
        input_shape=(config.batch_size, config.in_channels, config.H, config.W),
        compile=True
    ).to(config.device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "Million Parameters\n")

    transform = transforms.Compose([
        transforms.ToTensor(), # (H, W, C)/(H, W) -> (C, H, W) AND [0, 255] -> [0.0, 1.0]
        transforms.Normalize(mean=(0.5,), std=(0.5,), inplace=True) # [0.0, 1.0] -> [-1.0, 1.0]
    ])
    
    rep_head = RepProjectionHead(config.d_model).to(config.device)

    ema_decay = 0.9999
    ema_model = get_diffution_transformer(
        config=config,
        input_shape=(config.batch_size, config.in_channels, config.H, config.W),
        compile=False,            # EMA doesn't need to be compiled
    ).to(config.device)

    ema_model.load_state_dict(model.state_dict())  # copy weights from compiled student
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    valset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    class DataLoader:
        def __init__(self, ds):
            self.ds = ds
            
        def iter_batches(self, batch_size):
            while True:
                self.dataset = torch.utils.data.DataLoader(
                    dataset=self.ds,
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=True,
                    drop_last=True
                )
                for X_batch, y_batch in self.dataset:
                    yield X_batch.to(config.device), y_batch.to(config.device)

    
    get_lr = CosineDecayWithWarmup(
        warmup_steps=config.warmup_steps,
        max_learning_rate=config.max_lr,
        decay_steps=config.decay_steps,
        min_learning_rate=config.min_lr
    ) if not config.no_decay else lambda _: config.max_lr

    train_iterator = iter(DataLoader(trainset).iter_batches(config.batch_size))
    val_iterator = iter(DataLoader(valset).iter_batches(config.batch_size))

    diff_utils = DiffusionUtils(config)

    ctx = torch.autocast(
            device_type=config.device.type,
            dtype={"bfloat16": torch.bfloat16,
                "float32" : torch.float32}[config.dtype_type]
        )

    @torch.no_grad()
    def evaluate(model:nn.Module):
        model.eval()
        gen_image = diff_utils.generate(model=model, labels=[random.randint(0, config.num_classes-1)]).detach().cpu().squeeze()
        model.train()
        print("max", gen_image.max(), "min", gen_image.min())
        return gen_image

    
    rep_loss_weight = 1e-3
    rep_loss_layer = 3  
    finetune = False
    checkpoint_path = "checkpoints/mnist_old_loss_new_weight.pt"
    
    if finetune:
        ckpt = torch.load(checkpoint_path, map_location=config.device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
        if missing:
            print("Missing keys:", missing)
        if unexpected:
            print("Unexpected keys:", unexpected)
        ema_model.load_state_dict(model.state_dict())
            
    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.max_lr,
        betas=(config.beta1, config.beta2),
        device_type=config.device.type
    )
    
    optimizer.add_param_group({
        "params": rep_head.parameters(),
        "weight_decay": 0.0,
    })
    
    def train(model:nn.Module, ema_model, rep_head):
        num_steps = 45_000
        start_rep_step = 10_000
        
        t0 = time.time()
        
        max_t = 400
        offset_low = 15
        offset_high = 30
        threshold = 370

        max_t = config.num_timesteps - 1
        losses = []
        
        for step in range(0, num_steps):
            use_rep_loss = False #step >= start_rep_step
                
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # if (step % config.eval_freq == 0 and step > 0) or step == config.num_steps-1:
            #     gen_image = evaluate(model)
            #     plt.imshow(gen_image.numpy(), cmap="gray")
            #     plt.show(block=False)
            #     plt.pause(2)
            #     plt.close()

            optimizer.zero_grad(set_to_none=True)
            X_batch, y_batch = next(train_iterator)

            timesteps = torch.randint(
                size=(X_batch.shape[0],), low=0, high=config.num_timesteps, device=config.device
            )
            
            #cont t drama
            
            # max_t = config.num_timesteps - 1
            # steps_per_level = 100             # every 100 steps, unlock one more timestep

            # # how many extra timesteps (besides max_t itself) have we unlocked so far
            # unlocked = step // steps_per_level               # 0, 1, 2, ...
            # unlocked = min(unlocked, max_t)                  # cap so we don't go past full range

            # # we move the lower bound downward from max_t to 0
            # t_low = max_t - unlocked                         # 400, 399, 398, ..., 0
            # t_high = max_t + 1                               # randint upper bound is exclusive

            # timesteps = torch.randint(
            #     size=(X_batch.shape[0],),
            #     low=t_low,
            #     high=t_high,                                  # [t_low, max_t] inclusive
            #     device=config.device,
            # )
            
            noisy_image_timestep, noise_true = diff_utils.noisy_it(X_batch, timesteps)
            z = noisy_image_timestep["noisy_images"]  # noisy at t

            # regularization t drama
            
            # low_bounds = torch.where(
            #     timesteps < threshold,
            #     timesteps + offset_low,
            #     timesteps
            # )
            # high_bounds = torch.where(
            #     timesteps < threshold,
            #     torch.clamp(timesteps + offset_high, max=max_t),
            #     torch.full_like(timesteps, max_t)
            # )
            
            # rand_u = torch.rand_like(timesteps, dtype=torch.float32)
            # low_f = low_bounds.to(torch.float32)
            # high_f = high_bounds.to(torch.float32)
            # higher_timesteps = (low_f + rand_u * (high_f - low_f + 1.0)).floor().to(torch.long)

            # noisy_image_high, _ = diff_utils.noisy_it(X_batch, higher_timesteps)
            # z_high = noisy_image_high["noisy_images"]  # noisy at t_high
            
            for _ in range(config.num_grad_accumalation_steps):
                with ctx:
                    rep_loss=0
                    emb_noisy_norm = 0.0
                    emb_more_noisy_norm = 0.0

                    B = X_batch.shape[0]
                    z = noisy_image_timestep["noisy_images"]
                    # denom = torch.sqrt(1 - noisy_image_timestep["alpha_bar_t"]).clamp(min=0.05)
                    x_pred = model(z, noisy_image_timestep["timesteps"], y_batch)
                    
                    # v_pred = (x_pred - z) / denom
                    # v = (X_batch - z) / denom
                    
                    per_pixel_loss = F.mse_loss(x_pred, X_batch, reduction='none')
                    per_example_loss = per_pixel_loss.view(B, -1).mean(dim=1)

                    alpha_bar_t = noisy_image_timestep["alpha_bar_t"]  
                    if alpha_bar_t.dim() > 1:
                        alpha_bar_t = alpha_bar_t.view(B)
                    denom = (1 - alpha_bar_t).clamp(min=0.05)
                    weight = alpha_bar_t / denom            # (B,)
                    weighted_loss = (weight * per_example_loss).mean()
                    
                    if use_rep_loss:
                        emb_more_noisy = ema_model.forward_until_layer(
                            z_high,
                            noisy_image_high['timesteps'],
                            y_batch,
                            num_blocks=rep_loss_layer,
                        )  # (B, N, d_model)

                        emb_noisy = model.forward_until_layer(
                            z,
                            noisy_image_timestep["timesteps"],
                            y_batch,
                            num_blocks=rep_loss_layer,
                        )  # (B, N, d_model)
                        
                        emb_noisy_proj = rep_head(emb_noisy) 

                        emb_more_noisy_detach = emb_more_noisy.detach()  # <- stopgrad here

                        rep_loss = F.mse_loss(emb_noisy_proj, emb_more_noisy_detach)

                        emb_noisy_norm = torch.linalg.vector_norm(
                            emb_noisy.detach().flatten(1), dim=1
                        ).mean()

                        emb_more_noisy_norm = torch.linalg.vector_norm(
                            emb_more_noisy.flatten(1), dim=1
                        ).mean()

                    total_step_loss = weighted_loss + rep_loss_weight * rep_loss
                    loss = total_step_loss / config.num_grad_accumalation_steps
                                                            
                loss.backward()
                
            if config.clipnorm is not None:
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clipnorm, error_if_nonfinite=True)
            optimizer.step()
            update_ema(model, ema_model, ema_decay)

            t1 = time.time()
            dt = t1-t0
            t0 = t1
            lossf = loss.cpu().item() * config.num_grad_accumalation_steps
            
            
            if step % config.log_interval == 0:
                print(
                    f"| Step: {step} || Loss: {lossf:.4f} |"
                    f"| LR: {lr:e} || dt: {dt*1000:.2f}ms |", end="")
                print(f"| Norm: {norm:.4f} |" if config.clipnorm is not None else "")
                print(f"| diffusion: {weighted_loss:.4f} |")
                print(f"| embedding: {rep_loss:.4f} |")
                print(f"| noisy norm: {emb_noisy_norm:.4f} |")
                print(f"| more noisy norm: {emb_more_noisy_norm:.4f} |")
            losses.append(lossf)
        return losses

    print("training")
    losses = train(model, ema_model, rep_head)
    
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rep_head": rep_head.state_dict(),
        },
        f"checkpoints/mnist_baseline_50_000.pt",
    )

    plt.plot(losses)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss vs Steps")
    plt.show(block=False)
    plt.pause(interval=5)
    plt.close()

    # print("Generating Images all digits from 0 to 9...")
    # for i in range(10):
    #     gen_image = diff_utils.generate(model=model, labels=[i]).detach()
    #     plt.imshow(gen_image.squeeze().cpu().numpy(), cmap="gray")
    #     plt.show(block=False)
    #     plt.pause(interval=5)
    #     plt.close()
    # print("Done. Succesfull")

