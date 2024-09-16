import torch
import torch.nn.functional as F

from tqdm import tqdm

from eps_standard import cf10_eps_standard, imagenet_eps_standard
from eps_calculation import reweight_t

class PGD:
    def __init__(self, get_logit, attack_steps=200, eps=8./255., step_size=0.007, target=None, eot=20):
        self.target = target
        self.clamp = (0,1)
        self.eps = eps
        self.step_size = step_size
        self.get_logit = get_logit
        self.attack_steps = attack_steps
        self.eot = eot

    def _random_init(self, x):
        x = x + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * self.eps
        x = torch.clamp(x, *self.clamp)
        return x

    def __call__(self, x, y):
        x_adv = self.forward(x, y)
        return x_adv
    
    def forward(self, x, y):
        x_adv = x.detach().clone()
        for _ in tqdm(range(self.attack_steps), desc="PGD Linf attack iters", leave=False):
            grad = torch.zeros_like(x_adv)
            for _ in range(self.eot):
                x_adv.requires_grad = True
                
                # Classification
                logits = self.get_logit(x_adv)
                
                # Calculate loss
                loss = F.cross_entropy(logits, y, reduction="sum")
                
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                x_adv = x_adv.detach()

            grad /= self.eot
            grad = grad.sign()
            x_adv = x_adv + self.step_size * grad

            # Projection
            x_adv = x + torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = x_adv.detach()
            x_adv = torch.clamp(x_adv, *self.clamp)

        return x_adv
    

class Whitebox_PGD:
    def __init__(self, diffusion, args, config, get_logit, attack_steps=200, eps=8./255., step_size=0.007, target=None, eot=20):
        self.target = target
        self.clamp = (0,1)
        self.eps = eps
        self.step_size = step_size
        self.get_logit = get_logit
        self.attack_steps = attack_steps
        self.eot = eot
        self.diffusion = diffusion
        self.args = args
        self.config = config

    def _random_init(self, x):
        x = x + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * self.eps
        x = torch.clamp(x, *self.clamp)
        return x

    def __call__(self, x, y):
        x_adv = self.forward(x, y)
        return x_adv
    
    def forward(self, x, y):
        x_adv = x.detach().clone()
        if self.args.dataset == "cifar10":
            eps_standard = cf10_eps_standard
        elif self.args.dataset == "imagenet":
            eps_standard = imagenet_eps_standard
        for _ in tqdm(range(self.attack_steps), desc="Whitebox PGD Linf attack iters", leave=False):
            grad = torch.zeros_like(x_adv)
            eps_range, adv_eps = reweight_t(x_adv, self.diffusion, eps_standard, self.args, self.config)


            for _ in range(self.eot):
                x_adv.requires_grad = True
                
                # Classification
                logits = self.get_logit(x_adv, eps_standard, adv_eps)
                # logits = self.get_logit(x_adv, eps_range, adv_eps)
                
                
                # Calculate loss
                loss = F.cross_entropy(logits, y, reduction="sum")
                
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                x_adv = x_adv.detach()

            grad /= self.eot
            grad = grad.sign()
            x_adv = x_adv + self.step_size * grad

            # Projection
            x_adv = x + torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = x_adv.detach()
            x_adv = torch.clamp(x_adv, *self.clamp)

        return x_adv