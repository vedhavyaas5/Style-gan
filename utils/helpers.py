import copy
import torch
import torch.nn.functional as F
from torchvision.utils import save_image


class EMA:
    """
    Exponential Moving Average of model parameters.
    Keeps smoothed weights for stable and high-quality generation.
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)


def compute_r1_gradient_penalty(discriminator: torch.nn.Module, real_images: torch.Tensor) -> torch.Tensor:
    """
    Computes R1 gradient penalty on real images:
    R1 = 0.5 * || grad(D(x_real)) ||^2
    """
    real_images = real_images.detach().requires_grad_(True)
    real_scores = discriminator(real_images)

    # Compute gradients w.r.t real images
    grad_outputs = torch.ones_like(real_scores)
    gradients = torch.autograd.grad(
        outputs=real_scores,
        inputs=real_images,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Flatten and compute L2 norm squared
    grad_penalty = gradients.square().sum(dim=[1, 2, 3]).mean() * 0.5
    return grad_penalty


def d_logistic_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """Non-saturating logistic loss for Discriminator."""
    loss_real = F.softplus(-real_scores).mean()
    loss_fake = F.softplus(fake_scores).mean()
    return loss_real + loss_fake


def g_logistic_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """Non-saturating logistic loss for Generator."""
    return F.softplus(-fake_scores).mean()


def save_sample_grid(images: torch.Tensor, filepath: str, nrow: int = 4):
    """Saves a batch of [-1, 1] images to a grid PNG file."""
    save_image(images, filepath, nrow=nrow, normalize=True, value_range=(-1, 1))


def slerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """
    Spherical linear interpolation between two latent vectors.
    """
    omega = torch.acos((low / torch.norm(low, dim=-1, keepdim=True) * 
                        high / torch.norm(high, dim=-1, keepdim=True)).sum(dim=-1, keepdim=True).clamp(-1, 1))
    so = torch.sin(omega)
    if so.abs().item() < 1e-6:
        return (1.0 - val) * low + val * high
    return torch.sin((1.0 - val) * omega) / so * low + torch.sin(val * omega) / so * high
