"""
StyleGAN Self-Contained Demo & Architecture Verification
Runs a fast dry run on CPU or GPU to verify all modules and pipelines.
"""
import torch
from models import Generator, Discriminator
from utils import compute_r1_gradient_penalty, d_logistic_loss, g_logistic_loss, EMA


def run_demo():
    print("=" * 60)
    print("        StyleGAN Architecture Verification & Demo           ")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")

    image_size = 64
    latent_dim = 256
    batch_size = 4

    print(f"[*] Instantiating Generator (res={image_size}x{image_size}, latent={latent_dim})...")
    generator = Generator(image_size=image_size, latent_dim=latent_dim, style_dim=latent_dim, max_channels=256).to(device)
    num_params_g = sum(p.numel() for p in generator.parameters())
    print(f"[+] Generator parameters: {num_params_g:,}")

    print(f"[*] Instantiating Discriminator (res={image_size}x{image_size})...")
    discriminator = Discriminator(image_size=image_size, max_channels=256).to(device)
    num_params_d = sum(p.numel() for p in discriminator.parameters())
    print(f"[+] Discriminator parameters: {num_params_d:,}")

    # 1. Forward Pass Test
    print("\n[*] Testing Generator forward pass with random latents...")
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_imgs = generator(z)
    print(f"[+] Generated images shape: {fake_imgs.shape} (Expected: [{batch_size}, 3, {image_size}, {image_size}])")
    assert fake_imgs.shape == (batch_size, 3, image_size, image_size), "Shape mismatch!"

    print("[*] Testing Discriminator score calculation...")
    scores = discriminator(fake_imgs)
    print(f"[+] Discriminator output shape: {scores.shape} (Expected: [{batch_size}, 1])")
    assert scores.shape == (batch_size, 1), "Score shape mismatch!"

    # 2. Gradient and Loss Verification
    print("\n[*] Testing Loss computation & Backpropagation...")
    real_imgs = torch.randn(batch_size, 3, image_size, image_size, device=device)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-3)

    # D step
    opt_d.zero_grad()
    real_scores = discriminator(real_imgs)
    fake_scores = discriminator(fake_imgs.detach())
    d_loss = d_logistic_loss(real_scores, fake_scores)
    r1_penalty = compute_r1_gradient_penalty(discriminator, real_imgs)
    total_d_loss = d_loss + 5.0 * r1_penalty
    total_d_loss.backward()
    opt_d.step()
    print(f"[+] D loss: {d_loss.item():.4f} | R1 penalty: {r1_penalty.item():.4f}")

    # G step
    opt_g.zero_grad()
    gen_scores = discriminator(generator(z))
    g_loss = g_logistic_loss(gen_scores)
    g_loss.backward()
    opt_g.step()
    print(f"[+] G loss: {g_loss.item():.4f}")

    # 3. EMA & Truncation Trick Test
    print("\n[*] Testing EMA & Truncation trick...")
    ema = EMA(generator, decay=0.99)
    ema.update(generator)
    truncated_imgs = ema.shadow.generate(z, truncation_psi=0.5)
    print(f"[+] Truncation inference sample shape: {truncated_imgs.shape}")

    print("\n" + "=" * 60)
    print("[SUCCESS] All StyleGAN modules and pipelines verified successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
