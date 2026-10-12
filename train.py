import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import Generator, Discriminator
from data import CustomImageDataset, SyntheticShapeDataset
from utils import (
    EMA,
    compute_r1_gradient_penalty,
    d_logistic_loss,
    g_logistic_loss,
    save_sample_grid,
)


def parse_args():
    parser = argparse.ArgumentParser(description="StyleGAN Training Pipeline")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to image dataset directory")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset for quick testing")
    parser.add_argument("--image_size", type=int, default=64, choices=[32, 64, 128], help="Generated image resolution")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of latent code z and style w")
    parser.add_argument("--r1_gamma", type=float, default=10.0, help="Weight of R1 gradient penalty")
    parser.add_argument("--r1_interval", type=int, default=16, help="Interval for lazy R1 regularization")
    parser.add_argument("--sample_interval", type=int, default=100, help="Batch interval to save image samples")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--samples_dir", type=str, default="samples", help="Directory to save generated samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device(args.device)
    print(f"[*] Training StyleGAN on device: {device} | Resolution: {args.image_size}x{args.image_size}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)

    # Initialize Dataset and DataLoader
    if args.synthetic or args.data_dir is None:
        print("[*] Using SyntheticShapeDataset (procedural geometric shapes)")
        dataset = SyntheticShapeDataset(num_samples=1000, image_size=args.image_size)
    else:
        print(f"[*] Loading CustomImageDataset from: {args.data_dir}")
        dataset = CustomImageDataset(root_dir=args.data_dir, image_size=args.image_size)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Instantiate Models
    generator = Generator(image_size=args.image_size, latent_dim=args.latent_dim, style_dim=args.latent_dim).to(device)
    discriminator = Discriminator(image_size=args.image_size).to(device)
    ema_generator = EMA(generator, decay=0.999)

    # Optimizers
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    # Fixed latent vectors for periodic evaluation visualization
    fixed_z = torch.randn(16, args.latent_dim, device=device)

    step = 0
    for epoch in range(1, args.epochs + 1):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for real_imgs in progress_bar:
            step += 1
            real_imgs = real_imgs.to(device)
            current_batch_size = real_imgs.size(0)

            # ========================================================
            # 1. Train Discriminator
            # ========================================================
            opt_d.zero_grad()
            z = torch.randn(current_batch_size, args.latent_dim, device=device)
            with torch.no_grad():
                fake_imgs = generator(z)

            real_scores = discriminator(real_imgs)
            fake_scores = discriminator(fake_imgs.detach())

            d_loss = d_logistic_loss(real_scores, fake_scores)

            # Lazy R1 Regularization
            if step % args.r1_interval == 0:
                r1_loss = compute_r1_gradient_penalty(discriminator, real_imgs)
                # Scale penalty by interval
                d_loss = d_loss + (args.r1_gamma / 2.0) * r1_loss * args.r1_interval
                r1_val = r1_loss.item()
            else:
                r1_val = 0.0

            d_loss.backward()
            opt_d.step()

            # ========================================================
            # 2. Train Generator
            # ========================================================
            opt_g.zero_grad()
            z = torch.randn(current_batch_size, args.latent_dim, device=device)
            gen_imgs = generator(z)
            gen_scores = discriminator(gen_imgs)

            g_loss = g_logistic_loss(gen_scores)
            g_loss.backward()
            opt_g.step()

            # Update Generator EMA
            ema_generator.update(generator)

            # Update progress bar
            progress_bar.set_postfix({
                "Loss_D": f"{d_loss.item():.4f}",
                "Loss_G": f"{g_loss.item():.4f}",
                "R1": f"{r1_val:.4f}",
            })

            # Save sample grid periodically
            if step % args.sample_interval == 0:
                with torch.no_grad():
                    sample_imgs = ema_generator.shadow.generate(fixed_z, truncation_psi=0.7)
                    sample_path = os.path.join(args.samples_dir, f"sample_step_{step:06d}.png")
                    save_sample_grid(sample_imgs, sample_path, nrow=4)

        # Save checkpoint at end of epoch
        ckpt_path = os.path.join(args.checkpoint_dir, f"stylegan_epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "step": step,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "ema_generator_state_dict": ema_generator.shadow.state_dict(),
            "opt_g_state_dict": opt_g.state_dict(),
            "opt_d_state_dict": opt_d.state_dict(),
        }, ckpt_path)
        print(f"[+] Saved checkpoint to {ckpt_path}")

    print("[*] Training completed successfully!")


if __name__ == "__main__":
    train()
