import os
import argparse
import torch

from models import Generator
from utils import save_sample_grid, slerp


def parse_args():
    parser = argparse.ArgumentParser(description="StyleGAN Generation and Latent Manipulation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint .pt file")
    parser.add_argument("--mode", type=str, default="random", choices=["random", "interpolate", "style_mix"], help="Generation mode")
    parser.add_argument("--image_size", type=int, default=64, choices=[32, 64, 128], help="Image resolution")
    parser.add_argument("--latent_dim", type=int, default=512, help="Latent dimension")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of random samples to generate")
    parser.add_argument("--truncation_psi", type=float, default=0.7, help="Truncation factor psi (0.0 to 1.0)")
    parser.add_argument("--steps", type=int, default=10, help="Number of interpolation steps")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save output images")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    return parser.parse_args()


def load_generator(args, device):
    generator = Generator(image_size=args.image_size, latent_dim=args.latent_dim, style_dim=args.latent_dim).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get("ema_generator_state_dict", checkpoint.get("generator_state_dict", checkpoint))
        generator.load_state_dict(state_dict)
        print(f"[+] Loaded weights from {args.checkpoint}")
    else:
        print("[!] No checkpoint specified or file not found. Using initialized weights for demonstration.")
    generator.eval()
    return generator


def generate_random(generator, args, device):
    print(f"[*] Generating {args.num_samples} random samples with truncation psi={args.truncation_psi}...")
    z = torch.randn(args.num_samples, args.latent_dim, device=device)
    with torch.no_grad():
        images = generator.generate(z, truncation_psi=args.truncation_psi)
    out_file = os.path.join(args.output_dir, "random_samples.png")
    save_sample_grid(images, out_file, nrow=int(args.num_samples ** 0.5) or 4)
    print(f"[+] Saved random samples grid to: {out_file}")


def generate_interpolation(generator, args, device):
    print(f"[*] Generating latent interpolation strip ({args.steps} steps)...")
    z1 = torch.randn(1, args.latent_dim, device=device)
    z2 = torch.randn(1, args.latent_dim, device=device)

    interpolated_latents = []
    for step in range(args.steps):
        alpha = step / float(args.steps - 1)
        z_interp = slerp(alpha, z1, z2)
        interpolated_latents.append(z_interp)

    z_batch = torch.cat(interpolated_latents, dim=0)
    with torch.no_grad():
        images = generator.generate(z_batch, truncation_psi=args.truncation_psi)

    out_file = os.path.join(args.output_dir, "interpolation_strip.png")
    save_sample_grid(images, out_file, nrow=args.steps)
    print(f"[+] Saved interpolation strip to: {out_file}")


def generate_style_mix(generator, args, device):
    print("[*] Generating Style Mixing matrix...")
    num_sources = 4
    z_rows = torch.randn(num_sources, args.latent_dim, device=device)
    z_cols = torch.randn(num_sources, args.latent_dim, device=device)

    with torch.no_grad():
        w_rows = generator.mapping(z_rows)
        w_cols = generator.mapping(z_cols)

        mixed_images = []
        for i in range(num_sources):
            row_images = []
            for j in range(num_sources):
                # Mix styles: early layers from col (style 1), late layers from row (style 2)
                x = generator.initial_constant.repeat(1, 1, 1, 1)
                x = generator.init_conv(x, w_cols[j : j + 1])
                for idx, block in enumerate(generator.blocks):
                    curr_w = w_cols[j : j + 1] if idx < len(generator.blocks) // 2 else w_rows[i : i + 1]
                    x = block(x, curr_w)
                rgb = generator.to_rgb(x, w_rows[i : i + 1])
                row_images.append(rgb)
            mixed_images.append(torch.cat(row_images, dim=0))

        all_images = torch.cat(mixed_images, dim=0)

    out_file = os.path.join(args.output_dir, "style_mixing_grid.png")
    save_sample_grid(all_images, out_file, nrow=num_sources)
    print(f"[+] Saved style mixing matrix to: {out_file}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    generator = load_generator(args, device)

    if args.mode == "random":
        generate_random(generator, args, device)
    elif args.mode == "interpolate":
        generate_interpolation(generator, args, device)
    elif args.mode == "style_mix":
        generate_style_mix(generator, args, device)


if __name__ == "__main__":
    main()
