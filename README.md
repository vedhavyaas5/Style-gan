# StyleGAN in PyTorch

A clean, modular, and educational PyTorch implementation of **StyleGAN (Style-based Generative Adversarial Network)** architecture.

---

## 🌟 Key Features

- **Mapping Network ($f: \mathcal{Z} \to \mathcal{W}$)**: 8-layer MLP with Pixel Normalization mapping Gaussian noise $z$ to an intermediate disentangled style space $w$.
- **Adaptive Instance Normalization (AdaIN)**: Style vector $w$ dynamically modulates normalized feature channels using learned affine transformations ($y_{s}, y_{b}$).
- **Explicit Noise Injection**: Dedicated per-channel learned scaling factors to inject spatial Gaussian noise for fine stochastic variation (e.g. textures, hair details).
- **Style Mixing Regularization**: Interleaves styles from multiple latent codes during training to encourage localized style disentanglement (coarse vs. fine styles).
- **Minibatch Standard Deviation**: Appends batch-level standard deviation statistics to discriminator feature maps to penalize mode collapse.
- **Equalized Learning Rate ($W$-Scale)**: Dynamic runtime weight scaling (He initialization) ensures consistent variance across all layers.
- **Non-saturating Logistic Loss with $R_1$ Regularization**: Enforces stable convergence via gradient penalty on real images ($\frac{\gamma}{2} \mathbb{E}[\|\nabla D(x)\|^2]$).
- **Inference & Latent Manipulation**: Built-in support for truncation trick ($\psi$), spherical latent interpolation (`slerp`), and style-mixing grids.
- **Built-in Procedural Synthetic Dataset**: Test and train immediately out-of-the-box without waiting to download multi-gigabyte datasets.

---

## 📁 Repository Structure

```
deeplearningstyleGAN/
├── models/
│   ├── __init__.py
│   ├── blocks.py           # PixelNorm, EqualizedConv2d, EqualizedLinear, AdaIN, NoiseInjection, MinibatchStdDev
│   ├── mapping.py          # 8-layer Mapping Network (Z -> W)
│   ├── generator.py        # StyleGAN Synthesis Generator with style mixing & truncation
│   └── discriminator.py    # ResNet-style downsampling Discriminator with MinibatchStdDev
├── data/
│   ├── __init__.py
│   └── dataset.py          # Custom image loader & procedural SyntheticShapeDataset
├── utils/
│   ├── __init__.py
│   └── helpers.py          # EMA, R1 gradient penalty, logistic losses, image grid saver, slerp
├── train.py                # Main training loop with lazy R1 and EMA
├── generate.py             # Inference script: random sampling, interpolation, style mixing
├── demo.py                 # Self-contained zero-config verification script
├── requirements.txt        # Project dependencies
└── README.md
```

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/Harshavarthan001/deeplearningstyleGAN.git
cd deeplearningstyleGAN
pip install -r requirements.txt
```

### 2. Verify Architecture & Pipeline (Zero Config)

Run the automated verification script to validate forward and backward passes, EMA, and loss computation:

```bash
python demo.py
```

---

## 🏋️ Training

### Training with Built-in Synthetic Dataset (Quick Test)

You can immediately test training without any external datasets:

```bash
python train.py --synthetic --epochs 10 --batch_size 16 --image_size 64
```

### Training with Your Own Image Dataset

Place your images (`.jpg`, `.png`, `.webp`) in a folder, then pass `--data_dir`:

```bash
python train.py --data_dir /path/to/images --epochs 50 --batch_size 16 --image_size 64 --lr 0.001
```

### Common Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `None` | Path to dataset directory |
| `--synthetic` | `False` | Train on procedural geometric shapes |
| `--image_size` | `64` | Output image resolution (32, 64, 128) |
| `--batch_size` | `16` | Batch size |
| `--epochs` | `20` | Number of training epochs |
| `--lr` | `0.001` | Learning rate for Adam optimizer |
| `--r1_gamma` | `10.0` | $R_1$ gradient penalty weight |
| `--r1_interval` | `16` | Lazy regularization step interval |
| `--checkpoint_dir` | `checkpoints` | Checkpoint save directory |
| `--samples_dir` | `samples` | Generated sample output directory |

---

## 🎨 Inference & Latent Manipulation

### 1. Unconditional Random Sampling (with Truncation Trick)

```bash
python generate.py --mode random --num_samples 16 --truncation_psi 0.7 --output_dir outputs
```

### 2. Spherical Latent Interpolation (`slerp`)

Smoothly transition between two randomly sampled latent codes in $\mathcal{Z}$:

```bash
python generate.py --mode interpolate --steps 12 --truncation_psi 0.7 --output_dir outputs
```

### 3. Style Mixing Matrix

Cross-mix coarse styles (pose, layout) and fine styles (color, texture) between source latents:

```bash
python generate.py --mode style_mix --output_dir outputs
```

---

## 🧠 Core Architecture Overview

### 1. Mapping Network ($f$)
Maps standard Gaussian latent $z \in \mathcal{Z} \sim \mathcal{N}(0, I)$ into an intermediate latent $w \in \mathcal{W}$:
$$\text{PixelNorm}(z) = \frac{z}{\sqrt{\frac{1}{C}\sum z_i^2 + \epsilon}}$$
$$\mathbf{w} = \text{MLP}_{8}(\text{PixelNorm}(z))$$

### 2. Adaptive Instance Normalization (AdaIN)
Each feature map $x_i$ is normalized and scaled/shifted by style-derived parameters:
$$\text{AdaIN}(x_i, \mathbf{w}) = y_{s, i} \left( \frac{x_i - \mu(x_i)}{\sigma(x_i)} \right) + y_{b, i}$$
where $(y_{s}, y_{b}) = \text{Affine}(\mathbf{w})$.

### 3. $R_1$ Regularization
Stabilizes GAN dynamics by penalizing discriminator gradients on the data manifold:
$$\mathcal{L}_{R_1} = \frac{\gamma}{2} \mathbb{E}_{x \sim p_{\text{data}}} \left[ \|\nabla_x D(x)\|^2 \right]$$

---

## 📜 References
- Karras et al., *"A Style-Based Generator Architecture for Generative Adversarial Networks"*, CVPR 2019. [arXiv:1812.04948](https://arxiv.org/abs/1812.04948)
- Mescheder et al., *"Which Training Methods for GANs do actually Converge?"*, ICML 2018.

---

## 📄 License
MIT License