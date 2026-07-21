# ArtBench Generative Modeling

Implementation and comparative evaluation of generative models from three model
families, trained to generate 32x32 artwork images from the ArtBench-10 dataset.
This is Project 1 (TP1) for the Generative AI course.

The project implements and compares:

- **VAE family**: unconditional VAE and conditional VAE (cVAE)
- **GAN family**: DCGAN and conditional DCGAN (cWGAN-GP)
- **Diffusion family**: pixel-space DDPM/DDIM and conditional diffusion with classifier-free guidance

Models are assessed both quantitatively (FID and KID against a fixed Inception
feature extractor) and qualitatively (visual inspection of per-style samples,
latent-space interpolation, and t-SNE projections).

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── statement/        Assignment PDFs
│   ├── report/           LaTeX report (main.tex) and compiled main.pdf
│   └── todo.txt          Working task list
├── notebooks/
│   └── ArtBench10_Student_Start_Pack.ipynb   Provided starter notebook
├── scripts/
│   ├── artbench_local_dataset.py   Loads the local Kaggle ArtBench-10 batches
│   └── image_fix.py                Utility to split/rebuild sample grids
└── src/
    ├── main.py                  End-to-end pipeline (VAE + GAN + diffusion + eval)
    ├── data_loader.py           Dataset loading, transforms, and the 20% subset
    ├── VAE.py / cVAE.py         Variational autoencoders
    ├── DCGAN.py / cDCGAN.py     GANs (DCGAN and conditional WGAN-GP)
    ├── diffusion.py / cDiffusion.py   Pixel-space diffusion models
    ├── evaluation.py            FID / KID and the evaluation protocol
    ├── qualitative_analysis.py  lerp / slerp interpolation and t-SNE plots
    ├── training_20_percent.csv  IDs defining the 20% training subset
    └── results/                 Generated samples, checkpoints, study DBs (gitignored)
```

Note on paths: the code locates the project root by looking for a folder that
contains both `scripts/` and `ArtBench-10/`, and each training script writes its
outputs to `src/results/`. Run the scripts from the repository root so these
lookups resolve correctly.

## Dataset

The models train on **ArtBench-10**: 60,000 images of size 32x32, evenly split
across 10 artistic-style classes (50,000 train, 10,000 test). An early
exploration phase uses a 20% training subset defined by
`src/training_20_percent.csv`; final runs use the full dataset.

The dataset is not tracked in this repository. Download the original Kaggle
ArtBench-10 (CIFAR-style Python batches) and place it at the repository root so
the layout is:

```
ArtBench-10/
├── ArtBench-10.csv
└── artbench-10-python/
    └── artbench-10-batches-py/
        ├── data_batch_1 ... data_batch_5
        └── test_batch
```

`scripts/artbench_local_dataset.py` reads these batches and exposes them as a
Hugging Face `DatasetDict` with `train` and `test` splits.

## Setup

Requires Python 3.12 and, for practical training times, a CUDA-capable GPU. The
code falls back to CPU automatically when CUDA is unavailable.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run everything from the repository root.

### Full pipeline

`src/main.py` trains a VAE, a conditional WGAN-GP, and a pixel-space diffusion
model, then runs the quantitative evaluation and writes samples and loss curves
to `src/results/`.

```bash
python src/main.py
```

### Individual models

Each model file is a standalone entry point with its own command-line flags.
Common flags include `--use-20pct` (train on the 20% subset), `--epochs`, and,
for GAN and diffusion models, `--bayes-search` to run an Optuna (TPE)
hyperparameter search.

```bash
# VAE on the 20% subset, evaluated over 10 seeds
python src/VAE.py --use-20pct --epochs 50 --eval-seeds 10

# DCGAN hyperparameter search
python src/DCGAN.py --use-20pct --bayes-search --n-trials 30

# Conditional WGAN-GP
python src/cDCGAN.py --use-20pct --epochs 50

# Pixel-space diffusion with a DDIM sampler
python src/diffusion.py --use-20pct --sampler ddim --sample-steps 100

# Conditional diffusion
python src/cDiffusion.py --use-20pct
```

Pass `--help` to any script for the full list of options.

### Qualitative analysis

`src/qualitative_analysis.py` produces latent-space interpolations and t-SNE
projections from saved latent vectors.

```bash
python src/qualitative_analysis.py \
    --latents src/results/latents.npy \
    --labels src/results/labels.npy \
    --output src/results/tsne_latents.png
```

## Evaluation

`src/evaluation.py` standardizes generated and real images to 299x299, extracts
Inception-v3 pool features, and computes:

- **FID** (Frechet Inception Distance) between real and generated feature statistics
- **KID** (Kernel Inception Distance) with a polynomial kernel

The protocol averages metrics across multiple random seeds using a configurable
number of generated images per model.

## Results summary

Reported on the full dataset (200 epochs, 10-seed protocol, 5000 generated
images per model). Lower FID is better.

| Model                       | FID    |
|-----------------------------|--------|
| VAE                         | high (blurry, pixel-wise loss ceiling) |
| DCGAN (WGAN-GP)             | 46.64  |
| Conditional DDIM (with CFG) | 34.62  |
| DDPM (unconditional)        | 27.06  |

The unconditional diffusion model achieved the best fidelity, while the
conditional DDIM variant preserved class boundaries that the conditional GAN
failed to separate. See `docs/report/main.pdf` for the full analysis.

## Authors

- Miguel Castela
- Miguel Martins

DEI, Universidade de Coimbra.

