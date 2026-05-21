# PyTorch Deep Dive

Hands-on PyTorch curriculum. Each module is a **Jupyter notebook** you run, edit, and explore.

## Setup

```powershell
# Activate venv
.\.venv\Scripts\Activate.ps1

# PyTorch with CUDA 12.4 (RTX 4070) — install as a matched set
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .

# Launch Jupyter
jupyter notebook
# or open .ipynb files directly in Cursor/VS Code (select .venv kernel)
```

## How to learn

1. Read `docs/CURRICULUM.md` for the full roadmap.
2. Open notebooks in order (`00_setup/phase0_setup.ipynb` → …).
3. Run cells, modify code, break things, fix them — that's where understanding lives.

## Structure

```
00_setup/          phase0_setup.ipynb — GPU & devices
01_foundations/    Tensors, autograd, computation graph
02_neural_nets/    nn.Module, layers, activations
03_training/       Loss, optimizers, training loops
04_data/           Datasets, DataLoader, transforms
05_gpu/            Device placement, memory, mixed precision
06_computer_vision/ CNNs, transfer learning
07_nlp/            Embeddings, RNNs, transformers intro
08_advanced/       Custom autograd, distributed, export
projects/          Capstone projects
shared/            Reusable utilities
```
