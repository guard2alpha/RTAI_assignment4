# RTAI Assignment 4 - α,β-Crown Neural Network Verification

## Overview

This project applies **α,β-Crown**, a state-of-the-art neural network verifier, to a simple MLP model trained on MNIST.  
The model (784 → 64 → ReLU → 10) is verified for **L∞ robustness** with perturbation radius ε = 0.03.

- **Model**: SimpleMLP (same model used in Assignment 3 with Marabou)
- **Dataset**: MNIST test set (50 samples)
- **Verification property**: L∞-ball robustness, ε = 0.03

## Requirements

- Python 3.11
- CUDA 12.6 (optional, CPU also supported)
- Conda environment recommended

## Installation

### 1. Create and activate conda environment

```bash
conda create -n RTAI4 python=3.11 -y
conda activate RTAI4
```

### 2. Install PyTorch (with CUDA)

```bash
pip install torch==2.11.0+cu126 torchvision==0.26.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

For CPU only:

```bash
pip install torch==2.11.0 torchvision==0.26.0
```

### 3. Install other dependencies

```bash
pip install -r requirements.txt
```

### 4. Clone and install α,β-Crown

```bash
git clone --depth 1 https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd alpha-beta-CROWN
git submodule update --init --recursive
pip install .
pip install -e auto_LiRPA
cd ..
```

## Running Verification

```bash
python test.py
```

This will run α,β-Crown on `my_mlp.onnx` using the config in `mnist_mlp_verify.yaml` and print a summary of results.

## Files

| File | Description |
|------|-------------|
| `test.py` | Runs α,β-Crown and displays verification results |
| `mnist_mlp_verify.yaml` | YAML configuration for α,β-Crown |
| `my_mlp.onnx` | Trained SimpleMLP model (MNIST) |
| `requirements.txt` | Python dependencies |
| `report.pdf` | Analysis report |
