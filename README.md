# Neural Network from Scratch — NumPy

A fully functional Multi-Layer Perceptron (MLP) implemented from scratch using **only NumPy** (no PyTorch/TensorFlow for model logic). Trains and evaluates on **MNIST** handwritten digit classification, with support for **6 optimizers** and YAML-based configuration.

## Highlights

- **Forward & backward pass** implemented manually with matrix operations
- **Xavier weight initialization** (normal & uniform)
- **6 optimizers**: SGD, Momentum SGD, Nesterov Accelerated Gradient (NAG), Adagrad, RMSProp, Adam
- **Negative Log-Likelihood (NLL)** loss with numerically stable softmax
- **YAML-driven configuration** via [yacs](https://github.com/rbgirshick/yacs)
- Automated training curves, confusion matrices, and per-class accuracy plots

## Project Structure

```
├── main.py                    # Entry point: data loading, training loop, evaluation
├── configuration.yaml         # Hyperparameters (seed, epochs, batch_size, layer_dims, optimizer)
├── train.sh                   # Quick-start training script
├── core/
│   └── functions.py           # train() and valid() batch loops
├── model/
│   ├── model.py               # MLP class (forward, backward, step)
│   ├── model_utils.py         # Linear_Layer with Xavier init
│   └── activations.py         # ReLU, Softmax
├── util/
│   ├── config.py              # YACS config loading
│   ├── loss.py                # NLL loss (forward + backward)
│   ├── optimizer.py           # Optimizer class (SGD, Momentum, NAG, Adagrad, RMSProp, Adam)
│   ├── utils.py               # Seed, standardization, accuracy, MNIST loader, arg parser
│   └── vizualization.py       # Training curves & classification metric plots
└── experiments/               # Auto-generated CSV logs and plots per run
```

## Architecture

The default MLP has **3 linear layers** with ReLU activations and a softmax output:

```
Input (784) → Linear → ReLU → Linear → ReLU → Linear → Softmax → Output (10)
         256            128             10
```

Layer dimensions are configurable via `configuration.yaml`:

```yaml
layer_dims: [784, 256, 128, 10]
```

## Optimizers

| Optimizer | Learning Rate | Momentum (β₁) | Momentum (β₂) |
|-----------|:---:|:---:|:---:|
| SGD | 0.01 | — | — |
| Momentum SGD | 0.01 | 0.9 | — |
| NAG | 0.01 | 0.9 | — |
| Adagrad | 0.01 | — | — |
| RMSProp | 0.001 | — | 0.9 |
| Adam | 0.001 | 0.9 | 0.999 |

## Getting Started

### Prerequisites

```bash
pip install numpy pandas tqdm matplotlib scikit-learn torchvision yacs
```

> **Note:** `torchvision` is used **only** for downloading MNIST data. All model logic is pure NumPy.

### Training

```bash
# Using the shell script
bash train.sh

# Or directly
python main.py --cfg configuration.yaml
```

### Configuration

Edit `configuration.yaml` to change hyperparameters:

```yaml
seed: 42
epochs: 15
batch_size: 64
layer_dims: [784, 256, 128, 10]

optimizer:
  otype: 'Adam'   # Options: SGD, Momentum_SGD, NAG, Adagrad, RMSProp, Adam
```

## Outputs

Training automatically generates under `experiments/MLP_{seed}_{batch_size}/`:

| File | Description |
|------|-------------|
| `train_stats_{optimizer}.csv` | Per-epoch train loss & accuracy |
| `test_stats_{optimizer}.csv` | Final test predictions vs ground truth |
| `training_curves_{optimizer}.png` | Loss & accuracy curves (train vs test) |
| `metrics_{optimizer}.png` | Confusion matrix & per-class accuracy |

## Implementation Details

| Component | Details |
|-----------|---------|
| **Weight Init** | Xavier (normal/uniform) |
| **Activations** | ReLU (forward + backward), Softmax (numerically stable) |
| **Loss** | NLL with combined softmax-cross-entropy gradient |
| **Backprop** | Manual chain rule through each layer |
| **Data** | MNIST (60k train / 10k test), pixel normalization + standardization |
