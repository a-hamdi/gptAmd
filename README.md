# GPT-2 Implementation with ROCm Support

A complete, modular implementation of GPT-2 using PyTorch with ROCm support for AMD GPUs.

## Features

- Full GPT-2 architecture implementation
- ROCm support for AMD GPUs
- Efficient parallelization
- Modular and extensible design
- Comprehensive tokenization pipeline
- Training and inference scripts
- Well-documented codebase

## Model Architecture

The implementation includes three model sizes:

- Small (45M parameters):
  - 6 transformer layers
  - 8 attention heads
  - 512 embedding dimensions

- Medium (117M parameters):
  - 12 transformer layers
  - 12 attention heads
  - 768 embedding dimensions

- Large (345M parameters):
  - 24 transformer layers
  - 16 attention heads
  - 1024 embedding dimensions

## Installation

### Prerequisites

- Python 3.8+
- ROCm (for AMD GPU support)
- PyTorch with ROCm support

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gpt2-rocm.git
cd gpt2-rocm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the model on Wikidata5M dataset:

```bash
python train_wikidata.py \
    --model_size small \
    --batch_size 32 \
    --max_length 512 \
    --buffer_size 5000 \
    --num_epochs 1 \
    --output_dir trained_model
```

### Inference

Generate text using a trained model:

```bash
python sample.py \
    --model_path trained_model/best_model.pt \
    --prompt "Once upon a time" \
    --max_length 100
```

## Project Structure

```
gpt2_implementation/
├── src/
│   ├── model.py          # Core GPT-2 architecture
│   ├── tokenizer.py      # Tokenization pipeline
│   ├── config.py         # Model configuration
│   ├── training.py       # Training loop and utilities
│   ├── inference.py      # Text generation pipeline
│   └── data.py          # Dataset and data loading
├── train_wikidata.py     # Training script
├── sample.py             # Inference script
└── requirements.txt      # Dependencies
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 