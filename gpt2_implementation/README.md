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

## Installation

### Prerequisites

- Python 3.8+
- ROCm (for AMD GPU support)
- PyTorch with ROCm support

### Setting up ROCm

1. Install ROCm following the official guide: [ROCm Installation](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)

2. Verify ROCm installation:
```bash
rocm-smi
```

### Installing the Package

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gpt2-implementation.git
cd gpt2-implementation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

```python
from src.training import train_model
from src.config import GPT2Config

# Configure model parameters
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12
)

# Start training
train_model(
    config=config,
    train_data_path="path/to/training/data",
    output_dir="path/to/save/model",
    batch_size=8,
    num_epochs=3
)
```

### Inference

```python
from src.inference import GPT2Inference

# Load model and generate text
model = GPT2Inference.from_pretrained("path/to/saved/model")
generated_text = model.generate(
    prompt="Once upon a time",
    max_length=100,
    temperature=0.7
)
print(generated_text)
```

## Project Structure

```
gpt2_implementation/
├── src/
│   ├── __init__.py
│   ├── model.py          # Core GPT-2 architecture
│   ├── tokenizer.py      # Tokenization pipeline
│   ├── config.py         # Model configuration
│   ├── training.py       # Training loop and utilities
│   ├── inference.py      # Text generation pipeline
│   └── utils.py          # Helper functions
├── tests/               # Unit tests
├── data/                # Training data and tokenizer files
├── requirements.txt     # Project dependencies
└── setup.py            # Package setup file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
