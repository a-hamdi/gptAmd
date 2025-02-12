import argparse
from pathlib import Path
import torch
import wandb
import os

from src.config import GPT2Config
from src.model import GPT2
from src.tokenizer import GPT2Tokenizer
from src.training import train_model
from src.data import create_dataloader

def main():
    # Force PyTorch to use ROCm/HIP backend
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    
    parser = argparse.ArgumentParser(description="Train GPT-2 on Wikidata5M")
    parser.add_argument("--output_dir", type=str, default="trained_model", help="Directory to save model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--buffer_size", type=int, default=5000, help="Size of streaming buffer")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "medium", "large"], help="Model size")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    args = parser.parse_args()

    # Verify GPU is available
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. Please check PyTorch installation with ROCm support.")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Set up model configuration based on size
    model_configs = {
        "small": dict(n_layer=6, n_head=8, n_embd=512),
        "medium": dict(n_layer=12, n_head=12, n_embd=768),
        "large": dict(n_layer=24, n_head=16, n_embd=1024),
    }
    
    config = GPT2Config(**model_configs[args.model_size])
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer()
    
    # Create data loader
    train_loader = create_dataloader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=0,  # Must be 0 for data on GPU
        split="train",
        buffer_size=args.buffer_size
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Calculate steps per epoch based on buffer size
    steps_per_epoch = args.buffer_size // args.batch_size
    
    # Train model
    train_model(
        config=config,
        train_dataset=train_loader.dataset,
        output_dir=str(output_dir),
        tokenizer=tokenizer,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_wandb=args.use_wandb,
        wandb_project="gpt2-wikidata5m",
        steps_per_epoch=steps_per_epoch
    )

if __name__ == "__main__":
    main() 