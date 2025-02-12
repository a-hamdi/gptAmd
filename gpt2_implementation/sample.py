import argparse
from pathlib import Path
import torch

from src.config import GPT2Config
from src.model import GPT2
from src.tokenizer import GPT2Tokenizer
from src.inference import GPT2Inference

def main():
    parser = argparse.ArgumentParser(description="Generate text using GPT-2")
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, help="Path to the tokenizer files")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Input prompt for generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering value")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) filtering value")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences to generate")
    args = parser.parse_args()

    # Check if model exists, if not, create a small model for demonstration
    if args.model_path is None:
        print("No model path provided. Creating a small model for demonstration...")
        config = GPT2Config(
            vocab_size=50257,
            n_positions=128,
            n_embd=256,
            n_layer=4,
            n_head=4
        )
        model = GPT2(config)
        tokenizer = GPT2Tokenizer()
        
        # Save model and tokenizer
        output_dir = Path("demo_model")
        output_dir.mkdir(exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.__dict__
        }, output_dir / "model.pt")
        
        args.model_path = str(output_dir / "model.pt")
        args.tokenizer_path = str(output_dir)
    
    # Initialize inference
    inference = GPT2Inference.from_pretrained(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path
    )
    
    # Generate text
    print(f"\nGenerating {args.num_return_sequences} sequence(s) from prompt: {args.prompt}\n")
    generated_sequences = inference.generate(
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_return_sequences=args.num_return_sequences
    )
    
    # Print generated sequences
    for i, sequence in enumerate(generated_sequences, 1):
        print(f"Generated sequence {i}:")
        print(sequence)
        print("-" * 50)
    
    # Score sequences
    scores = inference.score_sequences(generated_sequences)
    for i, (sequence, score) in enumerate(zip(generated_sequences, scores), 1):
        print(f"\nSequence {i} score: {score:.4f}")

if __name__ == "__main__":
    main() 