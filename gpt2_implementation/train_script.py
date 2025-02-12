from src.model import GPT2
from src.tokenizer import GPT2Tokenizer
from src.train import train

def main():
    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer()
    model = GPT2(
        vocab_size=len(tokenizer.encoder),
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12
    )
    
    # Train the model
    train(
        model=model,
        tokenizer=tokenizer,
        num_epochs=10,
        batch_size=8,
        learning_rate=3e-4,
        max_length=1024,
        save_dir="checkpoints"
    )

if __name__ == "__main__":
    main() 