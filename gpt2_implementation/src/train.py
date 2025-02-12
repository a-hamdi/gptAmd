import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from pathlib import Path
from .model import GPT2
from .tokenizer import GPT2Tokenizer
from .data_loader import get_dataloader

def train(
    model: GPT2,
    tokenizer: GPT2Tokenizer,
    save_dir: str = "checkpoints",
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    max_length: int = 1024,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_every: int = 1000,
):
    """Train the GPT-2 model on Wikidata5m dataset"""
    
    print(f"Training on device: {device}")
    model = model.to(device)
    model.train()
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup training
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    
    # Get data loader
    dataloader = get_dataloader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length
    )
    
    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        progress_bar = tqdm(dataloader, desc="Training")
        epoch_loss = 0
        
        for batch in progress_bar:
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            # Input is all tokens except last, target is all tokens except first
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]
            
            outputs = model(input_ids)
            
            # Calculate loss
            # Reshape outputs and targets for loss calculation
            loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                target_ids.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Save checkpoint
            if global_step % save_every == 0:
                checkpoint_path = save_dir / f"checkpoint_{global_step}.pt"
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                
            global_step += 1
            
        # End of epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Average epoch loss: {avg_epoch_loss:.4f}")
        
        # Save epoch checkpoint
        checkpoint_path = save_dir / f"epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, checkpoint_path) 