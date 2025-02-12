from datasets import load_dataset
from typing import List, Iterator, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from .tokenizer import GPT2Tokenizer

class WikiDataset(Dataset):
    """Dataset wrapper for Wikidata5m"""
    
    def __init__(
        self,
        tokenizer: GPT2Tokenizer,
        max_length: int = 1024,
        split: str = "train"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load the dataset
        print("Loading Wikidata5m dataset...")
        self.dataset = load_dataset("intfloat/wikidata5m")[split]
        print(f"Loaded {len(self.dataset)} examples")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Get text sample
        text = self.dataset[idx]["text"]
        
        # Tokenize and prepare for model input
        tokens = self.tokenizer.encode(text)
        
        # Truncate or pad sequence to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            # Pad with end of text token
            tokens.extend([self.tokenizer.encoder["<|endoftext|>"]] * (self.max_length - len(tokens)))
            
        return torch.tensor(tokens, dtype=torch.long)

def get_dataloader(
    tokenizer: GPT2Tokenizer,
    batch_size: int = 8,
    max_length: int = 1024,
    split: str = "train",
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """Create a DataLoader for the Wikidata5m dataset"""
    
    dataset = WikiDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        split=split
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    ) 