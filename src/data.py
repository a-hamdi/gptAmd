class WikidataDataset(Dataset):
    """Dataset class for Wikidata5M."""
    
    def __init__(
        self,
        tokenizer: GPT2Tokenizer,
        max_length: int = 1024,
        split: str = "train",
        buffer_size: int = 5000,
        device: torch.device = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dataset will use device: {self.device}")
        
        # Load dataset
        print("Loading Wikidata5M dataset...")
        full_dataset = load_dataset("intfloat/wikidata5m", split=split, streaming=True)
        
        # Load fixed number of examples
        self.examples = []
        print(f"Loading {buffer_size} examples...")
        for item in full_dataset:
            if len(self.examples) >= buffer_size:
                break
            if item and isinstance(item, dict) and "text" in item:
                processed = self._process_text(item["text"])
                self.examples.append(processed)
        
        print(f"Loaded {len(self.examples)} examples")
        if len(self.examples) == 0:
            raise RuntimeError("Could not load any examples")
    
    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Process a single text example."""
        # Ensure text is not empty
        if not text or not isinstance(text, str):
            text = "<|endoftext|>"
        
        # Tokenize text
        try:
            tokens = self.tokenizer.encode(text)
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            tokens = [self.tokenizer.encoder["<|endoftext|>"]]
        
        # Ensure we have at least 2 tokens (for input and label)
        if len(tokens) < 2:
            tokens = [self.tokenizer.encoder["<|endoftext|>"]] * 2
        
        # Truncate or pad sequence
        if len(tokens) > self.max_length:
            start_idx = np.random.randint(0, len(tokens) - self.max_length + 1)
            tokens = tokens[start_idx:start_idx + self.max_length]
        else:
            # Pad with end of text token
            tokens.extend([self.tokenizer.encoder["<|endoftext|>"]] * (self.max_length - len(tokens)))
        
        # Create input_ids and labels and move to device
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long, device=self.device)
        labels = torch.tensor(tokens[1:], dtype=torch.long, device=self.device)
        
        return {
            "input_ids": input_ids,
            "labels": labels
        }
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]

def create_dataloader(
    tokenizer: GPT2Tokenizer,
    batch_size: int = 8,
    max_length: int = 1024,
    num_workers: int = 0,
    split: str = "train",
    buffer_size: int = 5000,
    device: torch.device = None
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for the Wikidata5M dataset."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = WikidataDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        split=split,
        buffer_size=buffer_size,
        device=device
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Keep at 0 since data is already on GPU
        pin_memory=False  # No need for pin_memory since data is already on GPU
    )
    
    return dataloader 