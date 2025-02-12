import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

from .model import GPT2
from .config import GPT2Config
from .tokenizer import GPT2Tokenizer

class GPT2Inference:
    """Inference class for GPT-2 text generation."""
    
    def __init__(
        self,
        model: GPT2,
        tokenizer: GPT2Tokenizer,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        tokenizer_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None
    ) -> "GPT2Inference":
        """Load model and tokenizer from pretrained files."""
        model_path = Path(model_path)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        config = GPT2Config(**checkpoint['config'])
        
        # Initialize model and load weights
        model = GPT2(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize tokenizer
        if tokenizer_path is None:
            tokenizer_path = model_path.parent
        tokenizer = GPT2Tokenizer(
            vocab_file=Path(tokenizer_path) / "vocab.json",
            merges_file=Path(tokenizer_path) / "merges.txt"
        )
        
        return cls(model, tokenizer, device)
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        **kwargs
    ) -> List[str]:
        """Generate text based on the input prompt."""
        # Encode prompt
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0).repeat(num_return_sequences, 1)
        
        # Set up generation config
        gen_config = {
            'max_length': max_length,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'repetition_penalty': repetition_penalty,
            **kwargs
        }
        
        # Generate
        with torch.no_grad():
            output_sequences = self._generate_sequences(input_ids, gen_config)
        
        # Decode and return generated sequences
        generated_sequences = []
        for output in output_sequences:
            generated_sequence = self.tokenizer.decode(output.tolist())
            generated_sequences.append(generated_sequence)
        
        return generated_sequences
    
    def _generate_sequences(
        self,
        input_ids: torch.Tensor,
        gen_config: Dict[str, Any]
    ) -> torch.Tensor:
        """Core generation logic."""
        batch_size = input_ids.shape[0]
        
        # Track generated sequences
        generated = input_ids
        
        for _ in range(gen_config['max_length'] - input_ids.shape[1]):
            # Get logits for next token
            with torch.amp.autocast(device_type=self.device.type):
                outputs = self.model(generated)
                next_token_logits = outputs[:, -1, :] / gen_config['temperature']
            
            # Apply repetition penalty
            if gen_config['repetition_penalty'] != 1.0:
                for i in range(batch_size):
                    for previous_token in set(generated[i].tolist()):
                        next_token_logits[i, previous_token] /= gen_config['repetition_penalty']
            
            # Apply top-k filtering
            if gen_config['top_k'] is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, gen_config['top_k'])[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if gen_config['top_p'] is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > gen_config['top_p']
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat((generated, next_tokens), dim=1)
            
            # Check if any sequence has generated an EOS token
            if (next_tokens == self.model.config.eos_token_id).any():
                break
        
        return generated
    
    @torch.no_grad()
    def score_sequences(
        self,
        sequences: List[str],
        batch_size: int = 8
    ) -> List[float]:
        """Score a list of sequences using the model's log probabilities."""
        scores = []
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            # Tokenize sequences
            encodings = [self.tokenizer.encode(seq) for seq in batch_sequences]
            max_len = max(len(enc) for enc in encodings)
            
            # Pad sequences
            padded = torch.full(
                (len(batch_sequences), max_len),
                self.model.config.eos_token_id,
                dtype=torch.long,
                device=self.device
            )
            
            for j, enc in enumerate(encodings):
                padded[j, :len(enc)] = torch.tensor(enc, dtype=torch.long)
            
            # Get model outputs
            with torch.amp.autocast(device_type=self.device.type):
                logits = self.model(padded)
            
            # Calculate sequence scores
            for j, enc in enumerate(encodings):
                seq_len = len(enc)
                seq_logits = logits[j, :seq_len-1]  # -1 because we don't need to predict after the last token
                seq_targets = padded[j, 1:seq_len]  # Shift right to get targets
                
                # Calculate cross entropy loss
                seq_log_probs = F.log_softmax(seq_logits, dim=-1)
                token_scores = seq_log_probs.gather(1, seq_targets.unsqueeze(1)).squeeze(1)
                score = token_scores.mean().item()
                scores.append(score)
        
        return scores
