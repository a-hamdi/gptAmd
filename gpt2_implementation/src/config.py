from dataclasses import dataclass
from typing import Optional

@dataclass
class GPT2Config:
    """Configuration class for GPT-2 model parameters."""
    
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: Optional[int] = None
    activation_function: str = "gelu"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    use_cache: bool = True
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    
    def __post_init__(self):
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd
            
    @classmethod
    def from_pretrained(cls, model_type: str = "gpt2"):
        """Load a predefined configuration based on model type."""
        configs = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }
        
        if model_type not in configs:
            raise ValueError(f"Model type {model_type} not found. Available types: {list(configs.keys())}")
            
        return cls(**configs[model_type])
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
        }
