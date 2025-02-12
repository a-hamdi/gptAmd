import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class GPT2(nn.Module):
    """GPT-2 language model."""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.n_positions, config.n_embd),
            'drop': nn.Dropout(config.embd_pdrop),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon),
        })
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = input_ids.device
        b, t = input_ids.size()
        
        if position_ids is None:
            position_ids = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
            
        # Forward through embeddings
        token_embeddings = self.transformer.wte(input_ids)
        position_embeddings = self.transformer.wpe(position_ids)
        
        x = self.transformer.drop(token_embeddings + position_embeddings)
        
        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        
        # Project to vocabulary
        logits = F.linear(x, self.transformer.wte.weight)
        
        return logits 