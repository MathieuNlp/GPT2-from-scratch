import torch
import torch.nn as nn
from torch.nn import functional as F

from src.config.config import GPTConfig

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Clean residual pathways are desirable for gradients. (+ operation splits the gradient in backprob cf. micrograd)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        
        return x

class GPT(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

    
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)), # There are 12 layers in the gpt model => 12 Blocks
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)