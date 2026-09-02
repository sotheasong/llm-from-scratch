import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt

from configs.gpt import GPT_CONFIG_124M
from model.layers import LayerNorm, GELU

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(               
            *[DummyTransformerBlock(cfg)               
              for _ in range(cfg["n_layers"])]         
        )                                              
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])     
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class DummyLayerNorm(nn.Module):           
    def __init__(self, normalized_shape, eps=1e-5):    
        super().__init__()

    def forward(self, x):
        return x

class DummyTransformerBlock(nn.Module):    
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):     
        return x

if __name__ == "__main__":
    torch.manual_seed(123)
    torch.set_printoptions(sci_mode=False)
    batch_example = torch.randn(2, 5)
    layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
    out = layer(batch_example)
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    
    gelu, relu = GELU(), nn.ReLU() 
    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)
    plt.figure(figsize=(8,3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
        plt.subplot(1,2,i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)
    plt.tight_layout()
    plt.show()