import torch
import torch.nn as nn

from src.model.attention import MultiHeadAttention
from src.model.layers import LayerNorm, GELU, FeedForward


# 4.1 placeholders, so the skeleton runs end to end before the real pieces exist
# -> both are replaced by the real LayerNorm and TransformerBlock below
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # the arguments only exist to mimic the real LayerNorm interface

    def forward(self, x):
        return x


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


# 4.1 the GPT skeleton
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# 4.4 a 5 layer network used only to show what shortcut connections do
# -> forward computes the output of the current layer and adds the input back
#    to it when use_shortcut is True
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i + 1]), GELU())
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
                # the derivative of x + F(x) is 1 + F'(x)
                # -> that 1 is an additive path the gradient can take around the layer
            else:
                x = layer_output
                # shapes differ on the last layer (3 -> 1), so no shortcut there
        return x


def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"  {name} has gradient mean of {param.grad.abs().mean().item()}")


# 4.5 the transformer block, repeated n_layers times in the full model
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        # one norm per sub-block, the dropout module is stateless so it is reused

    def forward(self, x):
        shortcut = x                    # attention sub-block, mixes token positions
        x = self.norm1(x)               # pre-LayerNorm, the norm goes before the work
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x                    # feed forward sub-block, each position alone
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

if __name__ == "__main__":
    import tiktoken

    from configs.gpt import GPT_CONFIG_124M
    from src.model.gpt import GPTModel

    torch.manual_seed(123)
    torch.set_printoptions(sci_mode=False)

    tokenizer = tiktoken.get_encoding("gpt2")
    batch = torch.stack([
        torch.tensor(tokenizer.encode("Every effort moves you")),
        torch.tensor(tokenizer.encode("Every day holds a")),
    ], dim=0)
 
    model = GPTModel(GPT_CONFIG_124M)
    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters in the model: {total_params:,}")
    
    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)

    total_params_gpt2 = (
        total_params - sum(p.numel() for p in model.out_head.parameters())
    )
    print(f"\nTotal parameters in the model (excluding output layer): {total_params_gpt2:,}")

    total_size_bytes = total_params * 4       #1
    total_size_mb = total_size_bytes / (1024 * 1024)     #2
    print(f"Total size of the model: {total_size_mb:.2f} MB")