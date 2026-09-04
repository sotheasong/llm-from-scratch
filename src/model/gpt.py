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
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
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
        # positions are 0..seq_len-1, the same for every sequence in the batch
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        # logits are raw scores, one per vocab entry per position
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
        # shape in equals shape out, which is what lets these stack


if __name__ == "__main__":
    import tiktoken

    from configs.gpt import GPT_CONFIG_124M

    torch.manual_seed(123)
    torch.set_printoptions(sci_mode=False)

    print("4.1 the skeleton runs end to end")
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = torch.stack([
        torch.tensor(tokenizer.encode("Every effort moves you")),
        torch.tensor(tokenizer.encode("Every day holds a")),
    ], dim=0)
    print("  token IDs:", batch.tolist())
    logits = DummyGPTModel(GPT_CONFIG_124M)(batch)
    print("  logits shape:", tuple(logits.shape), "= (batch, tokens, vocab_size)")

    print("\n4.4 without shortcut connections")
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1., 0., -1.]])
    torch.manual_seed(123)
    print_gradients(ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False), sample_input)

    print("\n4.4 with shortcut connections")
    torch.manual_seed(123)
    print_gradients(ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True), sample_input)

    print("\n4.5 the transformer block")
    torch.manual_seed(123)
    block = TransformerBlock(GPT_CONFIG_124M)
    x = torch.rand(2, 4, GPT_CONFIG_124M["emb_dim"])
    print("  in ", tuple(x.shape), "-> out", tuple(block(x).shape))

    n = lambda m: sum(p.numel() for p in m.parameters())
    print(f"  attention    {n(block.att):>10,}")
    print(f"  feed forward {n(block.ff):>10,}  ({n(block.ff) / n(block.att):.0f}x the attention)")
    print(f"  norms        {n(block.norm1) + n(block.norm2):>10,}")
    print(f"  per block    {n(block):>10,}")
    print(f"  x{GPT_CONFIG_124M['n_layers']} blocks  {GPT_CONFIG_124M['n_layers'] * n(block):>10,}")
