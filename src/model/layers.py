import torch
import torch.nn as nn


# 4.2 layer normalization
# -> rescales each token's own feature vector to mean 0, variance 1
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # dim=-1 is the embedding dimension, so each token is normalised on its own
        # -> keepdim keeps the axis as size 1 so it broadcasts back against x
        # -> unbiased=False divides by n, matching GPT-2's original TensorFlow code
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
        # scale and shift start at 1 and 0, so this begins as pure normalisation


# 4.3 GELU, the curve-fitted approximation GPT-2 was trained with
# -> the exact form is x * phi(x), phi being the standard Gaussian CDF
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# 4.3 feed forward, expand by 4x, activate, contract back
# -> acts on each token position independently, attention is what mixes positions
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")            # save to a file instead of opening a window
    import matplotlib.pyplot as plt

    from configs.gpt import GPT_CONFIG_124M

    torch.manual_seed(123)
    torch.set_printoptions(sci_mode=False)

    print("4.2 what layer norm does")
    batch_example = torch.randn(2, 5)
    out = nn.Sequential(nn.Linear(5, 6), nn.ReLU())(batch_example)
    print("  before, mean:", out.mean(dim=-1).tolist())
    print("  before, var :", out.var(dim=-1, unbiased=False).tolist())

    out_ln = LayerNorm(emb_dim=6)(out)
    print("  after,  mean:", [round(v, 6) for v in out_ln.mean(dim=-1).tolist()])
    print("  after,  var :", [round(v, 6) for v in out_ln.var(dim=-1, unbiased=False).tolist()])

    print("\n4.3 GELU vs ReLU")
    gelu, relu = GELU(), nn.ReLU()
    for v in [-2.0, -0.75, 0.0, 1.0]:
        t = torch.tensor([v])
        print(f"  x={v:6.2f}  gelu={gelu(t).item():+.4f}  relu={relu(t).item():+.4f}")

    x = torch.linspace(-3, 3, 100)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, (fn, label) in zip(axes, [(gelu, "GELU"), (relu, "ReLU")]):
        ax.plot(x, fn(x))
        ax.set_title(f"{label} activation function")
        ax.set_xlabel("x"); ax.set_ylabel(f"{label}(x)"); ax.grid(True)
    fig.tight_layout()
    out_png = Path(__file__).resolve().parents[2] / "figures" / "gelu_vs_relu.png"
    fig.savefig(out_png, dpi=120)
    print(f"  plot saved to {out_png.relative_to(Path.cwd())}")

    print("\n4.3 FeedForward")
    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2, 3, GPT_CONFIG_124M["emb_dim"])
    hidden = ffn.layers[0](x)
    print(f"  {x.shape[-1]} -> {hidden.shape[-1]} -> {ffn(x).shape[-1]}")
    print("  in", tuple(x.shape), "-> out", tuple(ffn(x).shape))
