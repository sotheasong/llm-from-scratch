import torch
import torch.nn as nn

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)


# 3.3 simplified self-attention, no trainable weights
def simple_self_attention(x):
    attn_scores = x @ x.T
    attn_weights = torch.softmax(attn_scores, dim=-1)
    # dim = -1 applies softmax along the last dimension (across each row)
    # -> the values in each row sum up to 1
    return attn_weights @ x


# 3.4 self-attention with trainable weights
# -> nn.Parameter version, closest to the maths
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries = x @ self.W_query
        keys    = x @ self.W_key
        values  = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # divide by sqrt(d_k) before the softmax
        # -> Var(q . k) grows with d_k, big scores make softmax go one-hot and kill the gradient
        return attn_weights @ values


# -> nn.Linear version, same thing but with a better weight init
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
     
    def forward(self, x):
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        return attn_weights @ values


# 3.5 causal attention, masks everything after the current token
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # register_buffer keeps the mask out of the parameters but still moves it with .to(device)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        # -inf before the softmax, so those positions come out as exactly 0
        # -> slice to num_tokens so shorter sequences than context_length still work
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # dropout goes after the softmax, so after the mask
        return attn_weights @ values


# 3.6.1 multi-head by literally stacking CausalAttention instances
# -> works, but runs each head in its own loop iteration
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


# 3.6.2 multi-head by splitting one set of weights, this is the one used from here on
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        # each head gets its own slice of the columns, head i owns [i*head_dim : (i+1)*head_dim]

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        # out_proj mixes the heads back together after the concat
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)
        # (b, num_tokens, d_out)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim)
        # view only reinterprets the memory, it does not copy anything

        queries = queries.transpose(1, 2)
        keys    = keys.transpose(1, 2)
        values  = values.transpose(1, 2)
        # (b, num_heads, num_tokens, head_dim)
        # -> num_heads next to the batch dim, so matmul treats each head independently

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # keys.shape[-1] is head_dim here, not d_out, so the scaling is per head
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        # back to (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # transpose leaves the strides permuted, so contiguous() is needed before view()
        return self.out_proj(context_vec)


if __name__ == "__main__":
    torch.manual_seed(123)

    print("3.3 simplified self-attention")
    print(simple_self_attention(inputs), "\n")

    d_in, d_out = inputs.shape[-1], 2

    print("3.4 SelfAttention_v2")
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs), "\n")

    batch = torch.stack((inputs, inputs), dim=0)  # (2, 6, 3)
    context_length = batch.shape[1]

    print("3.5 CausalAttention")
    ca = CausalAttention(d_in, d_out, context_length, dropout=0.0)
    print(ca(batch), "\n")

    print("3.6.2 MultiHeadAttention")
    mha = MultiHeadAttention(d_in, d_out=2, context_length=context_length,
                             dropout=0.0, num_heads=2)
    out = mha(batch)
    print(out)
    print("shape:", out.shape)
