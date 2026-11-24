import torch


# 4.7 greedy decoding, the loop that turns logits into text
# -> the model only ever predicts one next token, so generating n tokens means
#    running it n times and feeding each output back in as input
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        # keep only the last context_size tokens
        # -> pos_emb has exactly context_size rows, so a longer input raises
        #    IndexError: index out of range in self
        # -> once the text passes that length the oldest tokens fall out of view
        with torch.no_grad():
            logits = model(idx_cond)
            # no_grad skips building the autograd graph, nothing here is trained
        logits = logits[:, -1, :]
        # the model scores every position, but only the last one predicts the
        # token that has not been written yet
        # -> (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # softmax is monotonic, so argmax picks the same id with or without it
        # -> kept because chapter 5 swaps argmax for sampling, which does need
        #    real probabilities
        # -> keepdim gives (batch, 1) instead of (batch,), the shape cat needs
        idx = torch.cat((idx, idx_next), dim=1)
        # the new token joins the sequence and becomes input on the next pass
    return idx


# 5.1.1 text <-> token ID helpers, the wrapper around the tokenizer calls
# -> unsqueeze(0) adds the batch dimension the model expects, (n,) becomes (1, n)
# -> squeeze(0) takes it back off before decoding
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def print_sample_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

if __name__ == "__main__":
    import tiktoken

    from configs.gpt import GPT_CONFIG_124M
    from src.model.gpt import GPTModel
    from matplotlib import pyplot as plt

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)

    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    # print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    # print("encoded_tensor.shape:", encoded_tensor.shape)
    # unsqueeze(0) adds the batch dimension, (4,) becomes (1, 4)

    model = GPTModel(GPT_CONFIG_124M)
    model.to("cpu")
    model.eval()
    # eval turns dropout off, otherwise the same prompt generates something
    # different every run
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    # print("Output:", out)
    # print("Output length:", len(out[0]))
    # 10, the 4 prompt tokens plus the 6 that were generated

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    # print("Decoded text:", decoded_text)
    # gibberish, and correctly so, the weights are still random
    # -> training is chapter 5

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text_simple(
        model = model,
        idx = text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens = 25,
        context_size = GPT_CONFIG_124M["context_length"]
    )
    # print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    vocab = {
        "closer": 0,
        "every": 1,
        "effort": 2,
        "forward": 3,
        "inches": 4,
        "moves": 5,
        "pizza": 6,
        "toward": 7,
        "you": 8,
    }

    inverse_vocab = {v: k for k, v in vocab.items()}
    
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )
    
    probas = torch.softmax(next_token_logits, dim=-1)
    next_token_id = torch.argmax(probas).item()
    # print(inverse_vocab[next_token_id])

    # print_sample_tokens(probas)

    temperatures = [1, 0.1, 5]
    scaled_probas = [softmax_with_temperature(next_token_logits, t) for t in temperatures]
    x = torch.arange(len(vocab))
    bar_width = 0.15
    fig, ax = plt.subplots(figsize = (5,3))
    for i, T in enumerate(temperatures):
        rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label = f"Temperature = {T}")
    ax.set_ylabel("Probability")
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation = 90)
    ax.legend()
    plt.tight_layout()
    # plt.savefig("../../figures/temperature.png", dpi = 300)

    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)
    print("Top logits:", top_logits)
    print("Top positions:", top_pos)
    
    new_logits = torch.where(
        condition = next_token_logits < top_logits[-1],
        input = torch.tensor(float("-inf")),
        other = next_token_logits
    )
    print("New logits:", new_logits)
    
    topk_probas = torch.softmax(new_logits, dim=0)
    print("Top-k probabilities:", topk_probas)