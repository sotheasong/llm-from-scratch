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


if __name__ == "__main__":
    import tiktoken

    from configs.gpt import GPT_CONFIG_124M
    from src.model.gpt import GPTModel

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)

    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)
    # unsqueeze(0) adds the batch dimension, (4,) becomes (1, 4)

    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    # eval turns dropout off, otherwise the same prompt generates something
    # different every run
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output:", out)
    print("Output length:", len(out[0]))
    # 10, the 4 prompt tokens plus the 6 that were generated

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print("Decoded text:", decoded_text)
    # gibberish, and correctly so, the weights are still random
    # -> training is chapter 5
