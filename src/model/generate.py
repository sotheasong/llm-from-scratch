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

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


if __name__ == "__main__":
    import tiktoken

    from configs.gpt import GPT_CONFIG_124M
    from src.model.gpt import GPTModel

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)

    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    # start_context = "Every effort moves you"
    # token_ids = generate_text_simple(
    #     model = model,
    #     idx = text_to_token_ids(start_context, tokenizer),
    #     max_new_tokens = 10,
    #     context_size = GPT_CONFIG_124M["context_length"]
    # )
    # print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]
    targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107, 588, 11311]])  #  " really like chocolate"]
    with torch.no_grad():
        logits = model(inputs)
    probas = torch.softmax(logits, dim=-1)
    # print(probas.shape)
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    # print("Output token IDs:\n", token_ids)
    # print("Target token IDs:\n", token_ids_to_text(targets[0].flatten(), tokenizer)) 
    # print("Output text:\n", token_ids_to_text(token_ids[0].flatten(), tokenizer))
    
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 1:", target_probas_1)
    
    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 2:", target_probas_2)
    
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    avg_log_proba = torch.mean(log_probas)
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(loss)