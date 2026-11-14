import torch
from src.model.gpt import GPTModel
from configs.gpt import GPT_CONFIG_124M
from src.model.generate import generate_text_simple, token_ids_to_text, text_to_token_ids
import tiktoken

if __name__ == "__main__":
    from configs.gpt import GPT_CONFIG_124M
    from src.model.gpt import GPTModel

    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    # eval

    inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                        [40,    1107, 588]])   #  "I really like"]
    targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                            [1107, 588, 11311  ]]) #  " really like cohocolate"[]

    with torch.no_grad():
        logits = model(inputs)
    probas = torch.softmax(logits, dim=-1)
    print("probas.shape:", probas.shape)
    # outputs: probas.shape: torch.Size([2, 3, 50257])
    # first dimension is batch size, 
    # second is number of tokens in each input (row)
    # third is embedding dimension, which is the number of possible tokens in the vocabulary (50257 for GPT-2)

    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("token_ids", token_ids)
    
    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Predictions batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

    text_idx = 0
    target_probas_1 = probas[text_idx, [0,1,2], targets[text_idx]]
    print(f"Target probas batch 1: {target_probas_1}")

    text_idx = 1
    target_probas_2 = probas[text_idx, [0,1,2], targets[text_idx]]
    print(f"Target probas batch 2: {target_probas_2}")
    
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    neg_log_probas = -torch.mean(log_probas)
    print(f"Negative log probabilities: {neg_log_probas}")
    
    print("Logits shape:", logits.shape)
    print("Targets shape:", targets.shape)
    
    logits_flat = logits.flatten(start_dim=0, end_dim=1)
    targets_flat = targets.flatten()
    print("Logits flat shape:", logits_flat.shape)
    print("Targets flat shape:", targets_flat.shape)
    
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(f"Cross entropy loss: {loss}")