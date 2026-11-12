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