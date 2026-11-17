import torch
from src.model.gpt import GPTModel
from configs.gpt import GPT_CONFIG_124M
from src.model.generate import generate_text_simple, token_ids_to_text, text_to_token_ids
import tiktoken

# calculates the cross entropy loss of a given batch returned from training and validation loader.
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(start_dim=0, end_dim=1),
        target_batch.flatten()
    )
    return loss

# iterates over all batches in a given data loader, accumulating the loss for each batch and returning the average loss over all batches.
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader) # iterate over all batches if no num_batches is specified
    else:
        num_batches = min(num_batches, len(data_loader)) # reduces the number of batches to match the specified num_batches if it is less than the total number of batches in the data_loader
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


if __name__ == "__main__":
    from pathlib import Path
    from src.data.dataset import create_dataloader_v1

    torch.manual_seed(123)

    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")

    # 5.1.2 the worked example, two 3-token sequences and their targets
    inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                        [40,    1107, 588]])   #  "I really like"]
    targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                            [1107, 588, 11311  ]]) #  " really like chocolate"

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

    # 5.1.3 the same loss, but over the whole corpus split into train and val
    file_path = (Path(__file__).resolve().parents[2] / "data" / "raw" / "the-verdict.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    # total_characters = len(text_data)
    # total_tokens = len(tokenizer.encode(text_data))
    # print("Characters:", total_characters)
    # print("Tokens:", total_tokens)

    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    print(f"Train loss: {train_loss}")
    print(f"Validation loss: {val_loss}")