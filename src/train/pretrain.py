from src.eval.loss import calc_loss_batch, calc_loss_loader
from src.model.generate import generate_text_simple, token_ids_to_text, text_to_token_ids
import torch

# 5.2 the train and val loss at one point in training
# -> eval() turns dropout off so the number is stable, train() puts it back
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# 5.2 the qualitative counterpart to the loss, one sample per epoch
# -> context_size is read off the model, pos_emb has one row per position
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

# 5.2 the training loop
# -> global_step starts at -1 so the first increment gives 0, firing a baseline eval
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses = []
    val_losses = []
    track_tokens_seen = []
    tokens_seen = 0
    global_step = -1
    
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen

if __name__ == "__main__":
    import tiktoken
    from configs.gpt import GPT_CONFIG_124M
    from src.model.gpt import GPTModel
    from src.data.dataset import create_dataloader_v1
    from pathlib import Path
    
    torch.manual_seed(123)
    
    file_path = (Path(__file__).resolve().parents[2] / "data" / "raw" / "the-verdict.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    train_ratio = 0.90
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

    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("device:", device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    # 0.004 is ten times too large, the loss stalls near 5.5 and never learns English
    # .parameters() method returns all trainable weight parameters of model
    tokenizer = tiktoken.get_encoding("gpt2")
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader, 
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )
