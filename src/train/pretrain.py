import torch
import tiktoken
from pathlib import Path
from src.eval.loss import calc_loss_loader
from src.eval.loss import calc_loss_batch
from src.data.dataset import create_dataloader_v1
from src.model.generate import text_to_token_ids, token_ids_to_text, generate_text_simple
from configs.gpt import GPT_CONFIG_124M
from src.model.gpt import GPTModel

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    # back to train mode, otherwise dropout stays off for the rest of the run
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    # read the context length off the model, pos_emb has one row per position
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model = model,
            idx = encoded,
            max_new_tokens = 50,
            context_size = context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer) 
    print(decoded_text.replace("\n", " "))
    model.train()

def train_model_simple(model, train_loader, val_loader, 
                       optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses = []
    val_losses = []
    track_tokens_seen = []
    tokens_seen = 0
    global_step = -1
    # starts at -1 so the first increment gives 0, which fires a baseline eval
    
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            # PyTorch accumulates into .grad, so last batch's gradients have to go
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            # writes .grad on every parameter, changes no weights
            optimizer.step()
            # reads .grad and writes the new weights
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                # inside the batch loop, not the epoch loop
                # -> at the epoch level this only fires on steps 8, 17, 26, ...
                #    and 8 of 10 epochs would silently print nothing
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
        # one sample per epoch, the qualitative counterpart to the loss
    return train_losses, val_losses, track_tokens_seen

if __name__ == "__main__":
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    file_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "the-verdict.txt"
    text_data = file_path.read_text(encoding="utf-8")

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 0.0004, weight_decay=0.1
    )
    # 0.004 is ten times too large, the loss stalls around 5.5 and never learns English
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )
