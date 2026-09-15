import torch
import tiktoken
from src.eval.loss import calc_loss_loader
from src.eval.loss import calc_loss_batch
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
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
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
    
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
        
        if global_step % eval_freq == 0:
            train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            track_tokens_seen.append(tokens_seen)
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{global_step}], Tokens"
                    f"Seen: {tokens_seen}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            generate_and_print_sample(
                model, tokenizer, device, start_context
            )
    return train_losses, val_losses, track_tokens_seen

if __name__ == "__main__":
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("mps") 

    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 0.004, weight_decay=0.1
    )
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        
    )