from pathlib import Path

import torch

from src.model.gpt import GPTModel

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"
# gitignored; weights alone are about 650 MB, weights plus optimizer about 2 GB


# 5.4 save the weights


# 5.4 load the weights back into a fresh model


# 5.4 save the weights and the optimizer together


# 5.4 load both back, ready to carry on training


if __name__ == "__main__":
    import tiktoken

    from configs.gpt import GPT_CONFIG_124M
    from src.data.dataset import create_dataloader_v1
    from src.train.pretrain import train_model_simple

    device = torch.device("cpu")
    # cpu for the same reason as 5.2, MPS drifts the weights off the book's
    tokenizer = tiktoken.get_encoding("gpt2")

    text_data = (Path(__file__).resolve().parents[2]
                 / "data" / "raw" / "the-verdict.txt").read_text(encoding="utf-8")
    split_idx = int(0.90 * len(text_data))
    train_loader = create_dataloader_v1(
        text_data[:split_idx], batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True, shuffle=True, num_workers=0
    )
    val_loader = create_dataloader_v1(
        text_data[split_idx:], batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False, shuffle=False, num_workers=0
    )

    # the 5.2 run, so there is a trained model and a warmed-up optimizer worth saving
    # -> about 70s on cpu
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=10, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    # 5.4 from here
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CHECKPOINT_DIR / "model.pth"
    both_path = CHECKPOINT_DIR / "model_and_optimizer.pth"
    # the book writes bare "model.pth", which lands in whatever directory you
    # ran from, and the project root is not gitignored the way checkpoints/ is

    torch.save(model.state_dict(), model_path)
    print("saved weights to", model_path)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, both_path)
    print("saved weights and optimizer to", both_path)

    checkpoint = torch.load(both_path, map_location=device)
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    # the fresh model is built on cpu, so it has to be moved before the
    # optimizer state goes in, otherwise the two sit on different devices
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train()
    print("restored, optimizer step count:",
          optimizer.state_dict()["state"][0]["step"])
