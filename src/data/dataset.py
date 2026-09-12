import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


# 2.6 sliding window over the token stream
# -> the target is the input shifted along by one
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# 2.6 the loader that batches the dataset, needed from ch. 5 onwards
# -> the tokenizer is built inside, so callers pass raw text, not token IDs
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                         shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        # drops a final short batch, which would otherwise spike the loss
        num_workers=num_workers,
        # 0 keeps loading on the main process, fine for a corpus this small
    )
    # stride == max_length means no overlap, every token used exactly once
    # -> a smaller stride reuses tokens across examples and risks overfitting


if __name__ == "__main__":
    from pathlib import Path

    raw_text = (Path(__file__).resolve().parents[2] / "data" / "raw" / "the-verdict.txt").read_text()

    loader = create_dataloader_v1(raw_text, batch_size=8, max_length=4,
                                  stride=4, shuffle=False)
    inputs, targets = next(iter(loader))
    print("inputs shape:", inputs.shape)
    print("inputs[0] :", inputs[0].tolist())
    print("targets[0]:", targets[0].tolist())
    # same row shifted one token left, which is the whole training signal
    print("batches in the loader:", len(loader))
