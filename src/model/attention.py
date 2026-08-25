import re
import urllib.request
from src.tokenization.bpe import SimpleTokenizerV2, GPTDatasetV1
import tiktoken
import torch
from torch.utils.data import DataLoader

torch.manual_seed(123)

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader

with open("data/raw/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer= tiktoken.get_encoding("gpt2")
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=4,
    stride=4,
    shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

input_ids = torch.tensor([2,3,5,1])
vocab_size = 50257 
output_dim = 256
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4
dataloader 