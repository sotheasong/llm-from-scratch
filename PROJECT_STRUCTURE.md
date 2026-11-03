# Project structure

A map of where every chapter's code lands, from chapter 1 through appendix E.

## The organising rule

Files are named after the concern, not the chapter. `src/model/attention.py` holds
all of chapter 3 because all of chapter 3 is attention. `src/eval/loss.py` holds
5.1.3 and gains chapter 6's classification loss later, because both are loss.

The chapter number lives in two places and nowhere else:

- a comment above each definition, `# 4.2 layer normalization`, followed by
  `# -> why` lines
- the notes in `notes/`, one file per chapter

This matters because the book revisits things. `generate_text_simple` arrives in
4.7 and gets replaced by `generate` in 5.3.3. `create_dataloader_v1` is written in
2.6 and not used in anger until 5.1.3. Chapter-named files would scatter that, and
you would end up with four copies of `MultiHeadAttention`.

## Target tree

```
llm from scratch/
├── configs/
│   ├── gpt.py                  GPT_CONFIG_124M, model_configs, BASE_CONFIG
│   ├── classify.py             ch. 6 fine-tuning config
│   └── instruct.py             ch. 7 fine-tuning config
├── data/                       gitignored
│   ├── raw/                    the-verdict.txt, sms_spam_collection.zip, instruction-data.json
│   └── clean/                  train.csv, validation.csv, test.csv
├── checkpoints/                gitignored
│   ├── gpt2/124M/              OpenAI weights (5.5)
│   ├── model.pth               5.4
│   ├── model_and_optimizer.pth 5.4
│   ├── review_classifier.pth   6.7
│   └── gpt2-medium355M-sft.pth 7.7
├── src/
│   ├── tokenization/
│   │   ├── simple.py           2.2-2.4  regex tokenizer
│   │   └── bpe.py              2.5      tiktoken BPE
│   ├── data/
│   │   ├── embeddings.py       2.7-2.8  token + positional embeddings
│   │   ├── dataset.py          2.6      GPTDatasetV1, create_dataloader_v1
│   │   ├── spam.py             6.2-6.3  SpamDataset, loaders
│   │   └── instructions.py     7.2-7.4  format_input, InstructionDataset, collate
│   ├── model/
│   │   ├── attention.py        3.3-3.6  all four attention passes
│   │   ├── layers.py           4.2-4.3  LayerNorm, GELU, FeedForward
│   │   ├── gpt.py              4.1-4.6  TransformerBlock, GPTModel
│   │   ├── generate.py         4.7, 5.1.1, 5.3  decoding
│   │   ├── classifier.py       6.5, 6.8 classification head
│   │   └── lora.py             E.4      LoRALayer, LinearWithLoRA
│   ├── pretrained/
│   │   ├── gpt_download.py     5.5      download_and_load_gpt2 (from the book repo)
│   │   └── load_openai.py      5.5      assign, load_weights_into_gpt
│   ├── train/
│   │   ├── pretrain.py         5.2, D.4 train_model_simple, train_model
│   │   ├── schedule.py         D.1-D.3  warmup, cosine decay, grad clipping
│   │   ├── checkpoint.py       5.4      save/load state dicts
│   │   ├── finetune_classifier.py  6.7
│   │   ├── finetune_instruct.py    7.6
│   │   └── finetune_lora.py        E.4
│   ├── eval/
│   │   ├── loss.py             5.1.2-5.1.3, 6.6  loss over batch and loader
│   │   ├── classify.py         6.6      calc_accuracy_loader
│   │   ├── ollama.py           7.8      query_model, generate_model_scores
│   │   └── plot.py             5.2, 6.7 plot_losses, plot_values
│   └── stats/                  scratch space for one-off numbers
├── scripts/                    runnable entry points, one job each
├── exercises/                  ch02.py … ch07.py, answers in appendix C
├── notes/                      one markdown file per chapter
├── figures/                    gitignored, written by the demo blocks
├── results/                    gitignored, generated JSON and scores
├── tests/
├── experiments.md              what you tried, what the loss did
└── README.md
```

## Conventions worth keeping

Every module ends with a `if __name__ == "__main__":` block that replays the
chapter's printed output. `python -m src.model.layers` prints the layer norm means
and saves the GELU plot. That is the thing that proves the module works, and it
costs nothing at import time. Keep doing this.

Paths resolve off the file, not the cwd. `Path(__file__).resolve().parents[2] /
"data" / "raw" / "the-verdict.txt"` works from anywhere. Relative paths like
`../../data/raw/...` break the moment you run the module from the project root.

`data/`, `figures/`, `checkpoints/` and `results/` stay out of git. The GPT-2 124M
weights alone are about 500 MB, and the 355M medium model is 1.4 GB.

Anything with a `Dummy` prefix or `_v1` suffix is a rung on the ladder, kept so the
notes have something to point at. `SelfAttention_v1`, `MultiHeadAttentionWrapper`,
`DummyTransformerBlock`, `custom_collate_draft_1`. Keep them, comment why they lost.

## Chapter by chapter

Status is as of the chapter 5 reset. Chapters 1 to 4 are done, chapter 5 is back
to a blank slate, chapters 6 onward are the forward plan.

### Chapter 1, understanding LLMs

No code. `notes/ch01-02.md` covers it. Done.

### Chapter 2, working with text data

| Section                                       | File                         | Status                         |
| --------------------------------------------- | ---------------------------- | ------------------------------ |
| 2.1 word embeddings                           | notes only                   | done                           |
| 2.2-2.4 tokenizing, token IDs, special tokens | `src/tokenization/simple.py` | done, `SimpleTokenizerV2` only |
| 2.5 byte pair encoding                        | `src/tokenization/bpe.py`    | to do                          |
| 2.6 sliding window                            | `src/data/dataset.py`        | done                           |
| 2.7 token embeddings                          | `src/data/embeddings.py`     | to do                          |
| 2.8 encoding word positions                   | `src/data/embeddings.py`     | to do                          |

2.5 is thin, it is a tiktoken call and a demo of how BPE handles the nonsense word
`Akwirw ier`. Worth its own file anyway so the notes have something to run.

2.7 and 2.8 end up as `tok_emb` and `pos_emb` inside `GPTModel`, so
`embeddings.py` exists only to hold the standalone demo. Low priority, but it is
the file that shows an embedding layer is a lookup table and nothing more.

### Chapter 3, coding attention mechanisms

All in `src/model/attention.py`, all done. `simple_self_attention` (3.3),
`SelfAttention_v1` and `_v2` (3.4), `CausalAttention` (3.5),
`MultiHeadAttentionWrapper` (3.6.1), `MultiHeadAttention` (3.6.2).

Nothing to add. One file holding four versions of the same idea is correct here,
because the whole point of the chapter is the diff between them.

### Chapter 4, implementing a GPT model

| Section                    | File                                                           | Status          |
| -------------------------- | -------------------------------------------------------------- | --------------- |
| 4.1 LLM architecture       | `src/model/gpt.py` (`DummyLayerNorm`, `DummyTransformerBlock`) | done            |
| 4.2 layer normalization    | `src/model/layers.py` (`LayerNorm`)                            | done            |
| 4.3 feed forward with GELU | `src/model/layers.py` (`GELU`, `FeedForward`)                  | done            |
| 4.4 shortcut connections   | `src/model/gpt.py` (`ExampleDeepNeuralNetwork`)                | done, misplaced |
| 4.5 transformer block      | `src/model/gpt.py` (`TransformerBlock`)                        | done            |
| 4.6 the GPT model          | `src/model/gpt.py` (`GPTModel`)                                | done            |
| 4.7 generating text        | `src/model/generate.py` (`generate_text_simple`)               | done            |

One move to consider. `ExampleDeepNeuralNetwork` and `print_gradients` are a
gradient-vanishing demo, not part of the model, and `gpt.py` is imported by
almost everything downstream. They would sit better in
`src/model/shortcut_demo.py`. Not urgent, just noise in an otherwise clean file.

### Chapter 5, pretraining on unlabeled data

| Section                          | File                                                                                          | Status |
| -------------------------------- | --------------------------------------------------------------------------------------------- | ------ |
| 5.1.1 using GPT to generate text | `src/model/generate.py` (`text_to_token_ids`, `token_ids_to_text`)                            | to do  |
| 5.1.2 text generation loss       | `src/eval/loss.py` demo block                                                                 | to do  |
| 5.1.3 train and val losses       | `src/eval/loss.py` (`calc_loss_batch`, `calc_loss_loader`)                                    | to do  |
| 5.2 training an LLM              | `src/train/pretrain.py` (`train_model_simple`, `evaluate_model`, `generate_and_print_sample`) | to do  |
| 5.2 loss curves                  | `src/eval/plot.py` (`plot_losses`)                                                            | to do  |
| 5.3.1 temperature scaling        | `src/model/generate.py` (`print_sampled_tokens`, `softmax_with_temperature`)                  | to do  |
| 5.3.2 top-k sampling             | `src/model/generate.py`                                                                       | to do  |
| 5.3.3 modified generate          | `src/model/generate.py` (`generate`)                                                          | to do  |
| 5.4 saving and loading weights   | `src/train/checkpoint.py`                                                                     | to do  |
| 5.5 loading OpenAI weights       | `src/pretrained/gpt_download.py`, `src/pretrained/load_openai.py`                             | to do  |

Starting order, and the one config change to make first.

`configs/gpt.py` is back to `context_length` 1024, the chapter 4 value. Section
5.1.3 is where the book drops it to 256 so training fits on a laptop. Make that
edit when you reach 5.1.3, not before, so the reason for it lands with the
section that explains it.

Then 5.1.1 and 5.1.2 in `generate.py` and `loss.py`, 5.1.3's two loader functions,
5.2's training loop, and `plot_losses` last. `plot.py` is the piece that shows the
train and val curves diverging, which is the whole observation of 5.2. It writes
`figures/pretrain_loss.png`.

5.3 goes into `generate.py` alongside `generate_text_simple`, which it supersedes.
Keep both. The comment in `generate_text_simple` already says why softmax is kept
before the argmax, that note pays off here. The new `generate` takes `top_k`,
`temperature` and `eos_id`. Chapters 6 and 7 both call it.

5.5 wants its own package because three later chapters pull from it.
`gpt_download.py` is the one file in this project you copy rather than write, it is
TensorFlow checkpoint plumbing from the book's repo and teaches nothing. Writing
`load_weights_into_gpt` yourself is the real exercise, it is a shape-by-shape
audit of every parameter in the model you just built. `configs/gpt.py` grows
`model_configs` and a `BASE_CONFIG` with `context_length` 1024 and `qkv_bias`
True, which is what the OpenAI weights expect.

### Chapter 6, fine-tuning for classification

| Section                                  | File                                                                      |
| ---------------------------------------- | ------------------------------------------------------------------------- |
| 6.1 categories of fine-tuning            | notes only                                                                |
| 6.2 preparing the dataset                | `scripts/prepare_spam_data.py` → `data/clean/{train,validation,test}.csv` |
| 6.3 data loaders                         | `src/data/spam.py` (`SpamDataset`)                                        |
| 6.4 initializing with pretrained weights | reuses `src/pretrained/load_openai.py`, config in `configs/classify.py`   |
| 6.5 classification head                  | `src/model/classifier.py`                                                 |
| 6.6 loss and accuracy                    | `src/eval/classify.py` (`calc_accuracy_loader`), `src/eval/loss.py`       |
| 6.7 fine-tuning                          | `src/train/finetune_classifier.py` (`train_classifier_simple`)            |
| 6.7 plots                                | `src/eval/plot.py` (`plot_values`)                                        |
| 6.8 classifying new texts                | `src/model/classifier.py` (`classify_review`)                             |

The download, balance and split (listings 6.1 to 6.3) belong in a script, not a
module. They run once, they write CSVs, and nothing imports them afterwards.
`src/data/spam.py` then just reads `data/clean/*.csv`.

`src/model/classifier.py` is where the chapter's one surprising move lives. You
swap `out_head` for a 2-output `nn.Linear`, freeze everything, then unfreeze the
last transformer block, the final norm and the new head. The file should hold that
as a function so the "which layers to unfreeze" experiment in 6.6 is one argument.

Loss is nearly the chapter 5 function with `logits[:, -1, :]` instead of all
positions. Put the classification variant in `src/eval/loss.py` next to
`calc_loss_batch` rather than a new file, and comment the one-line difference.
`calc_accuracy_loader` is genuinely new, so `src/eval/classify.py`.

### Chapter 7, fine-tuning to follow instructions

| Section                      | File                                                                                |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| 7.1 intro                    | notes only                                                                          |
| 7.2 dataset, prompt format   | `scripts/download_instruction_data.py`, `src/data/instructions.py` (`format_input`) |
| 7.3 training batches         | `src/data/instructions.py` (`custom_collate_fn`)                                    |
| 7.4 data loaders             | `src/data/instructions.py` (`InstructionDataset`)                                   |
| 7.5 loading a pretrained LLM | reuses `src/pretrained/`, `configs/instruct.py` with gpt2-medium                    |
| 7.6 fine-tuning              | `src/train/finetune_instruct.py`, reuses `train_model_simple`                       |
| 7.7 extracting responses     | `scripts/generate_responses.py` → `results/instruction-data-with-response.json`     |
| 7.8 evaluating               | `src/eval/ollama.py`                                                                |
| 7.9 conclusions              | notes only                                                                          |

`custom_collate_fn` is the hardest function in the book to get right and deserves
all three drafts kept in the file. Draft 1 pads, draft 2 shifts targets, the final
version masks padding with `-100` and truncates to `allowed_max_length`. The
`-100` is not arbitrary, it is PyTorch's `cross_entropy` default `ignore_index`,
which is worth a `# ->` line.

`src/eval/ollama.py` talks to a local Llama 3 over HTTP to score responses. It is
the only module in the project with a network dependency at eval time, so keep it
isolated and let `check_if_running` fail loudly.

355M at 1.4 GB on CPU is about 16 minutes for two epochs on an M-series Mac. The
config should make switching to `gpt2-small (124M)` a one-line change, which is
why `CHOOSE_MODEL` is a config constant and not a literal.

### Appendix A, introduction to PyTorch

Skippable. You are past chapter 5, so autograd, `nn.Module` and training loops are
already familiar. If anything, `notes/appendix-a.md` for A.9 on device handling,
since this project runs on MPS and the book assumes CUDA.

### Appendix B, references

No code.

### Appendix C, exercise solutions

Suggests a top-level `exercises/` directory, `ch02.py` through `ch07.py`. The
exercises are not throwaway. 3.2 asks for 2-dimensional embedding vectors from a
multi-head setup, 4.1 asks you to count parameters in feed forward against
attention, 5.2 asks what temperature and top-k combination suits a given task.
Those answers are the check that the notes are not just transcription.

### Appendix D, bells and whistles for the training loop

| Section                        | File                                              |
| ------------------------------ | ------------------------------------------------- |
| D.1 learning rate warmup       | `src/train/schedule.py`                           |
| D.2 cosine decay               | `src/train/schedule.py`                           |
| D.3 gradient clipping          | `src/train/schedule.py` (`find_highest_gradient`) |
| D.4 modified training function | `src/train/pretrain.py` (`train_model`)           |

Worth doing right after 5.5 rather than at the end. `train_model_simple` is
already written, and `train_model` is the same loop with three additions. Keeping
both in `pretrain.py` makes the diff readable, which is the point of the appendix.

### Appendix E, parameter-efficient fine-tuning with LoRA

| Section                   | File                                                                            |
| ------------------------- | ------------------------------------------------------------------------------- |
| E.1 intro                 | notes only                                                                      |
| E.2-E.3 dataset and model | reuses chapter 6's `src/data/spam.py` and `configs/classify.py`                 |
| E.4 LoRA                  | `src/model/lora.py` (`LoRALayer`, `LinearWithLoRA`, `replace_linear_with_lora`) |
| E.4 training              | `src/train/finetune_lora.py`                                                    |

E reruns chapter 6 with LoRA swapped in, so it should import chapter 6's data code
unchanged. If it cannot, chapter 6's data code was written wrong.

## Gaps in the repo right now

Four things, roughly in order of how much they will bite.

Two bugs from the first chapter 5 attempt, worth not writing again. `loss.py` built
its data path as `Path("../../data/raw/the-verdict.txt")`, which only resolves if
you run the file from inside `src/eval/`. Use the `parents[2]` idiom the other
modules use. And its device line read
`torch.device("mps" if torch.cuda.is_available() else "cpu")`, which asks CUDA
whether it exists and then uses MPS. On this Mac `cuda.is_available()` is False, so
every run that looked like it was on the GPU was on the CPU. `pretrain.py` had it
right with `mps.is_available()`.

The chapter 5 code also carried no `# 5.x` section comments, the only code in the
repo missing them. Chapters 2, 3 and 4 all have the `# <section> <what>` line plus
`# -> why` follow-ups. Restore that on the rewrite.

No `__init__.py` anywhere under `src/`. Imports work because the project root
lands on `sys.path`, but `setuptools` found no packages, so the editable install
is doing nothing. `llm_from_scratch.egg-info/SOURCES.txt` lists loose `.py` files
rather than packages. Either add the `__init__.py` files or switch `pyproject.toml`
to `packages.find` with `namespaces = true`.

`README.md` and `experiments.md` are both empty. The second one is the more
valuable of the two. The comment in `pretrain.py` about `lr = 0.004` being ten
times too large and stalling the loss near 5.5 is exactly what belongs there, and
right now that finding only exists as a code comment.

Empty directories with no owner yet: `scripts/`, `tests/`, `tokenizers/`,
`results/`, `src/stats/`, `data/clean/`. Fine to leave, but `tokenizers/` has no
job in the plan above, since BPE comes from tiktoken and nothing gets trained or
saved. Drop it unless you plan the Khmer tokenizer that `data/raw/km.txt` hints at.

## The one open question

`data/raw/km.txt` is Khmer text and the book never asks for it. If the plan is to
retrain the tokenizer or pretrain on Khmer, that changes things. Chapter 2's BPE
section becomes real work instead of a tiktoken call, `tokenizers/` gets a job,
and `vocab_size` in `configs/gpt.py` stops being 50257. Say the word and I will
add that track to the plan.
