import numpy as np
import torch

from src.model.gpt import GPTModel


# 5.5 check the shapes line up, then hand back the values as a Parameter


# 5.5 walk the OpenAI params dict and copy every tensor into a GPTModel


if __name__ == "__main__":
    import tiktoken

    from configs.gpt import GPT_CONFIG_124M
    from src.model.generate import generate, text_to_token_ids, token_ids_to_text
    from src.pretrained.cache import load_gpt2_params

    device = torch.device("cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    settings, params = load_gpt2_params("124M")
    # reads checkpoints/gpt2/124M/params.npz when it is there, 0.08s
    # -> first call builds that cache off the TensorFlow checkpoint, ~4.5s
    # -> pass refresh=True to rebuild it from the checkpoint
    # -> settings is hparams.json: n_vocab 50257, n_ctx 1024, n_embd 768,
    #    n_head 12, n_layer 12
    print("settings:", settings)

    # what the mapping has to cover, shapes confirmed by reading the checkpoint:
    #
    #   params["wte"]                      (50257, 768)  token embedding
    #   params["wpe"]                      (1024, 768)   positional embedding
    #   params["g"] / params["b"]          (768,)        final layer norm
    #   params["blocks"][i]                one dict per block, 12 of them
    #     attn/c_attn/w  (768, 2304)   q, k and v stacked in one matrix
    #     attn/c_attn/b  (2304,)
    #     attn/c_proj/w  (768, 768)    out_proj
    #     attn/c_proj/b  (768,)
    #     ln_1/g, ln_1/b (768,)        norm1
    #     ln_2/g, ln_2/b (768,)        norm2
    #     mlp/c_fc/w     (768, 3072)   feed forward in
    #     mlp/c_fc/b     (3072,)
    #     mlp/c_proj/w   (3072, 768)   feed forward out
    #     mlp/c_proj/b   (768,)
    #
    # two things to watch:
    # -> c_attn packs query, key and value into one (768, 2304) matrix, so it
    #    has to be split three ways
    # -> OpenAI stores weights transposed relative to nn.Linear

    print("Parameter dictionary keys:", params.keys())
