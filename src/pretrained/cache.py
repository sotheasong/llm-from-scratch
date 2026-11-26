import json
from pathlib import Path

import numpy as np

DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "checkpoints" / "gpt2"


# the OpenAI params arrive as a nested dict with a list of 12 blocks, and
# np.savez only takes flat keyword arrays, so the nesting goes into the key
# -> params["blocks"][0]["attn"]["c_attn"]["w"] becomes "blocks.0/attn/c_attn/w"
def _flatten(d, prefix=""):
    out = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, name + "/"))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                out.update(_flatten(item, f"{name}.{i}/"))
        else:
            out[name] = value
    return out


def _unflatten(flat, n_layer):
    params = {"blocks": [{} for _ in range(n_layer)]}
    for key, array in flat.items():
        parts = key.split("/")
        if parts[0].startswith("blocks."):
            target = params["blocks"][int(parts[0].split(".")[1])]
            parts = parts[1:]
        else:
            target = params
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = array
    return params


# settings and params, read from a local npz when one is there
# -> a cache hit skips the tensorflow import and the checkpoint parse, which is
#    where the ~4s goes; that import is inside the miss branch for that reason
# -> the npz is about 475 MB on top of the 498 MB checkpoint, both gitignored
def load_gpt2_params(model_size="124M", models_dir=None, refresh=False):
    models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
    cache = models_dir / model_size / "params.npz"
    meta = models_dir / model_size / "settings.json"

    if not refresh and cache.exists() and meta.exists():
        try:
            settings = json.loads(meta.read_text())
            with np.load(cache) as z:
                flat = {k: z[k] for k in z.files}
            return settings, _unflatten(flat, settings["n_layer"])
        except (OSError, ValueError, KeyError):
            pass
            # a truncated or stale npz just falls through and gets rebuilt

    from src.pretrained.gpt_download import download_and_load_gpt2

    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir=str(models_dir)
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **_flatten(params))
    meta.write_text(json.dumps(settings))
    return settings, params


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()
    settings, params = load_gpt2_params()
    print(f"{time.perf_counter() - t0:.2f}s")
    print("settings:", settings)
    print("blocks:", len(params["blocks"]))
