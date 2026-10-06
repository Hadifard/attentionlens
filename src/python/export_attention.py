# ==============================================================================
# AttentionLens - Explainable AI for Transformers (PyTorch)
# Author  : Hadi Sarhangi Fard  |  GitHub: @Hadifard
# File    : export_attention.py
# Purpose : Exports real attention weights of the trained model to JSON for the browser viewer.
# License : MIT (see LICENSE)
# ==============================================================================
import json, os, torch
from analyze import load_model, RES
from data import make_batch


def main(n_decimals=3):
    model = load_model()
    x, _, _, _ = make_batch(1, L=12, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        _, att = model(x)
    data = {"name": "TinyGPT (trained in PyTorch) on the repeated-random-sequence task | by Hadi Sarhangi Fard",
            "tokens": [str(int(t)) for t in x[0]],
            "attention": [[[[round(v, n_decimals) for v in row] for row in head.tolist()] for head in layer] for layer in att[0]]}
    json.dump(data, open(os.path.join(RES, "attention_sample.json"), "w"))
    print("saved attention_sample.json")


if __name__ == "__main__":
    main()
