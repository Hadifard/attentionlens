# ==============================================================================
# AttentionLens - Explainable AI for Transformers (PyTorch)
# Author  : Hadi Sarhangi Fard  |  GitHub: @Hadifard
# ==============================================================================
import json, os, torch
from model import TinyGPT
from data import make_batch, repeat_accuracy
from train import head_scores
import visualize as viz

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RES, IMG = os.path.join(ROOT, "results"), os.path.join(ROOT, "docs", "images")


def load_model(path=os.path.join(RES, "model.pt")):
    ck = torch.load(path)
    model = TinyGPT(**ck["cfg"])
    model.load_state_dict(ck["state"])
    return model.eval()


@torch.no_grad()
def accuracy(model, x, y, m, head_mask=None):
    return repeat_accuracy(model(x, head_mask)[0], y, m)


@torch.no_grad()
def attention_entropy(att, Ls):
    """Mean Shannon entropy (nats) of every head's attention distribution on predictable positions."""
    ent = -(att.clamp_min(1e-9).log() * att).sum(-1)                       # (B, L, H, T)
    valid = (torch.arange(att.size(-1))[None, :] >= Ls[:, None]).float()[:, None, None, :]
    return (ent * valid).sum((0, -1)) / valid.sum()


@torch.no_grad()
def ablation_study(model, x, y, m):
    """Zero-ablate every head (and every whole layer) and measure accuracy on predictable positions."""
    L, H = model.cfg["n_layers"], model.cfg["n_heads"]
    single, layer = torch.zeros(L, H), []
    for l in range(L):
        for h in range(H):
            hm = torch.ones(L, H); hm[l, h] = 0
            single[l, h] = accuracy(model, x, y, m, hm)
        hm = torch.ones(L, H); hm[l] = 0
        layer.append(accuracy(model, x, y, m, hm))
    return single, layer


@torch.no_grad()
def composition_study(model, x, Ls, ind_head):
    """Induction score of the chosen head when each layer-0 head is ablated (circuit composition)."""
    L, H = model.cfg["n_layers"], model.cfg["n_heads"]
    out = []
    for h in range(H):
        hm = torch.ones(L, H); hm[0, h] = 0
        _, att = model(x, hm)
        out.append(head_scores(att, Ls)[0][ind_head[0], ind_head[1]].item())
    return out


def main():
    model = load_model()
    hist = json.load(open(os.path.join(RES, "history.json")))
    x, y, m, Ls = make_batch(1024, generator=torch.Generator().manual_seed(2024))   # held-out evaluation set
    with torch.no_grad():
        logits, att = model(x)
    ind, prev, dup = head_scores(att, Ls)
    ent = attention_entropy(att, Ls)
    base = repeat_accuracy(logits, y, m)
    single, layer = ablation_study(model, x, y, m)
    ih = divmod(int(ind.argmax()), ind.size(1))                     # (layer, head) of best induction head
    ph = divmod(int(prev.argmax()), prev.size(1))                   # best previous-token head
    comp = composition_study(model, x, Ls, ih)
    pheads = [h for h in range(prev.size(1)) if prev[0, h] > 0.3]      # layer-0 previous-token heads
    hm = torch.ones(2, 4); hm[0, pheads] = 0
    both_prev_ablated = accuracy(model, x, y, m, hm)                   # remove ALL previous-token heads
    hm = torch.zeros(2, 4); hm[0] = 1; hm[ih] = 1
    only_best_induction = accuracy(model, x, y, m, hm)                 # keep layer 0 + a single layer-1 head
    metrics = dict(baseline_accuracy=base, induction_head=ih, previous_token_head=ph,
                   induction_score=ind.tolist(), previous_score=prev.tolist(), duplicate_score=dup.tolist(),
                   entropy=ent.tolist(), ablation_single=single.tolist(), ablation_layer=layer,
                   induction_score_when_L0_head_ablated=comp,
                   params=sum(p.numel() for p in model.parameters()),
                   all_previous_token_heads_ablated=both_prev_ablated, only_best_induction_head_kept=only_best_induction)
    json.dump(metrics, open(os.path.join(RES, "metrics.json"), "w"), indent=2)
    xs, ys, _, Lx = make_batch(1, L=12, generator=torch.Generator().manual_seed(5))   # display sequence (period 12)
    with torch.no_grad():
        _, a = model(xs)
    a = a[0]
    viz.fig_training(hist, IMG)
    viz.fig_attention_grid(a, xs[0], ih, 12, IMG)
    viz.fig_head_scores(ind, prev, dup, ent, IMG)
    viz.fig_ablation(base, single, layer, comp, ih, IMG)
    viz.fig_head_view(a[ih[0], ih[1]], xs[0], ih, 12, IMG)
    viz.fig_architecture(model.cfg, pheads, ih, IMG)
    viz.fig_banner(a[ih[0], ih[1]], IMG)
    print(json.dumps({k: metrics[k] for k in ["baseline_accuracy", "induction_head", "previous_token_head", "params",
                      "induction_score", "previous_score", "ablation_single", "ablation_layer",
                      "induction_score_when_L0_head_ablated"]}))


if __name__ == "__main__":
    main()
