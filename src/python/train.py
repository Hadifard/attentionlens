# ==============================================================================
# AttentionLens - Explainable AI for Transformers (PyTorch)
# Author  : Hadi Sarhangi Fard  |  GitHub: @Hadifard
# File    : train.py
# Purpose : Trains the TinyGPT and logs loss, accuracy and per-head circuit scores over time.
# License : MIT (see LICENSE)
# ==============================================================================
import json, sys, time
import torch
import torch.nn.functional as F
from model import TinyGPT
from data import make_batch, repeat_accuracy


def head_scores(att, Ls):
    """Per-head mechanistic scores, each of shape (n_layers, n_heads), averaged over batch and queries.

    For a query at position t >= L (second period or later) of a sequence with period L:
    induction : attention to key t-L+1, the token AFTER the previous occurrence of the current token
    previous  : attention to the immediately preceding token (t-1)
    duplicate : attention to key t-L, the previous occurrence of the current token itself
    """
    B, nl, nh, T, _ = att.shape
    t = torch.arange(T)[None, :]
    valid = (t >= Ls[:, None]).float()                                          # (B, T)

    def offset_score(off):
        idx = (t - Ls[:, None] + off).clamp(0, T - 1)
        g = att.gather(-1, idx[:, None, None, :, None].expand(B, nl, nh, T, 1)).squeeze(-1)
        return (g * valid[:, None, None, :]).sum((0, -1)) / valid.sum()

    prev = att[..., torch.arange(1, T), torch.arange(0, T - 1)].mean((0, -1))
    return offset_score(1), prev, offset_score(0)


def train(steps=4000, batch=64, lr=2e-3, seed=0, out="../../results", log_every=20):
    torch.manual_seed(seed)
    model = TinyGPT()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    g = torch.Generator().manual_seed(123)
    ev_x, ev_y, ev_m, ev_L = make_batch(256, generator=g)                        # fixed evaluation batch
    hist = {k: [] for k in ["step", "loss", "acc", "induction_max", "previous_max", "ind_L0", "ind_L1"]}
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y, m, _ = make_batch(batch)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[m], y[m])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                lg, att = model(ev_x)
                ind, prev, _ = head_scores(att, ev_L)
            hist["step"].append(step); hist["loss"].append(loss.item())
            hist["acc"].append(repeat_accuracy(lg, ev_y, ev_m))
            hist["induction_max"].append(ind.max().item()); hist["previous_max"].append(prev.max().item())
            hist["ind_L0"].append(ind[0].max().item()); hist["ind_L1"].append(ind[1].max().item())
            if step % (log_every * 10) == 0:
                print(f"step {step:5d} loss {loss.item():.3f} acc {hist['acc'][-1]:.3f} "
                      f"ind {hist['induction_max'][-1]:.2f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"state": model.state_dict(), "cfg": model.cfg}, f"{out}/model.pt")
    json.dump(hist, open(f"{out}/history.json", "w"))
    return model, hist


if __name__ == "__main__":
    train(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 4000)
