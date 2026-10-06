# ==============================================================================
# AttentionLens - Explainable AI for Transformers (PyTorch)
# Author  : Hadi Sarhangi Fard  |  GitHub: @Hadifard
# File    : visualize.py
# Purpose : Matplotlib figures used in the README (training curves, attention maps, ablations).
# License : MIT (see LICENSE)
# ==============================================================================
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C = ["#e6550d", "#3182bd", "#31a354", "#9467bd"]
SIG = "AttentionLens  |  Hadi Sarhangi Fard (@Hadifard)"


def _save(fig, out, name):
    fig.text(0.995, -0.07, SIG, ha="right", va="bottom", fontsize=7, color="gray")
    fig.savefig(os.path.join(out, name), dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_training(h, out):
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].semilogy(h["step"], h["loss"], color=C[0]); ax[0].set(title="Training loss (predictable positions)", xlabel="step")
    ax[1].plot(h["step"], h["acc"], color=C[2]); ax[1].set(title="Accuracy (predictable positions)", xlabel="step", ylim=(0, 1.02))
    ax[2].plot(h["step"], h["previous_max"], color=C[1], label="best previous-token score (any head)")
    ax[2].plot(h["step"], h["ind_L1"], color=C[0], label="best induction score (layer 1)")
    ax[2].plot(h["step"], h["ind_L0"], color=C[3], ls="--", label="best induction score (layer 0)")
    ax[2].set(title="Circuit formation", xlabel="step", ylim=(0, 1.02)); ax[2].legend(fontsize=7)
    for a in ax: a.set_xscale("log") if False else None; a.grid(alpha=.3)
    for a in ax: a.set_xlim(0, 800)
    _save(fig, out, "training_curves.png")


def fig_attention_grid(att, tokens, ih, L0, out):
    L, H = att.shape[:2]
    fig, ax = plt.subplots(L, H, figsize=(3 * H, 3 * L))
    for l in range(L):
        for h in range(H):
            a = ax[l, h]
            a.imshow(att[l, h].numpy(), cmap="Blues", vmin=0, vmax=1)
            a.axhline(L0 - .5, color="red", lw=.6, ls="--"); a.axvline(L0 - .5, color="red", lw=.6, ls="--")
            a.set(xticks=[], yticks=[]); a.set_title(f"L{l} H{h}" + ("  (induction)" if (l, h) == tuple(ih) else ""), fontsize=10,
                                                 color=C[0] if (l, h) == tuple(ih) else "black")
            if h == 0: a.set_ylabel("query position")
            if l == L - 1: a.set_xlabel("key position")
    fig.suptitle("Attention matrices of all heads (one test sequence; dashed = start of the second period)", y=1.0)
    _save(fig, out, "attention_grid.png")


def fig_head_scores(ind, prev, dup, ent, out):
    fig, ax = plt.subplots(1, 4, figsize=(14, 2.8))
    for a, m, t in zip(ax, [prev, dup, ind, ent], ["Previous-token score", "Duplicate-token score", "Induction score", "Attention entropy (nats)"]):
        im = a.imshow(m.numpy(), cmap="viridis"); a.set_title(t, fontsize=10)
        a.set(xticks=range(m.size(1)), yticks=range(m.size(0)), xlabel="head", ylabel="layer")
        for (i, j), v in np.ndenumerate(m.numpy()): a.text(j, i, f"{v:.2f}", ha="center", va="center", color="w", fontsize=9)
    _save(fig, out, "head_scores.png")


def fig_ablation(base, single, layer, comp, ih, out):
    L, H = single.shape
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    names = [f"L{l}H{h}" for l in range(L) for h in range(H)] + [f"all L{l}" for l in range(L)]
    vals = single.flatten().tolist() + list(layer)
    cols = [C[0] if (i // H, i % H) == tuple(ih) else C[1] for i in range(L * H)] + ["gray"] * L
    ax[0].bar(names, vals, color=cols); ax[0].axhline(base, color="k", ls="--", lw=.8, label=f"baseline {base:.3f}")
    ax[0].set(title="Zero-ablation: accuracy (predictable positions)", ylim=(0, 1.3)); ax[0].tick_params(axis="x", rotation=60); ax[0].legend(fontsize=8, loc="upper right")
    ax[1].bar([f"abl. L0H{h}" for h in range(len(comp))], comp, color=C[3])
    ax[1].set(title=f"Induction score of L{ih[0]}H{ih[1]} when a layer-0 head is ablated", ylim=(0, 1.05)); ax[1].tick_params(axis="x", rotation=30)
    for a in ax: a.grid(axis="y", alpha=.3)
    _save(fig, out, "ablation.png")


def fig_head_view(att, tokens, ih, L0, out):
    n = att.size(0); fig, ax = plt.subplots(figsize=(6.5, 10))
    for q in range(n):
        for k in range(q + 1):
            w = float(att[q, k])
            if w > .04: ax.plot([0, 1], [-q, -k], color=C[0], lw=.5 + 4 * w, alpha=min(1, .15 + w))
    for i in range(n):
        ax.text(-.03, -i, f"{i}: tok {int(tokens[i])}", ha="right", va="center", fontsize=8, color="red" if i >= L0 else "black")
        ax.text(1.03, -i, f"tok {int(tokens[i])} :{i}", ha="left", va="center", fontsize=8)
    ax.axis("off"); ax.set_xlim(-.5, 1.5)
    ax.set_title(f"Head view of L{ih[0]}H{ih[1]} (queries left, keys right; red = second period onward)", fontsize=10)
    _save(fig, out, "head_view_induction.png")


def _box(ax, y, text, color, w=6.4, h=.9):
    ax.add_patch(FancyBboxPatch((0, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="#333", lw=1))
    ax.text(w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    return y


def fig_architecture(cfg, ph, ih, out):
    fig, ax = plt.subplots(figsize=(10, 6.2)); ax.axis("off"); ax.set_xlim(-.5, 11); ax.set_ylim(-.5, 8.2)
    rows = [("Output logits (vocab = %d) -> next token" % cfg["vocab"], "#fde0dd"),
            ("LayerNorm  ->  Unembedding (linear)", "#eeeeee"),
            ("Layer 1: LayerNorm -> %d-head causal self-attention -> residual" % cfg["n_heads"], "#deebf7"),
            ("Layer 0: LayerNorm -> %d-head causal self-attention -> residual" % cfg["n_heads"], "#deebf7"),
            ("Token embedding + learned position embedding (d_model = %d)" % cfg["d_model"], "#eeeeee"),
            ("Input tokens: [ random prefix | same prefix repeated ]  (%d tokens)" % cfg["max_len"], "#e5f5e0")]
    for i, (t, c) in enumerate(rows): _box(ax, 7 - i * 1.4, t, c)
    for i in range(5):   # data flows upward: from the lower box top edge to the upper box bottom edge
        ax.annotate("", xy=(3.2, 7 - i * 1.4 - .05), xytext=(3.2, 7 - (i + 1) * 1.4 + .95), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.text(6.8, 7 - 3 * 1.4 + .45, "L0 previous-token heads: " + ", ".join(f"H{h}" for h in ph) + "\n(copy 'what came before me' into each position)", fontsize=9, va="center", color=C[1])
    ax.text(6.8, 7 - 2 * 1.4 + .45, f"L1 induction heads (best: H{ih[1]})\n(find the earlier copy of my token,\nattend to the token AFTER it)", fontsize=9, va="center", color=C[0])
    ax.set_title("AttentionLens model: attention-only 2-layer transformer (PyTorch)", fontsize=11)
    _save(fig, out, "architecture.png")


def fig_banner(att, out):
    fig = plt.figure(figsize=(14, 3.6), facecolor="#0f1117"); ax = fig.add_axes([0, 0, .3, 1])
    ax.imshow(att.numpy(), cmap="magma", vmin=0, vmax=1); ax.axis("off")
    t = fig.add_axes([.32, 0, .68, 1]); t.axis("off"); t.set_facecolor("#0f1117")
    t.text(0, .68, "AttentionLens", fontsize=44, color="white", weight="bold", va="center")
    t.text(0, .38, "Explainable AI for Transformers: finding and testing induction heads in PyTorch", fontsize=15, color="#c9d1e0", va="center")
    t.text(0, .16, "by Hadi Sarhangi Fard  -  github.com/Hadifard", fontsize=13, color="#e6550d", va="center")
    fig.savefig(os.path.join(out, "banner.png"), dpi=140, facecolor=fig.get_facecolor()); plt.close(fig)
