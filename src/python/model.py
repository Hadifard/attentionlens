# ==============================================================================
# AttentionLens - Explainable AI for Transformers (PyTorch)
# Author  : Hadi Sarhangi Fard  |  GitHub: @Hadifard
# File    : model.py
# Purpose : Tiny causal transformer that exposes every attention matrix and supports head ablation.
# License : MIT (see LICENSE)
# ==============================================================================
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention that returns its attention weights."""

    def __init__(self, d_model, n_heads, max_len):
        super().__init__()
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)))

    def forward(self, x, head_mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) / self.dk ** 0.5            # (B, H, T, T)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf")).softmax(-1)
        out = att @ v                                               # (B, H, T, dk)
        if head_mask is not None:                                   # 1 = keep head, 0 = ablate head
            out = out * head_mask.view(1, -1, 1, 1)
        return self.proj(out.transpose(1, 2).reshape(B, T, D)), att


class Block(nn.Module):
    def __init__(self, d_model, n_heads, max_len, use_mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len)
        self.ln2 = nn.LayerNorm(d_model) if use_mlp else None
        self.mlp = (nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                  nn.Linear(4 * d_model, d_model)) if use_mlp else None)

    def forward(self, x, head_mask=None):
        a, att = self.attn(self.ln1(x), head_mask)
        x = x + a
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x, att


class TinyGPT(nn.Module):
    """Small decoder-only transformer (GPT-style). Attention-only by default for clean interpretability."""

    def __init__(self, vocab=64, d_model=64, n_heads=4, n_layers=2, max_len=31, use_mlp=False):
        super().__init__()
        self.cfg = dict(vocab=vocab, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                        max_len=max_len, use_mlp=use_mlp)
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, max_len, use_mlp) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx, head_mask=None):
        """Returns logits (B, T, V) and attention weights (B, n_layers, n_heads, T, T)."""
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1), device=idx.device))
        atts = []
        for l, block in enumerate(self.blocks):
            x, a = block(x, None if head_mask is None else head_mask[l])
            atts.append(a)
        return self.head(self.ln_f(x)), torch.stack(atts, 1)
