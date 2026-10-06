# ==============================================================================
# AttentionLens - Explainable AI for Transformers (PyTorch)
# Author  : Hadi Sarhangi Fard  |  GitHub: @Hadifard
# File    : data.py
# Purpose : Synthetic 'periodic random sequence' task with a variable period (forces content-based induction).
# License : MIT (see LICENSE)
# ==============================================================================
import torch

T_TOTAL = 32            # sequence length
MIN_L, MAX_L = 6, 16    # the period L (length of the random prefix) is sampled per sequence


def make_batch(batch, vocab=64, generator=None, L=None):
    """Sequence = a random prefix of L unique tokens, repeated until T_TOTAL tokens.

    Because L changes from sequence to sequence, the 'copy offset' is NOT constant, so a fixed positional
    shortcut cannot solve the task. After the first period the next token is fully determined by the
    token that followed the previous occurrence of the current token (the induction mechanism).

    Returns inputs (B, T-1), next-token targets (B, T-1), a mask of predictable positions (t >= L)
    and the period of every sequence.
    """
    prefix = torch.rand(batch, vocab, generator=generator).argsort(1)[:, :MAX_L]
    Ls = (torch.full((batch,), L) if L else torch.randint(MIN_L, MAX_L + 1, (batch,), generator=generator))
    seq = prefix.gather(1, torch.arange(T_TOTAL)[None, :] % Ls[:, None])
    mask = torch.arange(T_TOTAL - 1)[None, :] >= Ls[:, None]
    return seq[:, :-1], seq[:, 1:], mask, Ls


def repeat_accuracy(logits, targets, mask):
    """Next-token accuracy on the predictable positions only."""
    return ((logits.argmax(-1) == targets) & mask).sum().item() / mask.sum().item()
