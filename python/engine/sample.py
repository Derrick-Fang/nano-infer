"""Sampler: logits -> next token.

Mirrors mini-sglang's `engine/sample.py`. Only greedy (argmax) is
supported for now; temperature / top_k / top_p and BatchSamplingArgs
arrive in R5.
"""

from dataclasses import dataclass

import torch


@dataclass
class Sampler:
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: [batch, vocab] -> [batch]; argmax per row (greedy)."""
        return logits.argmax(dim=-1)
