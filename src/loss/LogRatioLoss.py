"""Modernized Log-Ratio loss compatible with torch.nn.Module."""

from __future__ import annotations

import torch
from torch import nn


class LogRatioLoss(nn.Module):
    """Pairwise log-ratio regression loss for distance metric learning."""

    def __init__(self, p: float = 2.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, inputs: torch.Tensor, gt_dist: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 2:
            raise ValueError("LogRatioLoss expects `inputs` with shape (batch_size, embedding_dim).")
        if inputs.size(0) < 2:
            raise ValueError("LogRatioLoss requires at least two samples in the batch.")

        anchor = inputs[0]
        pairs = inputs[1:]

        # Pairwise L2 distance between anchor and remaining samples.
        eps_dist = 1e-4 / max(anchor.numel(), 1)
        diff = torch.abs(pairs - anchor.unsqueeze(0))
        dist = torch.pow(diff, self.p).sum(dim=1)
        dist = torch.pow(dist + eps_dist, 1.0 / self.p)

        gt_dist = gt_dist.to(inputs)
        if gt_dist.dim() != 1 or gt_dist.size(0) != pairs.size(0):
            raise ValueError("LogRatioLoss expects `gt_dist` with shape (batch_size - 1,).")

        log_dist = torch.log(dist + self.eps)
        log_gt_dist = torch.log(gt_dist + self.eps)

        diff_log_dist = log_dist.unsqueeze(0) - log_dist.unsqueeze(1)
        diff_log_gt_dist = log_gt_dist.unsqueeze(0) - log_gt_dist.unsqueeze(1)

        mask = torch.triu(torch.ones_like(diff_log_dist, dtype=torch.bool), diagonal=1)
        if not torch.any(mask):
            return torch.zeros((), dtype=inputs.dtype, device=inputs.device)

        weights = mask.float()
        weights = weights / weights.sum()

        loss = (diff_log_dist - diff_log_gt_dist).pow(2)
        loss = (loss * weights).sum()
        return loss