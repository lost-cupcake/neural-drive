"""
Multi-task loss balancing.

Supports:
  - uncertainty  : Kendall et al. (2018) — learnable log-sigma per task
  - gradnorm     : Chen et al. (2018)    — gradient norm equalisation
  - fixed        : manual weights from config
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Per-task losses
# ──────────────────────────────────────────────────────────────────────────────

def gaussian_focal_loss(
    pred: torch.Tensor,   # (B, cls, H, W) sigmoid heatmap
    gt:   torch.Tensor,   # (B, cls, H, W) gaussian heatmap [0,1]
    alpha: float = 2.0,
    beta:  float = 4.0,
) -> torch.Tensor:
    """
    Modified focal loss for centrepoint heatmap regression.
    From CenterPoint / CornerNet.
    """
    pos_mask = gt.eq(1).float()
    neg_mask = gt.lt(1).float()

    neg_weight = (1 - gt) ** beta

    pos_loss = -(pred + 1e-6).log() * (1 - pred) ** alpha * pos_mask
    neg_loss = -(1 - pred + 1e-6).log() * pred ** alpha * neg_weight * neg_mask

    num_pos  = pos_mask.sum().clamp(min=1)
    loss     = (pos_loss.sum() + neg_loss.sum()) / num_pos
    return loss


def reg_l1_loss(
    pred:    torch.Tensor,   # (B, D, H, W)
    target:  torch.Tensor,   # (B, D, H, W)
    mask:    torch.Tensor,   # (B, H, W)  1 at object centres
) -> torch.Tensor:
    mask = mask.unsqueeze(1).expand_as(pred)
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    return loss / (mask.sum().clamp(min=1))


def silog_depth_loss(
    pred: torch.Tensor,   # (B*N, 1, H, W) positive depth
    gt:   torch.Tensor,   # (B*N, 1, H, W) GT depth (0 = invalid)
    lam:  float = 0.5,
) -> torch.Tensor:
    """Scale-Invariant Log loss (Eigen et al., 2014)."""
    valid = gt > 0
    if not valid.any():
        return pred.sum() * 0.0

    d = torch.log(pred[valid].clamp(min=1e-3)) - torch.log(gt[valid].clamp(min=1e-3))
    return (d ** 2).mean() - lam * (d.mean() ** 2)


def detection_loss(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Combines heatmap + offset + wlh + yaw + velocity losses.
    targets should have keys: heatmap, offset, wlh, yaw, velocity, centre_mask
    """
    losses = {}
    mask = targets.get('centre_mask', torch.ones_like(preds['heatmap'][:, 0]))

    losses['heatmap']  = gaussian_focal_loss(preds['heatmap'], targets['heatmap'])
    losses['offset']   = reg_l1_loss(preds['offset'],   targets['offset'],   mask)
    losses['wlh']      = reg_l1_loss(preds['wlh'],      targets['wlh'],      mask)
    losses['yaw']      = reg_l1_loss(preds['yaw'],      targets['yaw'],      mask)
    losses['velocity'] = reg_l1_loss(preds['velocity'], targets['velocity'], mask)

    losses['total'] = (losses['heatmap'] +
                       losses['offset'] +
                       losses['wlh'] +
                       losses['yaw'] * 0.5 +
                       losses['velocity'] * 0.2)
    return losses


def segmentation_loss(
    pred: torch.Tensor,    # (B, cls, H, W) logits
    gt:   torch.Tensor,    # (B, H, W)      int class labels
    ignore_index: int = 255,
) -> torch.Tensor:
    """Cross-entropy + Dice hybrid."""
    ce = F.cross_entropy(pred, gt, ignore_index=ignore_index)

    # Dice (per-class, foreground only)
    pred_soft = pred.softmax(dim=1)
    gt_oh     = F.one_hot(gt.clamp(0, pred.shape[1] - 1), pred.shape[1]).permute(0, 3, 1, 2).float()
    inter     = (pred_soft * gt_oh).sum(dim=(0, 2, 3))
    denom     = pred_soft.sum(dim=(0, 2, 3)) + gt_oh.sum(dim=(0, 2, 3)) + 1e-6
    dice      = 1 - (2 * inter / denom).mean()

    return ce + dice


# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty weighting (Kendall et al., 2018)
# ──────────────────────────────────────────────────────────────────────────────

class UncertaintyWeightedLoss(nn.Module):
    """
    Learns log(σ²) per task. The total loss is:
        Σ_t  L_t / (2σ_t²) + log(σ_t)
    Minimising this automatically down-weights high-uncertainty tasks.
    """

    def __init__(self, task_names: list[str]):
        super().__init__()
        # Initialise log_sigma² = 0 → σ = 1 → no initial bias
        self.log_sigma2 = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1))
            for name in task_names
        })

    def forward(self, task_losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """
        task_losses: {task_name: scalar loss}
        Returns (total_loss, info_dict with per-task weights)
        """
        total  = 0.0
        info   = {}
        for name, loss in task_losses.items():
            log_s2     = self.log_sigma2[name]
            weighted   = loss * (-log_s2).exp() + 0.5 * log_s2
            total      = total + weighted
            info[f'w_{name}'] = (-log_s2).exp().item()
            info[f'loss_{name}'] = loss.item()
        return total, info


# ──────────────────────────────────────────────────────────────────────────────
# GradNorm balancer (Chen et al., 2018)
# ──────────────────────────────────────────────────────────────────────────────

class GradNormLoss(nn.Module):
    """
    GradNorm: normalises gradient magnitudes across tasks.
    Requires access to shared backbone parameters (last shared layer).
    """

    def __init__(self, task_names: list[str], alpha: float = 1.5):
        super().__init__()
        self.task_names = task_names
        self.alpha      = alpha
        self.weights    = nn.ParameterDict({
            name: nn.Parameter(torch.ones(1))
            for name in task_names
        })
        # Keep initial losses for normalisation (set on first forward)
        self.register_buffer('L0', torch.zeros(len(task_names)))
        self._initialized = False

    def forward(
        self,
        task_losses:      dict[str, torch.Tensor],
        shared_params:    list[nn.Parameter],    # last shared layer params
    ) -> tuple[torch.Tensor, dict]:

        losses_t = torch.stack([task_losses[n] for n in self.task_names])

        if not self._initialized:
            self.L0          = losses_t.detach()
            self._initialized = True

        # Weighted sum
        w = torch.stack([self.weights[n] for n in self.task_names]).squeeze()
        weighted_losses = w * losses_t
        total = weighted_losses.sum()

        # GradNorm auxiliary loss (computed separately, update weights only)
        if shared_params:
            G_norms = []
            for wl in weighted_losses:
                grads = torch.autograd.grad(wl, shared_params, retain_graph=True, allow_unused=True)
                G = torch.stack([g.norm() for g in grads if g is not None])
                G_norms.append(G.mean())
            G_norms  = torch.stack(G_norms)
            G_mean   = G_norms.mean().detach()

            r = (losses_t.detach() / self.L0.clamp(min=1e-8))
            r = r / r.mean()
            target_G = (G_mean * (r ** self.alpha)).detach()

            gradnorm_loss = F.l1_loss(G_norms, target_G)
            gradnorm_loss.backward(retain_graph=True)  # update weights only

        # Re-normalise weights so they sum to num_tasks
        with torch.no_grad():
            w_sum = sum(self.weights[n].item() for n in self.task_names)
            for n in self.task_names:
                self.weights[n].clamp_(min=0.01)
                self.weights[n].data *= len(self.task_names) / w_sum

        info = {f'w_{n}': self.weights[n].item() for n in self.task_names}
        info.update({f'loss_{n}': task_losses[n].item() for n in self.task_names})
        return total, info


# ──────────────────────────────────────────────────────────────────────────────
# Fixed weights
# ──────────────────────────────────────────────────────────────────────────────

class FixedWeightLoss(nn.Module):
    def __init__(self, weights: dict[str, float]):
        super().__init__()
        self.weights = weights

    def forward(self, task_losses: dict[str, torch.Tensor], **kwargs) -> tuple[torch.Tensor, dict]:
        total = sum(self.weights.get(n, 1.0) * l for n, l in task_losses.items())
        info  = {f'loss_{n}': l.item() for n, l in task_losses.items()}
        return total, info


# ──────────────────────────────────────────────────────────────────────────────
# MTL Loss orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class MTLLoss(nn.Module):
    """
    Orchestrates all task losses and the balancing strategy.
    Use this as the single loss module in training.
    """

    BALANCERS = {
        'uncertainty': UncertaintyWeightedLoss,
        'gradnorm':    GradNormLoss,
        'fixed':       FixedWeightLoss,
    }

    def __init__(
        self,
        task_names:     list[str],
        balancer:       str   = 'uncertainty',
        fixed_weights:  dict  = None,
        alpha:          float = 1.5,
    ):
        super().__init__()
        self.task_names = task_names

        if balancer == 'uncertainty':
            self.balancer = UncertaintyWeightedLoss(task_names)
        elif balancer == 'gradnorm':
            self.balancer = GradNormLoss(task_names, alpha)
        elif balancer == 'fixed':
            fw = fixed_weights or {n: 1.0 for n in task_names}
            self.balancer = FixedWeightLoss(fw)
        else:
            raise ValueError(f"Unknown balancer: {balancer}")

    def compute_task_losses(
        self,
        preds:   dict,
        targets: dict,
    ) -> dict[str, torch.Tensor]:
        """
        Compute per-task scalar losses.
        preds and targets share the same key structure.
        """
        task_losses = {}

        # Detection
        if 'detection' in preds and 'detection' in targets:
            det_losses   = detection_loss(preds['detection'], targets['detection'])
            task_losses['detection'] = det_losses['total']

        # Segmentation
        if 'segmentation' in preds and 'segmentation' in targets:
            task_losses['segmentation'] = segmentation_loss(
                preds['segmentation'], targets['segmentation']
            )

        # Depth
        if 'depth' in preds and 'depth' in targets:
            task_losses['depth'] = silog_depth_loss(
                preds['depth'], targets['depth']
            )

        return task_losses

    def forward(
        self,
        preds:          dict,
        targets:        dict,
        shared_params:  list = None,     # for GradNorm
    ) -> tuple[torch.Tensor, dict]:

        task_losses = self.compute_task_losses(preds, targets)

        if isinstance(self.balancer, GradNormLoss):
            total, info = self.balancer(task_losses, shared_params or [])
        else:
            total, info = self.balancer(task_losses)

        return total, info


if __name__ == '__main__':
    # Quick smoke test
    tasks = ['detection', 'segmentation', 'depth']
    loss_fn = MTLLoss(tasks, balancer='uncertainty')

    fake_losses = {
        'detection':    torch.tensor(2.5, requires_grad=True),
        'segmentation': torch.tensor(0.8, requires_grad=True),
        'depth':        torch.tensor(1.2, requires_grad=True),
    }

    total, info = loss_fn.balancer(fake_losses)
    print(f"Total loss: {total.item():.4f}")
    for k, v in info.items():
        print(f"  {k}: {v:.4f}")
