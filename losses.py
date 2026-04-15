"""
손실함수 모음
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017)
    - gamma=0 → CrossEntropyLoss와 동일
    - gamma=2 권장 (기쁨/당황/슬픔 혼동 클래스 대응)
    """
    def __init__(self, gamma: float = 2.0, weight=None, reduction: str = 'mean'):
        super().__init__()
        self.gamma     = gamma
        self.weight    = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        ce    = F.nll_loss(log_p, targets, weight=self.weight, reduction='none')
        p_t   = torch.exp(-ce)
        loss  = (1 - p_t) ** self.gamma * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def build_criterion(loss_type: str, num_classes: int, device,
                    gamma: float = 2.0) -> nn.Module:
    if loss_type == 'focal':
        return FocalLoss(gamma=gamma).to(device)
    else:
        return nn.CrossEntropyLoss().to(device)
