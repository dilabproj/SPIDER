from typing import Dict, Optional, Tuple

import numpy as np
import scipy.optimize
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pytorch_metric_learning import losses


class HungarianLoss(nn.Module):
    def __init__(self):
        super(HungarianLoss, self).__init__()
        
    def forward(self, inputs, targets):
        # inputs & targets shape: (batch_size, num_slots, slot_dim)
        
        # l2 normalization
        inputs = F.normalize(inputs, dim=-1, p=2)
        targets = F.normalize(targets, dim=-1, p=2)
        
        # for each sample in the batch
        batch_loss = 0
        batch_size, *_ = inputs.shape  # (B, S, D)
        
        for i in range(batch_size):
            ip = inputs[i]  # (S, D)
            tg = targets[i]  # (S, D)
            
            cos_mat = torch.einsum('ad, bd -> ab', ip, tg)
            self_cos_mat = torch.einsum('ad, bd -> ab', ip, ip)
            self_cos_mat = torch.tril(self_cos_mat, diagonal=-1)[:, :-1] + torch.triu(self_cos_mat, diagonal=1)[:, 1:]
            
            selected_losses = self.calculate_hungarian(-cos_mat)
            target = torch.Tensor(selected_losses)[:, 1]
            target = target.to(torch.long).to(cos_mat.device)
            
            batch_loss += F.cross_entropy(torch.cat((cos_mat, self_cos_mat), axis=-1), target)
        
        return batch_loss / batch_size
    
    def calculate_hungarian(self, cost_mat_tensor):
        cost_mat_numpy = cost_mat_tensor.detach().cpu().numpy()
        row_idx, col_idx = linear_sum_assignment(cost_mat_numpy)
        
        assigned = [(r, c) for r, c in zip(row_idx, col_idx)]
            
        return assigned
    
    def __repr__(self):
        return f'{self.__class__.__name__}'

class BYOLSimLoss(nn.Module):
    def __init__(self):
        super(BYOLSimLoss, self).__init__()
    
    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, dim=-1, p=2)
        targets = F.normalize(targets, dim=-1, p=2)
        
        return 2 - 2 * (inputs * targets).sum(dim=-1)
        

class WeightedMSE(nn.Module):
    def __init__(self):
        super(WeightedMSE, self).__init__()
        
    def forward(self, inputs, targets):
        masks = targets.abs() < 3
        
        se = (inputs - targets) ** 2 * masks
        wse = se * torch.clip(targets.abs(), max=1)
        
        return wse.mean()


class HungarianLoss2(nn.Module):
    def __init__(self, 
                 temperature: float = 1.0, 
                 reduction: str = "mean",):
        super(HungarianLoss2, self).__init__()
        
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        inputs = inputs.unsqueeze(1)
        targets = targets.unsqueeze(1)
        slots = torch.cat((inputs, targets), axis=1)
        
        loss = matching_contrastive_loss(slots, temperature=self.temperature, reduction=self.reduction)
        
        return loss

    def __repr__(self):
        return f'{self.__class__.__name__}(temperature={self.temperature}, reduction={self.reduction})'

    
def matching_contrastive_loss(
    slots: Tensor,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """Contrastive object-wise loss, all vs. all.

    The vectors at ``[b, 0, :, :]`` and ``[b, 1, :, :]`` of ``slots`` must represent
    ``S`` slot embeddings of shape ``D`` of different augmentations of the b-th image.
    For each image pair ``((b, 0), (b, 1)), the ``S`` embeddings are 1:1 matched using
    linear-sum assignment to produce the targets for a ``2BS-1``-classes classification
    problem. The matching slot represents the positive class, and the remaining
    ``2BS-2`` slots are considered negatives.

    Worst case:
    if all embeddings collapse to the same value, the loss will be ``log(2BS-1)``.

    Best case:
    if each image gets an embedding that is orthogonal to all others,
    the loss will be ``log(exp(1/t) + 2BS - 2) - 1/t``.

    Args:
        slots: ``[B, 2, S, D]`` tensor of projected image features
        temperature: temperature scaling
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar loss over all samples and slots if reduction is 'mean' or 'sum'.
        A tensor ``[B, 2, S]`` of losses if reduction is 'none'

    Example:
        A batch of ``B=4`` images, augmented twice, each with ``S=3``.
        Note the symmetry of matches along the diagonal.
        The ``X`` represent positive matching targets for the cross entropy loss,
        the ``.`` represent negatives included in the loss (all except diagonal)::

                              img_0       img_1        img_2       img_3
                           ╭─────────╮ ╭─────────╮  ╭─────────╮ ╭─────────╮
                             0     1     0     1      0     1     0     1
                  ╭       ┃  . .│. . X┃. . .│. . .┃. . .│. . .┃. . .│. . .┃
                  │ aug_0 ┃.   .│. X .┃. . .│. . .┃. . .│. . .┃. . .│. . .┃
                  │       ┃. .  │X . .┃. . .│. . .┃. . .│. . .┃. . .│. . .┃
            img_0 │       ╋─────┼─────╋─────┼─────╋─────┼─────╋─────│─────╋
                  │       ┃. . X│  . .┃. . .│. . .┃. . .│. . .┃. . .│. . .┃
                  │ aug_1 ┃. X .│.   .┃. . .│. . .┃. . .│. . .┃. . .│. . .┃
                  ╰       ┃X . .│. .  ┃. . .│. . .┃. . .│. . .┃. . .│. . .┃
                          ┃━━━━━│━━━━━┃━━━━━│━━━━━╋━━━━━│━━━━━┃━━━━━│━━━━━┃
                  ╭       ┃. . .│. . .┃  . .│. X .┃. . .│. . .┃. . .│. . .┃
                  │ aug_0 ┃. . .│. . .┃.   .│X . .┃. . .│. . .┃. . .│. . .┃
                  │       ┃. . .│. . .┃. .  │. . X┃. . .│. . .┃. . .│. . .┃
            img_1 │       ┃─────┼─────┃─────┼─────╋─────┼─────╋─────┼─────┃
                  │       ┃. . .│. . .┃. X .│  . .┃. . .│. . .┃. . .│. . .┃
                  │ aug_1 ┃. . .│. . .┃X . .│.   .┃. . .│. . .┃. . .│. . .┃
                  ╰       ┃. . .│. . .┃. . X│. .  ┃. . .│. . .┃. . .│. . .┃
                          ┃━━━━━│━━━━━┃━━━━━│━━━━━╋━━━━━│━━━━━┃━━━━━╋━━━━━┃
                  ╭       ┃. . .│. . .┃. . .│. . .┃  . .│. . X┃. . .│. . .┃
                  │ aug_0 ┃. . .│. . .┃. . .│. . .┃.   .│X . .┃. . .│. . .┃
                  │       ┃. . .│. . .┃. . .│. . .┃. .  │. X .┃. . .│. . .┃
            img_2 │       ┃─────┼─────┃─────┼─────╋─────┼─────╋─────┼─────┃
                  │       ┃. . .│. . .┃. . .│. . .┃. X .│  . .┃. . .│. . .┃
                  │ aug_1 ┃. . .│. . .┃. . .│. . .┃. . X│.   .┃. . .│. . .┃
                  ╰       ┃. . .│. . .┃. . .│. . .┃X . .│. .  ┃. . .│. . .┃
                          ┃━━━━━│━━━━━┃━━━━━│━━━━━╋━━━━━│━━━━━┃━━━━━│━━━━━┃
                  ╭       ┃. . .│. . .┃. . .│. . .┃. . .│. . .┃  . .│. X .┃
                  │ aug_0 ┃. . .│. . .┃. . .│. . .┃. . .│. . .┃.   .│. . X┃
                  │       ┃. . .│. . .┃. . .│. . .┃. . .│. . .┃. .  │X . .┃
            img_3 │       ┃─────┼─────┃─────┼─────╋─────┼─────╋─────┼─────┃
                  │       ┃. . .│. . .┃. . .│. . .┃. . .│. . .┃. . X│  . .┃
                  │ aug_1 ┃. . .│. . .┃. . .│. . .┃. . .│. . .┃X . .│.   .┃
                  ╰       ┃. . .│. . .┃. . .│. . .┃. . .│. . .┃. X .│. .  ┃
    """
    B, A, S, D = slots.shape
    if A != 2:
        raise ValueError(f"Invalid shape {slots.shape}")
    
    # Full cosine similarity matrix between all slots of all images.
    # cos: [B, 2, S, B, 2, S]
    cos = cos_pairwise(slots)
    
    # Prepare cross-entropy targets by running linear sum assignment
    # on cosine similarity for each pair of augmented images.
    #
    # Thanks to symmetry w.r.t. the diagonal, matches need to be computed
    # only for the B blocks of size [S, S] that are in the top-right
    # quarter of each [A*S, A*S] block:
    #
    # for b in range(B):
    #     match(cos_pairwise(slots[b, 0], slots[B, 1]))
    #     match(cos_pairwise(slots[b, 1], slots[B, 0])) <- not needed, use argsort
    #
    # The only thing to take care of is to offset the column indices so that
    # they correspond to the desired location in the cos matrix.
    
    targets = torch.full((B, A, S), fill_value=-1)
    for b in range(B):
        cos_np = cos[b, 0, :, b, 1, :].detach().cpu().numpy()
        # First output is a vector of sorted row idxs [0, 1, ..., S]
        _, cols = scipy.optimize.linear_sum_assignment(cos_np, maximize=True)
        targets[b, 0, :] = torch.from_numpy(cols).add_(S * (A * b + 1))
        targets[b, 1, :] = torch.from_numpy(np.argsort(cols)).add_(S * A * b)
    
    targets = targets.reshape(B * A * S)
    cos = cos.reshape(B * A * S, B * A * S).div_(temperature).fill_diagonal_(-np.inf)
    loss = F.cross_entropy(cos, targets.to(slots.device), reduction=reduction)
    if reduction == "none":
        loss = loss.reshape(B, A, S)
    
    # Debug
    # probs = cos.detach().reshape(B*A*S, B*A*S).softmax(dim=-1)
    # print(probs.mul(100).int().cpu().numpy())
    # with np.printoptions(linewidth=150, formatter={"bool": ".X".__getitem__}):
    #     onehot = np.zeros(cos.shape, dtype=bool)
    #     onehot[np.arange(len(targets)), targets.cpu().numpy()] = 1
    #     print(onehot)

    return loss


def cos_pairwise(a: Tensor, b: Optional[Tensor] = None) -> Tensor:
    """Cosine between all pairs of entries in two tensors.

    Args:
        a: [*N, C] tensor, where ``*N`` can be any number of leading dimensions.
        b: [*M, C] tensor, where ``*M`` can be any number of leading dimensions.
            Defaults to ``a`` if missing.

    Returns:
        [*N, *M] tensor of cosine values.
    """
    a = l2_normalize(a)
    b = a if b is None else l2_normalize(b)
    N = a.shape[:-1]
    M = b.shape[:-1]
    a = a.flatten(end_dim=-2)
    b = b.flatten(end_dim=-2)
    cos = torch.einsum("nc,mc->nm", a, b)
    
    return cos.reshape(N + M)


def l2_normalize(a: Tensor) -> Tensor:
    """L2 normalization along the last dimension.

    Args:
        a: [..., C] tensor to normalize.

    Returns:
        A new tensor containing normalized rows.
    """
    # norm = torch.linalg.vector_norm(a, dim=-1, keepdim=True)
    norm = torch.linalg.norm(a, ord=2, dim=-1, keepdim=True)
    return a / norm.clamp_min(1e-10)


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        
        self.temperature = temperature
        self.loss_func = losses.NTXentLoss(temperature=self.temperature)
        
    def forward(self, inputs, targets, labels=None):
        batch_size = 1 if inputs.dim()==1 else inputs.shape[0]
        
        if labels is None:
            labels = torch.arange(batch_size).repeat(2)
        
        return self.loss_func(torch.cat((inputs, targets), dim=0), labels)