import torch
from torch import nn
import torch.nn.functional as F


class HierarchicalLoss(nn.Module):
    def __init__(self, alpha=0.5, temporal_unit=3, l2_norm=True):
        super(HierarchicalLoss, self).__init__()
        
        self.alpha = alpha
        self.temporal_unit = temporal_unit
        self.l2_norm = l2_norm
    
    def forward(self, inputs, targets):
        loss = hierarchical_contrastive_loss(inputs, targets, 
                                             alpha=self.alpha, 
                                             temporal_unit=self.temporal_unit, 
                                             l2_norm=self.l2_norm)
        
        return loss
    
    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha}, temporal_unit={self.temporal_unit}, l2_norm={self.l2_norm})'
    
    
def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=3, l2_norm=True):
    # (B, c) -> (B, C, T=1)
    z1 = z1.unsqueeze(2) if len(z1.shape) == 2 else z1
    z2 = z2.unsqueeze(2) if len(z2.shape) == 2 else z2
    
    # z.shape: (B, C, T) -> (B, T, C)
    z1 = z1.transpose(1, 2)
    z2 = z2.transpose(1, 2)
    
    if l2_norm:
        z1 = F.normalize(z1, dim=-1, p=2)
        z2 = F.normalize(z2, dim=-1, p=2)
    
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:  # sequence lengeth > 1
        if alpha != 0:  # instance loss weight
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    
    return loss / d


def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    
    # removing diagonal elements (self similarity)
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2  # only include z1xz2 and z2xz1 parts
    
    return loss


def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    
    # removing diagonal elements
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    
    return loss
