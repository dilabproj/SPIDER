import torch
import torch.nn as nn

from .slot_attention import build_grid, SoftPositionEmbed, SlotAttention
             

class PoolingAndPosEncoding(nn.Module):
    def __init__(self, dim, target_time_steps=100):
        super(PoolingAndPosEncoding, self).__init__()
        
        self.dim = dim
        self.target_time_steps = target_time_steps
        
        self.time_pooling = nn.AdaptiveMaxPool1d(self.target_time_steps)
        self.pos_embed = SoftPositionEmbed(hidden_size=self.dim, resolution=(self.target_time_steps,))
        
    def forward(self, x):
        # input shape: (B, C, T)
        # output shape: input shape
        out = self.time_pooling(x)
        out = self.pos_embed(out.permute((0, 2, 1))).permute((0, 2, 1))
        
        return out
    

class TemporalSegmentation(nn.Module):
    def __init__(self, num_slots):
        super(TemporalSegmentation, self).__init__()
        self.pooling = nn.AdaptiveMaxPool1d(num_slots)
    
    def forward(self, x, ecg=None, return_k_iter=False):
        d = len(x.shape)
        assert d in [2, 3], f"The input dimension slould be either 2 or 3, but get {d}."
        
        if d == 2:  # (C, L)
            return self.pooling(x).permute((1, 0))
            
        else:  # (N, C, L)
            return self.pooling(x).permute((0, 2, 1))
    
    
class SlotEncoder(nn.Module):
    def __init__(self, dim, num_slots, num_iter, 
                 target_time_steps=100, k=1, learnable_init=False):
        super(SlotEncoder, self).__init__()
        
        self.dim = dim
        self.num_slots = num_slots
        self.num_iter = num_iter
        self.target_time_steps = target_time_steps
        self.k = k
        self.learnable_init = learnable_init
              
        self.time_pooling = nn.AdaptiveMaxPool1d(self.target_time_steps)
        self.pos_embed = SoftPositionEmbed(hidden_size=self.dim, resolution=(self.target_time_steps,))
        self.slot_att = SlotAttention(num_slots=self.num_slots,
                                      dim=self.dim,
                                      input_dim=self.dim,
                                      iters = self.num_iter,
                                      eps = 1e-8, 
                                      hidden_dim = self.dim,
                                      k = self.k,
                                      learnable_init = self.learnable_init)
    
    def forward(self, x, ecg=None, return_k_iter=False):
        # input shape: (B, C, T)
        # output shape: (B, num_slot, slot_dim)
        out = self.time_pooling(x)
        out = self.pos_embed(out.permute((0, 2, 1)), ecg=ecg)
        out = self.slot_att(out, return_k_iter=return_k_iter)
        
        return out