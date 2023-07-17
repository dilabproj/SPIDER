import torch
import torch.nn as nn

from .slot_attention import build_grid, SoftPositionEmbed, SlotAttention
from .local_branches import PoolingAndPosEncoding

class GlobalAverPooling1d(nn.Module):
    def __init__(self):
        super(GlobalAverPooling1d, self).__init__()
        self.pooling = nn.AdaptiveMaxPool1d(output_size=1)
    
    def forward(self, x, ecg=None):
        d = len(x.shape)
        assert d in [2, 3], f"The input dimension slould be either 2 or 3, but get {d}."
        if d == 2:  # (C, L)
            C, L = x.shape
            return self.pooling(x).view((C, ))
            
        else:  # (N, C, L)
            N, C, L = x.shape
            return self.pooling(x).view((N, C))

    
class SingleTransEncoder(nn.Module):
    def __init__(self, dim, nhead=1, num_layers=1, 
                 target_time_steps=100):
        super(SingleTransEncoder, self).__init__()
        
        self.dim = dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.target_time_steps = target_time_steps
        
        self.time_pooling = nn.AdaptiveMaxPool1d(self.target_time_steps)
        self.pos_embed = SoftPositionEmbed(hidden_size=self.dim, resolution=(self.target_time_steps,))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        self.aggregator = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x, ecg=None, agg=True):
        # input shape: (B, C, T)
        # output shape: (B, C)
        out = self.time_pooling(x)
        out = self.pos_embed(out.permute((0, 2, 1)), ecg=ecg)
        out = self.transformer_encoder(out)  # out: (B, T, C)
        if agg:
            out = self.aggregator(out.permute((0, 2, 1))).squeeze()
        else:
            out = out.permute((0, 2, 1))  # out: (B, C, T)
        
        return out