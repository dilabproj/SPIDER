import numpy as np

from torch import nn
import torch
import torch.nn.functional as F


"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution, device="cpu"):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.resolution = resolution
        
        self.dimension = len(self.resolution)
        self.embedding = nn.Linear(self.dimension * 2, hidden_size, bias=True)
        self.grid = self._build_grid()
        
        self.device = device

    def forward(self, inputs):
        if self.device != inputs.device:
            self.device = inputs.device
            self.to(self.device)
            self.grid = self.grid.to(self.device)
        
        grid = self.embedding(self.grid)
        return inputs + grid
    
    def _build_grid(self):
        ranges = [np.linspace(0., 1., num=res) for res in self.resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [r for r in self.resolution] + [-1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)

        return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))

    
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, input_dim = None, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.dim = dim
        self.input_dim = dim if input_dim is None else input_dim
        
        self.slots_mu = nn.Parameter(torch.randn(1, 1, self.dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, self.dim).abs())

        self.to_q = nn.Linear(self.dim, self.dim)
        self.to_k = nn.Linear(self.input_dim, self.dim)
        self.to_v = nn.Linear(self.input_dim, self.dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(self.dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.dim)

        self.norm_input  = nn.LayerNorm(self.input_dim)
        self.norm_slots  = nn.LayerNorm(self.dim)
        self.norm_pre_ff = nn.LayerNorm(self.dim)

    def forward(self, inputs, num_slots = None):
        b, n, _ = inputs.shape
        d = self.dim
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma.abs())

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            
            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [r for r in resolution] + [-1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))


        
    
   