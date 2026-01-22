import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_inner, n_ssm, dt_rank, dropout=0.2):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_inner)
        self.conv1d  = nn.Conv1d(d_inner, d_inner, 3, padding=1, groups=d_inner)
        self.out_proj= nn.Linear(d_inner, d_model)
        self.A_log = nn.Parameter(torch.randn(d_inner, n_ssm))
        self.D     = nn.Parameter(torch.randn(d_inner))
        self.x_proj= nn.Linear(d_inner, dt_rank + n_ssm)
        self.dt_proj= nn.Linear(dt_rank, d_inner)
        self.dropout= nn.Dropout(dropout)

    def ssm(self, x):
        d_in, n = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x); total = self.x_proj.out_features
        expected = total - n
        delta, B = x_dbl.split([expected, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        y = self.selective_scan(x, delta, A, B, D)
        return y

    def selective_scan(self, u, delta, A, B, D):
        b, l, d_in = u.shape; n = A.shape[1]
        deltaA   = torch.exp(einsum(delta, A, 'b l d, d n -> b l d n'))
        deltaB_u = einsum(delta, B, u, 'b l d, b l n, b l d -> b l d n')
        x = torch.zeros((b, d_in, n), device=u.device, dtype=u.dtype)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            ys.append(x.sum(dim=-1))
        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y

    def forward(self, x):
        b,l,d = x.shape
        x = self.in_proj(x)
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d l -> b l d')
        x = F.silu(x); x = self.dropout(x)
        y = self.ssm(x); y = self.dropout(y)
        out = self.out_proj(y)
        return out


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class ConvMotifBank(nn.Module):
    def __init__(self, d_model, ks=(3,5,7), dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, k, padding=k//2, groups=1),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, 1),
            ) for k in ks
        ])
        self.proj = nn.Linear(d_model, d_model)
        self.dp = nn.Dropout(dropout)

    def forward(self, x): 
        b,l,d = x.shape
        xt = x.transpose(1,2)  
        ys = [branch(xt) for branch in self.branches] 
        y  = torch.stack(ys, dim=0).sum(dim=0) / len(self.branches)  
        y  = y.transpose(1,2)  
        return self.dp(self.proj(y))


class ResidualBlock(nn.Module):
    def __init__(self, d_model, d_inner, n_ssm, dt_rank, dropout=0.2):
        super().__init__()
        self.norm_m = nn.LayerNorm(d_model)
        self.norm_c = nn.LayerNorm(d_model)
        self.mamba  = MambaBlock(d_model, d_inner, n_ssm, dt_rank, dropout)
        self.conv   = ConvMotifBank(d_model, ks=(3,5,7), dropout=dropout)
        self.dp     = nn.Dropout(dropout)

    def forward(self, x):
        m = self.mamba(self.norm_m(x))
        c = self.conv (self.norm_c(x))
        y = (m + c) * 0.5
        return self.dp(x + y)

class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, n_ssm, dt_rank, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.blocks = nn.ModuleList([
            ResidualBlock(d_model, 2*d_model, n_ssm, dt_rank, dropout)
            for _ in range(n_layer)
        ])
        self.norm = RMSNorm(d_model)
        self.dp = nn.Dropout(dropout)

    def forward(self, ids): 
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.dp(self.norm(x))  # (B,L,d)

