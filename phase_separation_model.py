from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from sequence_embedding import SequenceEncoder, RMSNorm
from egnn import EGNN

class RBFExpand(nn.Module):
    def __init__(self, num_k=16, cutoff=20.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, num_k))
        self.gamma = nn.Parameter(torch.tensor(10.0))  
    def forward(self, d):  # (E,1)
        # exp( -gamma * (d - c)^2 )
        diff = d - self.centers.view(1, -1)   # (E,K)
        return torch.exp(-self.gamma * diff * diff)    # (E,K)

class AttentionPooling(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1, bias=False)
        )
        self.drop = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        w = self.scorer(x).squeeze(-1)   # (B,L)
        if mask is not None:
            w = w.masked_fill(~mask, -1e9)
        w = torch.softmax(w, dim=1)
        return self.drop(torch.einsum("blc,bl->bc", x, w))   # (B,C)

class PhaseSeparationModel(nn.Module):
    def __init__(
        self,
        aa_vocab: int    = 22,
        d_model: int     = 128,
        esm_dim: int     = 1280,
        n_layer: int     = 6,
        n_ssm: int       = 4,
        dt_rank: int     = 1,
        dropout: float   = 0.1,
        egnn_layers: int = 4,
        hidden_nf: int   = 128,
        rbf_k: int       = 16,
    ):
        super().__init__()
        self.encoder  = SequenceEncoder(aa_vocab, d_model, n_layer, n_ssm, dt_rank, dropout)
        self.proj_esm = nn.Linear(esm_dim, d_model)
        self.egnn_in = d_model + d_model
        self.rbf = RBFExpand(num_k=rbf_k, cutoff=20.0)
        self.egnn = EGNN(
            in_node_nf=self.egnn_in,
            hidden_nf=hidden_nf,
            out_node_nf=d_model,
            in_edge_nf=rbf_k,
            n_layers=egnn_layers,
            dropout=dropout,
            normalize=True,
            tanh=True,
            edge_gating=True,
        )
        self.pool = AttentionPooling(d_model, 128, dropout)
        self.cls  = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(
        self,
        ids: torch.LongTensor,       
        esm_emb: torch.Tensor,      
        xyz: torch.Tensor,          
        edge_index: torch.Tensor,    
        edge_attr: torch.Tensor,     
    ) -> torch.Tensor:               
        mask = (ids != 0)

        h_seq = self.encoder(ids)            
        h_esm = self.proj_esm(esm_emb)       
        h_node = torch.cat([h_seq, h_esm], dim=-1)  

        B, L, _ = h_node.shape
        h_flat   = h_node.reshape(B*L, -1)
        xyz_flat = xyz.reshape(B*L, 3)

        e_feat = self.rbf(edge_attr)         

        h_out, _ = self.egnn(h_flat, xyz_flat, edge_index, e_feat)  
        h_out = h_out.reshape(B, L, -1)

        g = self.pool(h_out, mask)           
        logit = self.cls(g).squeeze(-1)      
        return logit

