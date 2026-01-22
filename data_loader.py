
from __future__ import annotations
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
from Bio import PDB

AA_VOCAB = {a: i + 1 for i, a in enumerate("ACDEFGHIKLMNPQRSTVWYXBZUO")}  
PAD_IDX = 0
UNK_IDX = AA_VOCAB["X"]

def read_fasta(path: str) -> str:
    seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq.append(line)
    return "".join(seq)

def build_protein_graph(pdb_file: str, cutoff: float = 10.0):
    """
    Returns:
      coords     : (L,3) float32
      edge_index : (2,E) long, UNDIRECTED (i,j) & (j,i)
      edge_attr  : (E,1) float32, Euclidean distance in Å
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_file)[0]
    residues = list(structure.get_residues())

    coords, edges, dists = [], [], []

    for res in residues:
        if "CA" in res:
            coords.append(list(res["CA"].get_coord()))
        elif "N" in res:
            coords.append(list(res["N"].get_coord()))
        elif "C" in res:
            coords.append(list(res["C"].get_coord()))
        else:
            coords.append([np.nan, np.nan, np.nan])

    for i, res_i in enumerate(residues):
        at1 = "CA" if "CA" in res_i else ("N" if "N" in res_i else ("C" if "C" in res_i else None))
        if at1 is None:
            continue
        for j, res_j in enumerate(residues[i + 1 :], start=i + 1):
            at2 = "CA" if "CA" in res_j else ("N" if "N" in res_j else ("C" if "C" in res_j else None))
            if at2 is None:
                continue
            dist = res_i[at1] - res_j[at2]  # Å
            if dist <= cutoff:
                edges.append((i, j)); dists.append(dist)
                edges.append((j, i)); dists.append(dist)

    coords = torch.tensor(np.asarray(coords), dtype=torch.float32).nan_to_num_()
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(dists, dtype=torch.float32).unsqueeze(-1)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 1, dtype=torch.float32)

    return coords, edge_index, edge_attr

class PhaseSeparationDataset(Dataset):
    """Return: ids, esm_emb, label, coords, edge_index, edge_attr"""

    def __init__(self, llps_list_file: str, non_llps_list_file: str, root: str = "."):
        super().__init__()
        self.samples: List[Tuple[str, int]] = []
        with open(llps_list_file) as f:
            self.samples += [(l.strip(), 1) for l in f if l.strip()]
        with open(non_llps_list_file) as f:
            self.samples += [(l.strip(), 0) for l in f if l.strip()]

        self.seq_dir = os.path.join(root, "LLPS")  # *.fasta
        self.emb_dir = os.path.join(root, "LLM")   # *.rep_1280.npy
        self.pdb_dir = os.path.join(root, "pdb")   # *.pdb

        self.seq_lens = [len(read_fasta(os.path.join(self.seq_dir, f"{n}.fasta"))) for n, _ in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        name, label = self.samples[idx]

        # ids
        seq = read_fasta(os.path.join(self.seq_dir, f"{name}.fasta"))
        ids = torch.tensor([AA_VOCAB.get(a, UNK_IDX) for a in seq], dtype=torch.long)

        # ESM
        emb_np = np.load(os.path.join(self.emb_dir, f"{name}.rep_1280.npy"))
        esm_emb = torch.from_numpy(emb_np).float()

        # graph
        coords, ei, ea = build_protein_graph(os.path.join(self.pdb_dir, f"{name}.pdb"))

        # sanity
        L = len(ids)
        assert esm_emb.shape[0] == L, f"{name}: seq {L} vs emb {esm_emb.shape[0]}"
        assert coords.shape[0] == L, f"{name}: seq {L} vs coords {coords.shape[0]}"

        return ids, esm_emb, torch.tensor(label, dtype=torch.float32), coords, ei, ea

def collate(batch):
    ids, embs, labels, xyzs, ei_list, ea_list = zip(*batch)
    B = len(batch)
    L_max = max(t.size(0) for t in ids)
    feat_dim = embs[0].size(-1)

    pad_ids = torch.full((B, L_max), PAD_IDX, dtype=torch.long)
    pad_emb = torch.zeros(B, L_max, feat_dim, dtype=torch.float32)
    pad_xyz = torch.zeros(B, L_max, 3, dtype=torch.float32)

    src_all, dst_all, ea_all = [], [], []
    offset = 0
    for b, (ii, ee, xyz, ei, ea) in enumerate(zip(ids, embs, xyzs, ei_list, ea_list)):
        L = ii.size(0)
        pad_ids[b, :L] = ii
        pad_emb[b, :L] = ee
        pad_xyz[b, :L] = xyz
        if ei.numel() > 0:
            src_all.append(ei[0] + offset)
            dst_all.append(ei[1] + offset)
            ea_all.append(ea)
        offset += L

    if src_all:
        edge_index = torch.stack([torch.cat(src_all), torch.cat(dst_all)], dim=0)
        edge_attr = torch.cat(ea_all, dim=0)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 1, dtype=torch.float32)

    labels = torch.stack(labels)  # (B,)
    return pad_ids, pad_emb, labels, pad_xyz, edge_index, edge_attr

class SizeBucketBatchSampler(BatchSampler):
    """Bin by length to reduce padding (optional helper)."""
    def __init__(self, seq_lens: List[int], batch_size: int, threshold: int = 1000, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        pairs = list(enumerate(seq_lens))
        short = [i for i, L in pairs if L <= threshold]
        long = [i for i, L in pairs if L > threshold]
        self.batches = []
        for pool in (short, long):
            if not pool:
                continue
            if shuffle:
                import random as _r
                _r.shuffle(pool)
            for i in range(0, len(pool), batch_size):
                self.batches.append(pool[i : i + batch_size])
        if shuffle:
            import random as _r
            _r.shuffle(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

def build_dataloader(llps_list_file: str, non_llps_list_file: str,
                     batch_size: int = 8, shuffle: bool = True,
                     num_workers: int = 4, root: str = ".",
                     bucket_by_size: bool = False, threshold: int = 1000):
    ds = PhaseSeparationDataset(llps_list_file, non_llps_list_file, root)
    if bucket_by_size:
        sampler = SizeBucketBatchSampler(ds.seq_lens, batch_size, threshold, shuffle)
        return DataLoader(ds, batch_sampler=sampler, num_workers=num_workers, collate_fn=collate,
                          pin_memory=True, persistent_workers=(num_workers > 0))
    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate,
                          pin_memory=True, persistent_workers=(num_workers > 0))

