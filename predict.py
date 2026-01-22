import os, re, glob, argparse, warnings
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import torch
torch.backends.cudnn.enabled = False


from data_loader import read_fasta, build_protein_graph, AA_VOCAB, UNK_IDX, PAD_IDX  
from phase_separation_model import PhaseSeparationModel                                        

def load_llm_npy(path_npy: Path, expect_dim: int) -> torch.Tensor:
    if not path_npy.is_file():
        raise FileNotFoundError(f"LLM embedding not found: {path_npy}")
    arr = np.load(path_npy)
    if arr.ndim != 2 or arr.shape[1] != expect_dim:
        raise ValueError(f"Bad LLM shape {arr.shape} in {path_npy}, expect [L,{expect_dim}]")
    return torch.from_numpy(arr.astype(np.float32))

def seq_to_ids(seq: str) -> torch.Tensor:
    return torch.tensor([AA_VOCAB.get(a.upper(), UNK_IDX) for a in seq], dtype=torch.long)

def infer_config_from_state(state: dict):
    import re


    vocab_size, d_model = state["encoder.embed.weight"].shape

    esm_dim = state["proj_esm.weight"].shape[1]


    blk_ids = []
    pat = re.compile(r"^encoder\.blocks\.(\d+)\.")
    for k in state.keys():
        m = pat.match(k)
        if m:
            blk_ids.append(int(m.group(1)))
    n_layer = max(blk_ids) + 1 if blk_ids else 6


    a_key = None
    for cand in [
        "encoder.blocks.0.mamba.A_log",
        "encoder.blocks.0.A_log",
    ]:
        if cand in state:
            a_key = cand
            break
    if a_key is None:

        for k in state:
            if k.endswith(".mamba.A_log") or k.endswith(".A_log"):
                a_key = k
                break
    if a_key is None:
        raise KeyError("Cannot locate Mamba A_log in state_dict.")
    n_ssm = state[a_key].shape[1]


    dt_key = None
    for cand in [
        "encoder.blocks.0.mamba.dt_proj.weight",
        "encoder.blocks.0.dt_proj.weight",
    ]:
        if cand in state:
            dt_key = cand
            break
    if dt_key is None:

        for k in state:
            if k.endswith("mamba.dt_proj.weight") or k.endswith("dt_proj.weight"):
                dt_key = k
                break
    if dt_key is None:
        raise KeyError("Cannot locate Mamba dt_proj.weight in state_dict.")
    dt_rank = state[dt_key].shape[1]


    gcl_ids = []
    pat2 = re.compile(r"^egnn\.gcl_(\d+)\.")
    for k in state.keys():
        m = pat2.match(k)
        if m:
            gcl_ids.append(int(m.group(1)))
    egnn_layers = max(gcl_ids) + 1 if gcl_ids else 4

    hidden_nf = state["egnn.embedding_in.weight"].shape[0]

    rbf_k = int(state["rbf.centers"].numel())

    cfg = dict(
        aa_vocab=vocab_size,
        d_model=d_model,
        esm_dim=esm_dim,
        n_layer=n_layer,
        n_ssm=n_ssm,
        dt_rank=dt_rank,
        egnn_layers=egnn_layers,
        hidden_nf=hidden_nf,
        rbf_k=rbf_k,
        dropout=0.0,  
    )
    return cfg


def build_model_from_ckpt(ckpt_path: Path, device: torch.device):
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:

        state = torch.load(ckpt_path, map_location="cpu")
    cfg = infer_config_from_state(state)
    model = PhaseSeparationModel(**cfg).to(device)   
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg

@torch.no_grad()
def predict_one(name: str, device, dirs, models, esm_dim: int, threshold: float):
    fasta_path = Path(dirs["fasta_dir"]) / f"{name}.fasta"
    pdb_path   = Path(dirs["pdb_dir"])   / f"{name}.pdb"
    llm_path   = Path(dirs["llm_dir"])   / f"{name}.rep_1280.npy"


    seq = read_fasta(str(fasta_path))
    if not seq:
        raise ValueError(f"Empty or invalid FASTA: {fasta_path}")
    ids = seq_to_ids(seq)                      # (L,)
    emb = load_llm_npy(llm_path, expect_dim=esm_dim)  # (L, 1280)
    coords, edge_index, edge_attr = build_protein_graph(str(pdb_path))  # coords(L,3), edge_attr(E,1)

    L = ids.size(0)
    if emb.size(0) != L or coords.size(0) != L:
        raise ValueError(f"{name}: length mismatch: seq={L}, emb={emb.size(0)}, coords={coords.size(0)}")

    # batch = 1
    ids  = ids.unsqueeze(0).to(device)               # (1,L)
    emb  = emb.unsqueeze(0).to(device)               # (1,L,1280)
    xyz  = coords.unsqueeze(0).to(device)            # (1,L,3)
    ei   = edge_index.to(device)                     # (2,E)
    ea   = edge_attr.to(device)                      # (E,1) 

    probs = []
    for m in models:
        logit = m(ids, emb, xyz, ei, ea).view(-1)    # (1,)
        prob = torch.sigmoid(logit)[0].item()
        probs.append(prob)
    score = float(sum(probs) / len(probs))
    tag = "yes" if score >= threshold else "no"
    return score, tag

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names_txt", required=True)
    ap.add_argument("--fasta_dir", required=True)
    ap.add_argument("--pdb_dir",   required=True)
    ap.add_argument("--llm_dir",   required=True)
    ap.add_argument("--ckpt_dir",  required=True)
    ap.add_argument("--out",       required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")
    print(f"Device: {device}")


    ckpt_files = sorted(glob.glob(str(Path(args.ckpt_dir) / "*.pt")))
    if not ckpt_files:
        raise FileNotFoundError(f"No .pt under {args.ckpt_dir}")

    models = []
    first_model, cfg0 = build_model_from_ckpt(Path(ckpt_files[0]), device)
    models.append(first_model)
    print(f"Loaded: {ckpt_files[0]}")
    for p in ckpt_files[1:]:
        m, cfgi = build_model_from_ckpt(Path(p), device)

        assert cfgi["d_model"] == cfg0["d_model"] and cfgi["esm_dim"] == cfg0["esm_dim"], \
            f"Config mismatch between {ckpt_files[0]} and {p}"
        models.append(m)
        print(f"Loaded: {p}")
    print(f"Loaded {len(models)} checkpoints.")

    dirs = dict(fasta_dir=args.fasta_dir, pdb_dir=args.pdb_dir, llm_dir=args.llm_dir)


    with open(args.names_txt, "r") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    if not names:
        raise ValueError("names_txt is empty.")


    out_lines = []
    for name in tqdm(names, desc="Predicting", unit="name"):
        try:
            score, tag = predict_one(name, device, dirs, models, esm_dim=cfg0["esm_dim"], threshold=args.threshold)
            out_lines.append(f"{name}\t{score:.6f}\t{tag}")
        except Exception as e:
            warnings.warn(f"[{name}] failed: {e}")
            out_lines.append(f"{name}\tNaN\tno")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for ln in out_lines:
            f.write(ln + "\n")
    print(f"✅ Done. Wrote {len(out_lines)} lines to {args.out}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

