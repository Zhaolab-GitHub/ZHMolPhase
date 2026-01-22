import sys
import os
from typing import Dict, List, Tuple

import pandas as pd

FASTA_ROOT = "example/sequence"
WINDOW_LEN = 11
HALF = 5


def avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def extract_full_score(path):
    full_score = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# full_score:"):
                try:
                    full_score = float(line.split(":", 1)[1].strip())
                except:
                    pass
                break
    return full_score


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:

    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [list(intervals[0])]

    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:  
            if e > last_e:
                merged[-1][1] = e
        else:
            merged.append([s, e])

    return [(s, e) for s, e in merged]


def reconstruct_hash_from_scores(prob_by_pos: Dict[int, float]):

    hash_k: Dict[int, float] = {}
    for pos, score in prob_by_pos.items():
        k = pos - 10
        if k >= 1:
            hash_k[k] = score

    if not hash_k:
        return {}, [], 0

    ordered_keys = sorted(hash_k.keys())
    prob_list = [hash_k[k] for k in ordered_keys]
    length = len(prob_list)

    return hash_k, prob_list, length


def select_seed_indices(hash_k: Dict[int, float],
                        prob_list: List[float],
                        length: int) -> List[int]:


    if not hash_k:
        return []

    scores = sorted(hash_k.values())
    n = len(scores)
    if n == 0:
        return []

    # 90 
    idx = int(n * 0.9)
    if idx >= n:
        idx = n - 1
    threshold = scores[idx]

    seeds = [k for k, v in hash_k.items() if v >= threshold]

    seeds = sorted(seeds)

    return seeds


def compute_regions_from_hash(hash_k: Dict[int, float],
                              seeds: List[int]) -> Dict[int, int]:


    if not seeds:
        return {}

    intervals: List[Tuple[int, int]] = []
    for k in seeds:
        s = k - HALF
        e = k + HALF
        if s < 1:
            s = 1
        intervals.append((s, e))

    merged = merge_intervals(intervals)

    region_rank = {}
    for s, e in merged:
        vals = []
        for i in range(s, e + 1):
            if i in hash_k:
                vals.append(hash_k[i])
        if not vals:
            continue
        region_avg = avg(vals)
        key_str = f"query\t{s}\t{e}\t{region_avg}"
        region_rank[key_str] = region_avg

    if not region_rank:
        return {}

    limit = 5
    region: Dict[int, int] = {}
    n = 0
    for key in sorted(region_rank.keys(), key=lambda k: region_rank[k], reverse=True):
        n += 1
        query, s_s, e_s, score_s = key.split("\t")
        s = int(s_s)
        e = int(e_s)
        for i in range(s, e + 1):
            region[i] = region.get(i, 0) + 1
        if n >= limit:
            break

    return region


def recalc_dregion_for_sequence(prob_by_pos: Dict[int, float],
                                max_pos: int) -> Dict[int, int]:

    hash_k, prob_list, length = reconstruct_hash_from_scores(prob_by_pos)
    if not hash_k:
        return {pos: 0 for pos in range(1, max_pos + 1)}

    seeds = select_seed_indices(hash_k, prob_list, length)
    region_k = compute_regions_from_hash(hash_k, seeds)   

    dregion_by_pos: Dict[int, int] = {}
    for pos in range(1, max_pos + 1):
        k = pos - 10         
        if k in region_k:
            dregion_by_pos[pos] = 1
        else:
            dregion_by_pos[pos] = 0

    return dregion_by_pos


def write_output(
    path_out: str,
    aa_by_pos: Dict[int, str],
    prob_by_pos: Dict[int, float],
    dregion_by_pos: Dict[int, int],
    max_pos: int,
    full_score: float,   
):

    with open(path_out, "w") as f:
        f.write("# DRegion with 1 denoted predicted key residues\n")
        if full_score is not None:
            f.write(f"# full_score: {full_score:.6f}\n")
        f.write("Pos\tAA\tProb\tDRegion\n")
        for pos in range(1, max_pos + 1):
            aa = aa_by_pos.get(pos, "X")
            prob = prob_by_pos.get(pos, None)
            prob_str = "-" if prob is None else f"{prob:.6f}"
            d = dregion_by_pos.get(pos, 0)
            f.write(f"{pos}\t{aa}\t{prob_str}\t{d}\n")


def read_fasta(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: The FASTA is missing: {path}")
    seq_lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                continue
            seq_lines.append(line)
    seq = "".join(seq_lines)
    if not seq:
        raise ValueError(f"Error: No valid sequence found in the FASTA file: {path}")
    return seq


def get_name_from_occlusion(path: str) -> str:

    name = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if line.lower().startswith("# name:"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        name = parts[1].strip()
                        break
                continue
            else:
                break

    if not name:
        base = os.path.basename(path)
        name = os.path.splitext(base)[0]
    return name


def convert_occlusion_to_center_prob(in_tsv: str):

    name = get_name_from_occlusion(in_tsv)
    fasta_path = os.path.join(FASTA_ROOT, f"{name}.fasta")
    seq = read_fasta(fasta_path)
    seq_len = len(seq)
    full_score = extract_full_score(in_tsv)

    df = pd.read_csv(
        in_tsv,
        sep="\t",
        comment="#",
        header=None,
        names=["pos_start", "pos_end", "delta_score"],
    )

    if df.empty:
        raise ValueError(f"Error: No valid data found in the occlusion file: {in_tsv}")


    max_end = int(df["pos_end"].max())
    expected_len = max_end + 1  
    if expected_len != seq_len:
        raise ValueError(
            f"Sequence length mismatch: fasta lenth={seq_len}, "
            f"occlusion-inferred length={expected_len} (max pos_end={max_end})"
        )


    center_0 = (df["pos_start"] + df["pos_end"]) // 2
    df["center_pos"] = center_0 + 1

    max_center = int(df["center_pos"].max())
    if max_center > seq_len:
        raise ValueError(
            f"Center residue position exceeds sequence length: max center_pos={max_center}, seq_len={seq_len}"
        )

    grouped = df.groupby("center_pos")["delta_score"].max()

    prob_by_pos: Dict[int, float] = {
        int(pos): float(score) for pos, score in grouped.items()
    }

    aa_by_pos: Dict[int, str] = {pos: seq[pos - 1] for pos in range(1, seq_len + 1)}

    max_pos: int = seq_len

    return aa_by_pos, prob_by_pos, max_pos,  full_score



def main():
    if len(sys.argv) != 3:
        print("Usage: python try_PSPHunter_region.py XX_occlusion.tsv out.txt")
        sys.exit(1)

    in_tsv = sys.argv[1]
    out_path = sys.argv[2]

    aa_by_pos, prob_by_pos, max_pos, full_score = convert_occlusion_to_center_prob(in_tsv)

    dregion_by_pos = recalc_dregion_for_sequence(prob_by_pos, max_pos)

    write_output(out_path, aa_by_pos, prob_by_pos, dregion_by_pos, max_pos, full_score)
    print(f"[INFO] Driving Region: {out_path}")


if __name__ == "__main__":
    main()

