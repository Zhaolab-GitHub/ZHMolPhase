import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch


FASTA_EXTS = (".fasta", ".fa", ".faa", ".fas")
EXPECTED_ESM2_NAME = "esm2_t33_650M_UR50D"


def read_names(names_txt: str, deduplicate: bool = True) -> List[str]:
    """Read one protein name / UniProt ID per line. Ignore blank lines and comments."""
    if not os.path.isfile(names_txt):
        raise FileNotFoundError(f"[ERROR] names_txt not found: {names_txt}")

    names: List[str] = []
    seen = set()
    with open(names_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            name = s.split()[0]
            if deduplicate:
                if name in seen:
                    continue
                seen.add(name)
            names.append(name)

    if not names:
        raise ValueError(f"[ERROR] no valid names found in {names_txt}")
    return names


def find_fasta(name: str, fasta_dir: str) -> str:
    """Find <name>.fasta/.fa/.faa/.fas in fasta_dir."""
    for ext in FASTA_EXTS:
        path = os.path.join(fasta_dir, name + ext)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"[ERROR] FASTA not found for {name}. Tried: "
        + ", ".join(os.path.join(fasta_dir, name + ext) for ext in FASTA_EXTS)
    )


def read_fasta_sequence(fasta_file: str) -> Tuple[str, str]:
    """Read the first FASTA record without requiring BioPython."""
    header: Optional[str] = None
    seq_parts: List[str] = []

    with open(fasta_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                if header is not None:
                    # Only read the first record.
                    break
                header = s[1:].strip()
            else:
                seq_parts.append(s.replace(" ", ""))

    sequence = "".join(seq_parts).upper()
    if not sequence:
        raise ValueError(f"[ERROR] empty sequence in {fasta_file}")
    return header or os.path.basename(fasta_file), sequence


def import_esm_package():
    """Import the installed fair-esm package."""
    try:
        import esm  # type: ignore
        return esm
    except ImportError as e:
        raise ImportError(
            "[ERROR] Cannot import 'esm'. This script no longer uses an esm-main/ folder.\n"
            "Please install fair-esm in the current conda environment, for example:\n"
            "  pip install fair-esm\n"
        ) from e


def load_esm2_model(model_path: str, device: torch.device):
    """
    Load ESM-2 from a user-provided local .pt checkpoint.

    We intentionally avoid esm.pretrained.esm2_t33_650M_UR50D(), because that
    function downloads/loads by model name from the PyTorch cache. Instead, this
    function loads the exact checkpoint path provided by the user.

    For embedding extraction, contact-regression weights are not required.
    If a co-located <model>-contact-regression.pt file exists, it will be used;
    otherwise the model is loaded without contact-regression weights.
    """
    esm = import_esm_package()

    model_path = os.path.abspath(os.path.expanduser(model_path))
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"[ERROR] ESM-2 checkpoint not found: {model_path}")

    model_stem = Path(model_path).stem
    if EXPECTED_ESM2_NAME not in model_stem:
        print(
            f"[WARN] The checkpoint name does not contain '{EXPECTED_ESM2_NAME}': {os.path.basename(model_path)}",
            file=sys.stderr,
        )
        print(
            "[WARN] ZHMolPhase was trained with esm2_t33_650M_UR50D embeddings; using another ESM checkpoint may change feature dimensions or predictions.",
            file=sys.stderr,
        )

    # Prefer explicit local loading through fair-esm internals so that a missing
    # contact-regression file does not prevent embedding extraction.
    try:
        model_data = torch.load(model_path, map_location="cpu")
        regression_path = str(Path(model_path).with_suffix("")) + "-contact-regression.pt"
        regression_data = None
        if os.path.isfile(regression_path):
            print(f"[INFO] Contact-regression weights found: {regression_path}")
            regression_data = torch.load(regression_path, map_location="cpu")
        else:
            print(
                "[INFO] Contact-regression weights were not found. This is OK for residue embedding extraction."
            )

        model, alphabet = esm.pretrained.load_model_and_alphabet_core(
            model_stem, model_data, regression_data
        )
    except Exception as e_core:
        # Fallback to the public fair-esm local loader. This may require the
        # co-located contact-regression file for some ESM checkpoints.
        try:
            model, alphabet = esm.pretrained.load_model_and_alphabet(model_path)
        except Exception as e_public:
            raise RuntimeError(
                "[ERROR] Failed to load the ESM-2 checkpoint from the provided path.\n"
                f"  model_path: {model_path}\n"
                f"  load_model_and_alphabet_core error: {e_core}\n"
                f"  load_model_and_alphabet error: {e_public}\n"
            )

    model.eval()
    model.to(device)
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


def iter_segments(sequence: str, chunk_len: int) -> List[Tuple[str, str]]:
    """Split long proteins into ESM-compatible chunks."""
    if chunk_len <= 0:
        raise ValueError("[ERROR] chunk_len must be positive")
    return [(f"segment_{i // chunk_len + 1}", sequence[i : i + chunk_len]) for i in range(0, len(sequence), chunk_len)]


def compute_one_embedding(
    sequence: str,
    model,
    alphabet,
    batch_converter,
    device: torch.device,
    repr_layer: int = 33,
    chunk_len: int = 1022,
    segment_batch_size: int = 1,
) -> np.ndarray:
    """Compute per-residue ESM-2 embeddings with shape (L, 1280)."""
    segments = iter_segments(sequence, chunk_len)
    sequence_representations: List[np.ndarray] = []

    if segment_batch_size <= 0:
        raise ValueError("[ERROR] segment_batch_size must be positive")

    with torch.no_grad():
        for start in range(0, len(segments), segment_batch_size):
            batch_data = segments[start : start + segment_batch_size]
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
            token_representations = results["representations"][repr_layer]

            for i, tokens_len in enumerate(batch_lens):
                tokens_len_int = int(tokens_len.item())
                if tokens_len_int <= 2:
                    continue
                # Remove <cls> and <eos>.
                segment_embedding = token_representations[i, 1 : tokens_len_int - 1].detach().cpu().numpy()
                sequence_representations.append(segment_embedding)

            # Reduce peak memory during long batch runs.
            del batch_tokens, results, token_representations
            if device.type == "cuda":
                torch.cuda.empty_cache()

    embeddings = np.concatenate(sequence_representations, axis=0)
    if embeddings.shape[0] != len(sequence):
        raise RuntimeError(
            f"[ERROR] embedding length mismatch: sequence length = {len(sequence)}, embedding length = {embeddings.shape[0]}"
        )
    if embeddings.shape[1] != 1280:
        raise RuntimeError(
            f"[ERROR] unexpected embedding dimension: {embeddings.shape}. Expected (L, 1280) for esm2_t33_650M_UR50D."
        )
    return embeddings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch compute ESM-2 esm2_t33_650M_UR50D residue embeddings for ZHMolPhase."
    )

    # Positional quick use: name.txt sequence LLM /path/to/esm2_t33_650M_UR50D.pt
    parser.add_argument(
        "positional",
        nargs="*",
        help="Optional positional form: name.txt fasta_dir out_dir model_path",
    )

    parser.add_argument("--names_txt", default=None, help="Path to name.txt; one protein name / UniProt ID per line.")
    parser.add_argument("--fasta_dir", default=None, help="Directory containing FASTA files named <name>.fasta/.fa/.faa.")
    parser.add_argument("--out_dir", "--llm_dir", dest="out_dir", default=None, help="Output directory for <name>.rep_1280.npy.")
    parser.add_argument(
        "--model_path",
        "--esm2_model_path",
        dest="model_path",
        default=None,
        help="Local path to esm2_t33_650M_UR50D.pt. Required.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for ESM-2 inference.")
    parser.add_argument("--chunk_len", type=int, default=1022, help="Maximum sequence length per ESM-2 segment.")
    parser.add_argument("--segment_batch_size", type=int, default=1, help="Number of sequence segments processed together.")
    parser.add_argument("--repr_layer", type=int, default=33, help="ESM-2 representation layer to export.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing non-empty .npy files.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop immediately when one protein fails.")
    parser.add_argument("--keep_duplicates", action="store_true", help="Do not remove duplicated names in name.txt.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Support: python compute_esm2_batch.py name.txt sequence LLM /path/to/model.pt
    if args.positional:
        if len(args.positional) != 4:
            raise SystemExit(
                "[ERROR] positional usage requires exactly 4 arguments:\n"
                "  python compute_esm2_batch.py name.txt fasta_dir out_dir /path/to/esm2_t33_650M_UR50D.pt"
            )
        if args.names_txt or args.fasta_dir or args.out_dir or args.model_path:
            raise SystemExit("[ERROR] use either positional arguments or named arguments, not both.")
        args.names_txt, args.fasta_dir, args.out_dir, args.model_path = args.positional

    if not args.names_txt or not args.fasta_dir or not args.out_dir or not args.model_path:
        raise SystemExit(
            "[ERROR] Missing required arguments.\n"
            "Usage:\n"
            "  python compute_esm2_batch.py --names_txt name.txt --fasta_dir sequence --out_dir LLM --model_path /path/to/esm2_t33_650M_UR50D.pt\n"
            "or:\n"
            "  python compute_esm2_batch.py name.txt sequence LLM /path/to/esm2_t33_650M_UR50D.pt"
        )

    names = read_names(args.names_txt, deduplicate=not args.keep_duplicates)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("[ERROR] --device cuda was requested but CUDA is not available.")

    print(f"[INFO] Number of proteins : {len(names)}")
    print(f"[INFO] FASTA directory    : {os.path.abspath(args.fasta_dir)}")
    print(f"[INFO] Output directory   : {os.path.abspath(args.out_dir)}")
    print(f"[INFO] Device             : {device}")
    print(f"[INFO] ESM-2 checkpoint   : {os.path.abspath(os.path.expanduser(args.model_path))}")
    print("[INFO] Loading ESM-2 model from the user-provided checkpoint path")

    model, alphabet, batch_converter = load_esm2_model(args.model_path, device)

    n_done = 0
    n_skip = 0
    failed: List[Tuple[str, str]] = []

    for idx, name in enumerate(names, 1):
        out_path = os.path.join(args.out_dir, f"{name}.rep_1280.npy")
        print("=" * 80)
        print(f"[{idx}/{len(names)}] {name}")

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0 and not args.force:
            print(f"[SKIP] exists: {out_path}")
            n_skip += 1
            continue

        try:
            fasta_path = find_fasta(name, args.fasta_dir)
            header, sequence = read_fasta_sequence(fasta_path)
            print(f"[INFO] FASTA : {fasta_path}")
            print(f"[INFO] Header: {header}")
            print(f"[INFO] Length: {len(sequence)}")

            embeddings = compute_one_embedding(
                sequence=sequence,
                model=model,
                alphabet=alphabet,
                batch_converter=batch_converter,
                device=device,
                repr_layer=args.repr_layer,
                chunk_len=args.chunk_len,
                segment_batch_size=args.segment_batch_size,
            )
            np.save(out_path, embeddings)
            print(f"[DONE] saved: {out_path}  shape={embeddings.shape}")
            n_done += 1

        except Exception as e:
            msg = str(e)
            print(f"[FAILED] {name}: {msg}")
            failed.append((name, msg))
            if args.fail_fast:
                break

    print("=" * 80)
    print("[SUMMARY]")
    print(f"Total     : {len(names)}")
    print(f"Computed  : {n_done}")
    print(f"Skipped   : {n_skip}")
    print(f"Failed    : {len(failed)}")
    if failed:
        print("[FAILED LIST]")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
        sys.exit(2)


if __name__ == "__main__":
    main()
