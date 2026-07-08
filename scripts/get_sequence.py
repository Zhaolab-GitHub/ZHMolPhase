import argparse
import os
import sys
from typing import List, Tuple

import requests


def read_id_list(id_file: str, deduplicate: bool = True) -> List[str]:
    """
    Read UniProt IDs from a text file.

    Rules:
      - one ID per line
      - blank lines are ignored
      - lines starting with # are ignored
      - only the first whitespace-separated token is used
      - duplicated IDs are removed while preserving order by default
    """
    if not os.path.isfile(id_file):
        raise FileNotFoundError(f"[ERROR] ID file not found: {id_file}")

    ids: List[str] = []
    seen = set()

    with open(id_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            uid = s.split()[0].strip()
            if not uid:
                continue

            if deduplicate:
                if uid in seen:
                    print(f"[SKIP-DUP] line {line_no}: {uid}")
                    continue
                seen.add(uid)

            ids.append(uid)

    if not ids:
        raise ValueError(f"[ERROR] No valid UniProt ID found in: {id_file}")

    return ids


def resolve_output_path(uid: str, out_dir: str) -> str:
    """Save each FASTA as <out_dir>/<uid>.fasta."""
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{uid}.fasta")


def download_uniprot_fasta(uniprot_id: str, save_path: str, timeout: int = 15) -> None:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"

    try:
        r = requests.get(url, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"request failed: {e}")

    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")

    text = r.text.strip()
    if not text.startswith(">"):
        raise RuntimeError("unexpected FASTA content; response does not start with '>'")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch download UniProt FASTA files from a name.txt file."
    )
    parser.add_argument("id_file", help="Text file with one UniProt ID per line, e.g. name.txt")
    parser.add_argument("out_dir", help="Output directory. FASTA files will be saved as <out_dir>/<ID>.fasta")
    parser.add_argument("--timeout", type=int, default=15, help="HTTP timeout in seconds. Default: 15")
    parser.add_argument("--force", action="store_true", help="Overwrite existing non-empty FASTA files")
    parser.add_argument("--keep-duplicates", action="store_true", help="Do not remove duplicated IDs")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately when one ID fails")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ids = read_id_list(args.id_file, deduplicate=not args.keep_duplicates)
    print(f"[INFO] Input ID file : {os.path.abspath(args.id_file)}")
    print(f"[INFO] Output dir    : {os.path.abspath(args.out_dir)}")
    print(f"[INFO] Number of IDs : {len(ids)}")

    n_downloaded = 0
    n_skipped = 0
    failed: List[Tuple[str, str]] = []

    for i, uid in enumerate(ids, start=1):
        save_path = resolve_output_path(uid, args.out_dir)
        print(f"\n[{i}/{len(ids)}] UniProt ID: {uid}")
        print(f"[INFO] Output FASTA: {os.path.abspath(save_path)}")

        if os.path.exists(save_path) and os.path.getsize(save_path) > 0 and not args.force:
            print(f"[SKIP] exists: {save_path}")
            n_skipped += 1
            continue

        try:
            download_uniprot_fasta(uid, save_path, timeout=args.timeout)
            print(f"[DONE] FASTA downloaded: {uid}")
            n_downloaded += 1
        except Exception as e:
            msg = str(e)
            print(f"[FAILED] {uid}: {msg}")
            failed.append((uid, msg))
            if args.fail_fast:
                break

    print("\n===== Summary =====")
    print(f"Total IDs   : {len(ids)}")
    print(f"Downloaded  : {n_downloaded}")
    print(f"Skipped     : {n_skipped}")
    print(f"Failed      : {len(failed)}")

    if failed:
        print("\nFailed IDs:")
        for uid, msg in failed:
            print(f"  {uid}\t{msg}")
        sys.exit(2)

    print("[ALL DONE] FASTA batch download finished.")


if __name__ == "__main__":
    main()
