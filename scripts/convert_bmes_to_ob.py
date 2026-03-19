#!/usr/bin/env python3
"""
Convert parquet label columns from BMES scheme to B/O scheme.

BMES → B/O mapping (following infer_guwen.py logic):
  E (end of sentence)      → B  (sentence boundary)
  S (single-char sentence) → B  (sentence boundary)
  B (beginning of sentence) → O
  M (middle of sentence)    → O

Optimised for Kaggle (4 CPU cores, 30 GB RAM):
  - multiprocessing.Pool across parquet files
  - NumPy-vectorised label conversion (no Python-level per-element loop)
  - PyArrow zero-copy I/O where possible

Usage:
    python convert_bmes_to_ob.py --input_dir data --splits train val test
    python convert_bmes_to_ob.py --input_file data/test/segmentation_sliding_test.parquet
"""

import argparse
import multiprocessing as mp
import os
import time
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================================
# VECTORISED LABEL CONVERSION
# ============================================================================

# Pre-built lookup: byte value of label char → 1 (boundary) or 0 (non-boundary).
# 'E'=69, 'S'=83 → B;  'B'=66, 'M'=77 → O
_LOOKUP = np.zeros(128, dtype=np.uint8)
_LOOKUP[ord("E")] = 1
_LOOKUP[ord("S")] = 1
# 'B' and 'M' stay 0 → mapped to "O"

_ID_TO_LABEL = np.array(["O", "B"], dtype=object)


def bmes_to_ob_numpy(bmes_labels: List[str]) -> List[str]:
    """Vectorised BMES → O/B via NumPy lookup table."""
    if not bmes_labels:
        return []
    first_chars = np.frombuffer(
        "".join(lbl[0] for lbl in bmes_labels).encode("ascii"), dtype=np.uint8
    )
    ids = _LOOKUP[first_chars]
    return _ID_TO_LABEL[ids].tolist()


# ============================================================================
# LABEL STATISTICS
# ============================================================================

def count_labels_fast(labels_column) -> Counter:
    """Count label occurrences using flattened Arrow arrays."""
    flat = pa.concat_arrays(
        [chunk.values for chunk in labels_column.chunks]
    )
    counts: Counter = Counter()
    unique, unique_counts = np.unique(
        flat.to_numpy(zero_copy_only=False), return_counts=True
    )
    for lbl, cnt in zip(unique, unique_counts):
        counts[str(lbl)] = int(cnt)
    return counts


def print_label_stats(
    before: Counter, after: Counter, file_name: str
) -> None:
    """Print a side-by-side comparison of label distributions."""
    print(f"\n  📊 Label statistics for: {file_name}")
    print(f"  {'─' * 52}")
    print(f"  {'BEFORE (BMES)':<28} {'AFTER (B/O)':<24}")
    print(f"  {'─' * 52}")
    for lbl in sorted(before):
        print(f"    {lbl:<6} {before[lbl]:>12,}")
    print(f"  {'─' * 28}")
    for lbl in sorted(after):
        print(f"  {'':28}  {lbl:<6} {after[lbl]:>12,}")
    print(f"  {'─' * 52}")

    # Verification: E+S should equal B(after), B(before)+M should equal O(after)
    expected_b = before.get("E", 0) + before.get("S", 0)
    expected_o = before.get("B", 0) + before.get("M", 0)
    actual_b = after.get("B", 0)
    actual_o = after.get("O", 0)
    check = "✓" if expected_b == actual_b and expected_o == actual_o else "✗"
    print(
        f"  {check} Verification: E({before.get('E',0):,})+S({before.get('S',0):,})"
        f"=B({actual_b:,}),  "
        f"B({before.get('B',0):,})+M({before.get('M',0):,})=O({actual_o:,})"
    )
    total = sum(before.values())
    print(f"  Total labels: {total:,}\n")


# ============================================================================
# SINGLE-FILE CONVERSION (worker function)
# ============================================================================

def convert_parquet_bmes_to_ob(
    input_path: str,
    output_path: str | None = None,
    overwrite: bool = False,
) -> Tuple[str, int, Counter, Counter]:
    """Read a parquet file, convert its 'labels' column from BMES to B/O,
    and write the result to a new (or the same) parquet file.

    Returns (output_path, num_rows, before_counts, after_counts).
    """
    t0 = time.perf_counter()
    input_path = Path(input_path)
    table = pq.read_table(str(input_path))
    num_rows = table.num_rows

    # --- Count labels BEFORE ---
    labels_col = table.column("labels")
    before_counts = count_labels_fast(labels_col)

    # --- Vectorised conversion ---
    converted = [bmes_to_ob_numpy(row.as_py()) for row in labels_col]

    # --- Count labels AFTER ---
    after_counts: Counter = Counter()
    for seq in converted:
        after_counts.update(seq)

    # --- Print per-file stats ---
    print_label_stats(before_counts, after_counts, input_path.name)

    # --- Replace column & write ---
    idx = table.schema.get_field_index("labels")
    new_col = pa.array(converted, type=pa.list_(pa.string()))
    table = table.set_column(idx, "labels", new_col)

    if output_path is not None:
        out = Path(output_path)
    elif overwrite:
        out = input_path
    else:
        out = input_path.with_stem(input_path.stem + "_ob")

    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out), compression="snappy")

    elapsed = time.perf_counter() - t0
    print(f"  ✓ {input_path.name} → {out.name}  ({num_rows:,} rows, {elapsed:.2f}s)")

    return str(out), num_rows, before_counts, after_counts


# Wrapper for multiprocessing (must be top-level picklable).
def _convert_worker(args: Tuple[str, str | None, bool]) -> Tuple[str, int, Counter, Counter]:
    path, output_path, overwrite = args
    return convert_parquet_bmes_to_ob(path, output_path=output_path, overwrite=overwrite)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert parquet label columns from BMES to B/O scheme."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_file", type=str,
        help="Single parquet file to convert.",
    )
    group.add_argument(
        "--input_dir", type=str,
        help="Root data directory containing split folders with parquet files.",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="Split folders to process (only used with --input_dir).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help=(
            "Output root directory. Converted files are written to "
            "output_dir/train/, output_dir/val/, output_dir/test/ "
            "with the same filenames. If omitted, files are written "
            "next to originals with an _ob suffix (or overwritten "
            "if --overwrite is set)."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite original files in-place (ignored when --output_dir is set).",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: all available CPU cores).",
    )
    args = parser.parse_args()

    num_workers = args.workers or os.cpu_count() or 1
    t_start = time.perf_counter()

    if args.input_file:
        out = None
        if args.output_dir:
            out = str(Path(args.output_dir) / Path(args.input_file).name)
        convert_parquet_bmes_to_ob(
            args.input_file, output_path=out, overwrite=args.overwrite
        )
    else:
        # Collect all parquet files across requested splits.
        data_dir = Path(args.input_dir)
        # (input_path, output_path_or_None, overwrite)
        all_files: List[Tuple[str, str | None, bool]] = []
        for split in args.splits:
            split_dir = data_dir / split
            if not split_dir.exists():
                print(f"  ⚠ Split folder not found, skipping: {split_dir}")
                continue
            parquet_files = sorted(split_dir.glob("*.parquet"))
            if not parquet_files:
                print(f"  ⚠ No parquet files in {split_dir}")
                continue
            for pf in parquet_files:
                if args.output_dir:
                    # Mirror: output_dir / split / filename.parquet
                    out_path = str(Path(args.output_dir) / split / pf.name)
                else:
                    out_path = None
                all_files.append((str(pf), out_path, args.overwrite))

        if not all_files:
            print("No parquet files found. Nothing to do.")
            return

        print(f"\n🚀 Processing {len(all_files)} file(s) with {num_workers} worker(s)...\n")

        # --- Aggregate stats ---
        total_before: Counter = Counter()
        total_after: Counter = Counter()
        total_rows = 0

        if num_workers == 1 or len(all_files) == 1:
            # Sequential (avoids multiprocessing overhead for single files).
            for file_args in all_files:
                _, n, bc, ac = _convert_worker(file_args)
                total_rows += n
                total_before += bc
                total_after += ac
        else:
            with mp.Pool(processes=min(num_workers, len(all_files))) as pool:
                results = pool.map(_convert_worker, all_files)
            for _, n, bc, ac in results:
                total_rows += n
                total_before += bc
                total_after += ac

        # --- Grand total ---
        elapsed = time.perf_counter() - t_start
        print("\n" + "=" * 60)
        print(f"  🎉 ALL DONE — {len(all_files)} files, {total_rows:,} rows, {elapsed:.2f}s")
        print_label_stats(total_before, total_after, "TOTAL (all files)")


if __name__ == "__main__":
    main()
