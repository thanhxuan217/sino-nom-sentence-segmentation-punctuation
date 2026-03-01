#!/usr/bin/env python3
"""
Convert parquet label columns from BMES scheme to B/O scheme.

BMES → B/O mapping (following infer_guwen.py logic):
  E (end of sentence)     → B  (sentence boundary)
  S (single-char sentence) → B  (sentence boundary)
  B (beginning of sentence) → O
  M (middle of sentence)    → O

Usage:
    python convert_bmes_to_ob.py --input_dir data --splits train val test
    python convert_bmes_to_ob.py --input_file data/test/segmentation_sliding_test.parquet
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


def bmes_to_ob(bmes_labels: List[str]) -> List[str]:
    """Convert a sequence of BMES labels to O/B labels.

    E, S → B  (sentence boundary / punctuation position)
    B, M → O  (no boundary)
    """
    return ["B" if lbl in ("E", "S") else "O" for lbl in bmes_labels]


def count_labels(labels_column) -> Counter:
    """Count label occurrences across all rows of a labels column."""
    counter: Counter = Counter()
    for row in labels_column:
        counter.update(row.as_py())
    return counter


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
    print(f"  {check} Verification: E({before.get('E',0):,})+S({before.get('S',0):,})"
          f"=B({actual_b:,}), "
          f"B({before.get('B',0):,})+M({before.get('M',0):,})=O({actual_o:,})")
    total = sum(before.values())
    print(f"  Total labels: {total:,}\n")


def convert_parquet_bmes_to_ob(
    input_path: str,
    output_path: str | None = None,
    overwrite: bool = False,
) -> str:
    """Read a parquet file, convert its 'labels' column from BMES to B/O,
    and write the result to a new (or the same) parquet file.

    Also prints label distribution statistics before and after conversion.

    Parameters
    ----------
    input_path : str
        Path to the source parquet file (must contain 'text' and 'labels' columns).
    output_path : str or None
        Destination path. If None, defaults to ``<stem>_ob.parquet`` next to the
        input file. If *overwrite* is True and *output_path* is None, the input
        file is overwritten in-place.
    overwrite : bool
        If True and *output_path* is None, overwrite the original file.

    Returns
    -------
    str
        The path of the written file.
    """
    input_path = Path(input_path)
    table = pq.read_table(str(input_path))

    # Count labels BEFORE conversion
    labels_col = table.column("labels")
    before_counts = count_labels(labels_col)

    # Convert labels
    converted = [bmes_to_ob(row.as_py()) for row in labels_col]

    # Count labels AFTER conversion
    after_counts: Counter = Counter()
    for seq in converted:
        after_counts.update(seq)

    # Print comparison
    print_label_stats(before_counts, after_counts, input_path.name)

    # Replace the labels column
    idx = table.schema.get_field_index("labels")
    new_col = pa.array(converted, type=pa.list_(pa.string()))
    table = table.set_column(idx, "labels", new_col)

    # Determine output path
    if output_path is not None:
        out = Path(output_path)
    elif overwrite:
        out = input_path
    else:
        out = input_path.with_stem(input_path.stem + "_ob")

    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out))
    print(f"  ✓ {input_path} → {out}  ({table.num_rows} rows)")
    return str(out)


def main():
    parser = argparse.ArgumentParser(
        description="Convert parquet label columns from BMES to B/O scheme."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_file",
        type=str,
        help="Single parquet file to convert.",
    )
    group.add_argument(
        "--input_dir",
        type=str,
        help="Root data directory containing split folders with parquet files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Split folders to process (only used with --input_dir).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite original files instead of creating *_ob.parquet copies.",
    )
    args = parser.parse_args()

    if args.input_file:
        convert_parquet_bmes_to_ob(args.input_file, overwrite=args.overwrite)
    else:
        data_dir = Path(args.input_dir)
        for split in args.splits:
            split_dir = data_dir / split
            if not split_dir.exists():
                print(f"  ⚠ Split folder not found, skipping: {split_dir}")
                continue
            parquet_files = sorted(split_dir.glob("*.parquet"))
            if not parquet_files:
                print(f"  ⚠ No parquet files in {split_dir}")
                continue
            print(f"\n[{split}] Converting {len(parquet_files)} file(s)...")
            for pf in parquet_files:
                convert_parquet_bmes_to_ob(str(pf), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
