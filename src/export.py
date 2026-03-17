#!/usr/bin/env python3
"""
Text Export Utilities
=====================

Reusable functions for exporting stitched document predictions to .txt
files.  Designed to be imported by both batch evaluation scripts
(``evaluate_domain.py``) and future API endpoints.

Typical usage — batch
---------------------
::

    from src.export import prepare_export_dir, export_single_document

    # 1. Create a fresh export directory (clears old results)
    prepare_export_dir("outputs/domain_txt")

    # 2. Export documents one by one (streamed from stitcher)
    #    Files with the same domain/filename are APPENDED automatically.
    for doc in stitched_documents:
        export_single_document(
            pred_text=doc["pred_text"],
            gold_text=doc["gold_text"],
            domain="史藏",
            filename="史记.txt",
            output_dir="outputs/domain_txt",
        )

Typical usage — API (single request)
-------------------------------------
::

    from src.export import export_single_document

    paths = export_single_document(
        pred_text="太史公曰，...",
        gold_text=None,
        domain="史藏",
        filename="史记.txt",
        output_dir="./output",
        mode="w",          # overwrite — one request = one complete file
        write_gold=False,
    )

Directory structure produced::

    outputs/domain_txt/
    ├── pred/
    │   ├── 史藏/
    │   │   ├── 史记.txt        ← complete file (appended from all chunks)
    │   │   └── 汉书.txt
    │   └── 经藏/
    │       └── 周易.txt
    └── gold/
        ├── 史藏/
        │   ├── 史记.txt
        │   └── 汉书.txt
        └── 经藏/
            └── 周易.txt
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional


def sanitize_filename(name: str) -> str:
    """Replace characters that are invalid in file names."""
    for ch in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
        name = name.replace(ch, '_')
    return name.strip() or "unnamed"


def prepare_export_dir(output_dir: str, logger=None) -> None:
    """Create a fresh export directory.

    If the directory already exists, its ``pred/`` and ``gold/``
    sub-trees are deleted so that a new evaluation run does not mix
    results with a previous one.  This is important because
    ``export_single_document`` uses append mode by default.

    Call this **once** before the evaluation loop starts.

    Parameters
    ----------
    output_dir : str
        Root export directory.
    logger : optional
        Logger for info messages.
    """
    out = Path(output_dir)
    for sub in ("pred", "gold"):
        sub_dir = out / sub
        if sub_dir.exists():
            shutil.rmtree(sub_dir)
            if logger:
                logger.info(f"  Cleared old export dir: {sub_dir}")
    out.mkdir(parents=True, exist_ok=True)
    if logger:
        logger.info(f"  ✓ Export directory ready: {output_dir}")


def export_single_document(
    pred_text: str,
    gold_text: Optional[str],
    domain: str,
    filename: str,
    output_dir: str,
    write_gold: bool = True,
    mode: str = "a",
) -> Dict[str, Optional[str]]:
    """Export a single stitched document to .txt files.

    Because a large source file (e.g. 史记.txt) is split into many
    sliding-window chunks and the stitcher may yield multiple documents
    that share the same ``filename``, the default write mode is
    **append** (``mode="a"``).  This ensures all parts of the file are
    concatenated into a single complete .txt output.

    For API use (one request = one complete document), pass
    ``mode="w"`` to overwrite instead.

    Parameters
    ----------
    pred_text : str
        Predicted text (with punctuation / segmentation applied).
    gold_text : str or None
        Ground-truth text.  Ignored when *write_gold* is False.
    domain : str
        Domain category (e.g. "史藏").  Used as subdirectory name.
    filename : str
        Original source filename (e.g. "史记.txt").
    output_dir : str
        Root directory for output.
    write_gold : bool
        Whether to also write the gold text.
    mode : str
        File open mode.  ``"a"`` (default) appends — safe for
        sliding-window chunks that share the same filename.
        ``"w"`` overwrites — use for API / single-document export.

    Returns
    -------
    dict
        ``{"pred": <path>, "gold": <path or None>}``
    """
    domain_safe = sanitize_filename(domain or "unknown")
    fname_safe = sanitize_filename(filename or "unnamed")
    if not fname_safe.endswith(".txt"):
        fname_safe += ".txt"

    paths: Dict[str, Optional[str]] = {"pred": None, "gold": None}

    # --- Predicted text ---
    pred_dir = Path(output_dir) / "pred" / domain_safe
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / fname_safe
    with open(pred_path, mode, encoding="utf-8") as f:
        f.write(pred_text)
    paths["pred"] = str(pred_path)

    # --- Gold text ---
    if write_gold and gold_text is not None:
        gold_dir = Path(output_dir) / "gold" / domain_safe
        gold_dir.mkdir(parents=True, exist_ok=True)
        gold_path = gold_dir / fname_safe
        with open(gold_path, mode, encoding="utf-8") as f:
            f.write(gold_text)
        paths["gold"] = str(gold_path)

    return paths


def export_documents_to_txt(
    documents: List[Dict],
    output_dir: str,
    write_gold: bool = True,
    logger=None,
) -> List[Dict[str, Optional[str]]]:
    """Export a batch of stitched documents to .txt files.

    Calls ``prepare_export_dir`` first, then appends each document.
    Documents sharing the same ``(domain, filename)`` are concatenated
    into a single file — correct for sliding-window pipelines.

    Each element of *documents* should be a dict with at least::

        {
            "domain":    str,
            "filename":  str,
            "pred_text": str,
            "gold_text": str or None,
        }

    Parameters
    ----------
    documents : list of dict
        Stitched document records.
    output_dir : str
        Root export directory.
    write_gold : bool
        Whether to write gold text alongside predictions.
    logger : optional
        Logger for progress messages.

    Returns
    -------
    list of dict
        One ``{"pred": path, "gold": path}`` per document call.
    """
    prepare_export_dir(output_dir, logger=logger)

    all_paths: List[Dict[str, Optional[str]]] = []

    for i, doc in enumerate(documents):
        paths = export_single_document(
            pred_text=doc["pred_text"],
            gold_text=doc.get("gold_text"),
            domain=doc.get("domain", "unknown"),
            filename=doc.get("filename", f"doc_{i:05d}.txt"),
            output_dir=output_dir,
            write_gold=write_gold,
            mode="a",
        )
        all_paths.append(paths)

    if logger:
        # Count unique files written
        unique_pred = set(p["pred"] for p in all_paths if p["pred"])
        unique_gold = set(p["gold"] for p in all_paths if p["gold"])
        logger.info(
            f"  ✓ Exported {len(unique_pred)} predicted .txt files "
            f"({len(all_paths)} documents) to {output_dir}/pred/"
        )
        if write_gold:
            logger.info(
                f"  ✓ Exported {len(unique_gold)} gold .txt files to {output_dir}/gold/"
            )

    return all_paths
