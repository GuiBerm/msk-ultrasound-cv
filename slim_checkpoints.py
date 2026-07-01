#!/usr/bin/env python3
"""Strip optimizer/scheduler state from all .pth checkpoints.

Reduces each checkpoint from ~574 MB to ~250 MB (only model weights are kept).
The slim files are safe for inference; they are NOT suitable for resuming training.

Usage:
    # Dry run — only prints what would happen, touches nothing
    python slim_checkpoints.py --dir artifacts/models --dry-run

    # In-place: overwrites originals (originals are backed up with .bak extension)
    python slim_checkpoints.py --dir artifacts/models

    # Output to a separate directory (originals untouched)
    python slim_checkpoints.py --dir artifacts/models --out-dir slim_models
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


# Keys to KEEP in the slim checkpoint (everything else is dropped)
KEEP_KEYS = {'epoch', 'fold', 'backbone_name', 'val_loss', 'mean_qwk', 'model_state'}


def slim_checkpoint(src: Path, dst: Path, dry_run: bool) -> tuple[int, int]:
    """Load *src*, strip heavy state, save to *dst*.

    Returns (original_bytes, slim_bytes).
    """
    original_bytes = src.stat().st_size

    if dry_run:
        ckpt = torch.load(src, map_location='cpu', weights_only=False)
        dropped = sorted(set(ckpt.keys()) - KEEP_KEYS)
        print(f"  [dry-run] would drop keys: {dropped}")
        return original_bytes, -1

    ckpt = torch.load(src, map_location='cpu', weights_only=False)
    slim = {k: v for k, v in ckpt.items() if k in KEEP_KEYS}
    torch.save(slim, dst)

    slim_bytes = dst.stat().st_size
    return original_bytes, slim_bytes


def fmt_mb(n_bytes: int) -> str:
    return f"{n_bytes / 1024**2:.1f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Slim down .pth checkpoints by stripping optimizer state.")
    parser.add_argument('--dir', type=str, default='artifacts/models',
                        help='Root directory to search for .pth files (searched recursively).')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory for slim checkpoints. '
                             'Mirrors the source tree. '
                             'If omitted, files are overwritten in-place (originals backed up as .bak).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would happen without writing anything.')
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.exists():
        raise SystemExit(f"ERROR: directory not found: {root}")

    checkpoints = sorted(root.rglob('*.pth'))
    if not checkpoints:
        print(f"No .pth files found under {root}")
        return

    print(f"Found {len(checkpoints)} checkpoint(s) under '{root}'")
    if args.dry_run:
        print("*** DRY RUN — nothing will be written ***\n")

    total_before = 0
    total_after = 0

    for src in checkpoints:
        print(f"\n{src.relative_to(root)}")
        print(f"  Before: {fmt_mb(src.stat().st_size)}")

        if args.out_dir:
            # Mirror directory structure under out_dir
            rel = src.relative_to(root)
            dst = Path(args.out_dir) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
        else:
            # In-place: back up original then overwrite
            dst = src
            if not args.dry_run:
                bak = src.with_suffix('.pth.bak')
                shutil.copy2(src, bak)
                print(f"  Backup:  {bak.name}")

        before, after = slim_checkpoint(src, dst, dry_run=args.dry_run)
        total_before += before

        if not args.dry_run:
            saved = before - after
            ratio = before / after if after else 0
            print(f"  After:   {fmt_mb(after)}  (saved {fmt_mb(saved)}, {ratio:.2f}× smaller)")
            total_after += after

    print("\n" + "=" * 50)
    if args.dry_run:
        print(f"Total size (current): {fmt_mb(total_before)}")
    else:
        saved_total = total_before - total_after
        ratio_total = total_before / total_after if total_after else 0
        print(f"Total before: {fmt_mb(total_before)}")
        print(f"Total after:  {fmt_mb(total_after)}")
        print(f"Space saved:  {fmt_mb(saved_total)}  ({ratio_total:.2f}× smaller)")
    print("Done.")


if __name__ == '__main__':
    main()
