#!/usr/bin/env python
"""Split a case-list csv into N disjoint subsets, one per cluster.

predict_3d.py coordinates concurrent tasks through claim files in the output directory, which
only works among processes that share a filesystem. Across clusters there is nothing to
coordinate through, so the work has to be partitioned up front: give each cluster its own
subset and no case can be predicted twice.

Rows are dealt round-robin rather than in contiguous blocks, so every subset gets the same mix
of datasets, views and image quality -- contiguous blocks would hand one cluster all of camus
and another all of icardio, and their runtimes would differ by more than the split.

The partition is deterministic: re-running reproduces it exactly, so a lost subset can be
regenerated rather than needing the whole thing redone.

Usage:
    python rl4seg3d/scripts/split_case_list.py maple.csv --n 6 --out-dir splits
    python rl4seg3d/scripts/split_case_list.py maple.csv --n 6 --query 'dataset != "camus"'
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="case list to split (e.g. maple.csv)")
    p.add_argument("--n", type=int, default=6, help="number of subsets (default: 6)")
    p.add_argument("--out-dir", default=None,
                   help="where to write the subsets (default: alongside the input csv)")
    p.add_argument("--query", default=None,
                   help="pandas query applied before splitting, e.g. 'dataset == \"camus\"'")
    p.add_argument("--prefix", default=None,
                   help="output basename (default: the input csv's stem)")
    p.add_argument("--blocks", action="store_true",
                   help="split into contiguous blocks instead of dealing round-robin")
    args = p.parse_args()

    if args.n < 1:
        raise SystemExit("--n must be >= 1")

    src = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    prefix = args.prefix or src.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src, low_memory=False)
    print(f"read {len(df)} rows from {src}")
    if args.query:
        before = len(df)
        df = df.query(args.query)
        print(f"query {args.query!r} kept {len(df)}/{before}")
    if df.empty:
        raise SystemExit("nothing to split")

    written = []
    for i in range(args.n):
        if args.blocks:
            lo = round(i * len(df) / args.n)
            hi = round((i + 1) * len(df) / args.n)
            part = df.iloc[lo:hi]
        else:
            part = df.iloc[i::args.n]
        out = out_dir / f"{prefix}_part{i + 1}of{args.n}.csv"
        # index=False: the source csv already carries a pile of Unnamed:0 columns from earlier
        # round-trips, and there is no reason to add another
        part.to_csv(out, index=False)
        written.append((out, part))

    total = sum(len(part) for _, part in written)
    assert total == len(df), f"split lost rows: {total} != {len(df)}"
    ids = pd.concat([part[c] for _, part in written
                     for c in ["dicom_uuid"] if c in part.columns])
    if len(ids):
        assert ids.is_unique, "subsets overlap"

    print(f"\nwrote {args.n} subsets to {out_dir}/  ({total} rows, disjoint)")
    group = "dataset" if "dataset" in df.columns else None
    for out, part in written:
        mix = ""
        if group:
            mix = "  " + " ".join(f"{k}={v}" for k, v in
                                  part[group].value_counts().sort_index().items())
        print(f"  {out.name:<32} {len(part):>7} rows{mix}")


if __name__ == "__main__":
    main()
