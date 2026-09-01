#!/usr/bin/env python
"""Report how far a prediction sweep has got, per case list.

Progress lives in the claim directory <output_path>/csv, keyed by case id: a non-empty file
means the case completed, an empty one means a task holds it (or died holding it). Which cases
belong to which subset is only knowable from the case lists, so this intersects the two.

Usage:
    python rl4seg3d/scripts/predict_progress.py ~/scratch/preds/cheated25k \
        ~/scratch/data/maple_norm/csv_split
    python rl4seg3d/scripts/predict_progress.py <output> a.csv b.csv     # explicit lists
    python rl4seg3d/scripts/predict_progress.py <output> <lists> --failures
"""
import argparse
import os
from pathlib import Path

import pandas as pd

ID_COLUMNS = ("dicom_uuid", "case_identifier", "case_id")


def case_ids(csv):
    df = pd.read_csv(csv, low_memory=False)
    for c in ID_COLUMNS:
        if c in df.columns:
            return set(df[c].dropna().astype(str))
    raise SystemExit(f"no id column in {csv}; looked for {list(ID_COLUMNS)}")


def read_failures(out):
    """All recorded failure attempts, or an empty frame. One file per task, so concatenate."""
    fdir = out / "failures"
    rows = [pd.read_csv(f) for f in sorted(fdir.glob("*.csv"))] if fdir.is_dir() else []
    if not rows:
        return pd.DataFrame(columns=["case_id", "error_type", "error", "retryable"])
    return pd.concat(rows, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", help="output_path of the sweep (the dir holding csv/)")
    p.add_argument("lists", nargs="+", help="case list csv(s), or a directory of them")
    p.add_argument("--retry-list", metavar="OUT.csv", default=None,
                   help="write a case list of just the cases that failed for a reason a bigger "
                        "allocation could fix (OOM, walltime), for a targeted retry tier")
    p.add_argument("--failures", action="store_true",
                   help="also summarise <output>/failures/*.csv by error type")
    args = p.parse_args()

    out = Path(args.output)
    claims = out / "csv"
    files = os.listdir(claims) if claims.is_dir() else []
    # A non-empty claim means the case is settled, but that covers two outcomes: a real result,
    # or a failure row written for a non-retryable error (which is how such a case is marked
    # never to be retried). Separate them, or a growing pile of broken cases reads as progress.
    settled = {f[:-4] for f in files if (claims / f).stat().st_size}
    held = {f[:-4] for f in files if not (claims / f).stat().st_size}

    # retryable=False is exactly the case whose claim gets filled, so the failures log identifies
    # them without reading 46k claim files
    failures = read_failures(out)
    retired = set(failures[~failures.retryable]["case_id"].astype(str)) if len(failures) else set()
    done = settled - retired
    failed = settled & retired

    csvs = []
    for item in args.lists:
        item = Path(item)
        csvs += sorted(item.glob("*.csv")) if item.is_dir() else [item]
    if not csvs:
        raise SystemExit("no case lists found")

    print(f"{'case list':<28}{'done':>16}{'':>8}{'failed':>8}{'held':>8}{'left':>10}")
    tot = [0, 0, 0, 0, 0]
    for csv in csvs:
        ids = case_ids(csv)
        dn, fl, hl = len(ids & done), len(ids & failed), len(ids & held)
        left = len(ids) - dn - fl - hl
        print(f"  {csv.name:<26}{dn:>6}/{len(ids):<7}{100 * dn / len(ids):>7.1f}%"
              f"{fl:>8}{hl:>8}{left:>10}")
        tot = [a + b for a, b in zip(tot, (dn, fl, hl, left, len(ids)))]
    dn, fl, hl, left, n = tot
    print(f"  {'TOTAL':<26}{dn:>6}/{n:<7}{100 * dn / n:>7.1f}%{fl:>8}{hl:>8}{left:>10}")

    # claims present but belonging to no listed case list -- usually a different subset
    unlisted = (settled | held) - set().union(*(case_ids(c) for c in csvs))
    if unlisted:
        print(f"\n{len(unlisted)} claim(s) not in any listed case list")

    if args.retry_list:
        # Only the failures a larger allocation could fix. Cases never attempted are deliberately
        # excluded: a retry tier exists to spend a bigger GPU on the hard cases, not to redo the
        # backlog on it. Rows are carried over whole, so the retry list keeps every column the
        # predict config needs.
        want = set(failures[failures.retryable]["case_id"].astype(str)) - done - failed
        parts = []
        for csv in csvs:
            df = pd.read_csv(csv, low_memory=False)
            col = next(c for c in ID_COLUMNS if c in df.columns)
            parts.append(df[df[col].astype(str).isin(want)])
        retry = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        Path(args.retry_list).parent.mkdir(parents=True, exist_ok=True)
        retry.to_csv(args.retry_list, index=False)
        missing = len(want) - len(retry)
        print(f"\nwrote {len(retry)} case(s) to {args.retry_list}"
              + (f"  ({missing} failed id(s) not in any listed case list)" if missing else ""))

    if args.failures:
        if not len(failures):
            print("\nno failures recorded")
            return
        print(f"\nfailures ({len(failures)} recorded attempts):")
        for (etype, retry), grp in failures.groupby(["error_type", "retryable"]):
            note = "retried once released" if retry else "not retried, counted as failed"
            print(f"  {etype:<24}{len(grp):>6}   ({note})")


if __name__ == "__main__":
    main()
