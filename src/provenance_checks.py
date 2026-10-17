#!/usr/bin/env python3
"""
What the generator leaves behind: label conditioning and grouping structure.

Reviewer 1 (comments 1 and 7) and Reviewer 2 (comments 3 and 6) asked whether
the class label directly conditioned the raw fields, and whether related records
could appear on both sides of a random split. The generator is not available to
us, so we cannot answer from its source. We can, however, measure what its
output shows, and this script does that.

It reports three things:

1.  Cardinality of every raw attribute. A field with very few distinct values
    across 10,000 records is a template field, not free text.

2.  Label conditioning of the user-agent field: the contingency table of every
    distinct user-agent string against the binary label, flagging any string
    that occurs only in attack traffic. A string that never appears in benign
    traffic is a deterministic giveaway inside the feature set, and any model
    that sees it has been handed the label for those records.

3.  Grouping structure that a random split could leak across: repeated source
    or destination addresses, and repeated timestamps.

Nothing here is a model. The script reads the CSV and counts.

    python src/provenance_checks.py [--data ...]

Writes results/tables/table16_label_conditioning.csv
       results/tables/table16b_field_cardinality.csv
"""

import argparse
import os

import pandas as pd

TAB = "results/tables"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("IDS_DATA", "data/cybersecurity_dataset.csv"))
    args = ap.parse_args()
    os.makedirs(TAB, exist_ok=True)

    df = pd.read_csv(args.data)
    y = df["label"].astype(int)
    n = len(df)

    # ---- 1. field cardinality ----------------------------------------------
    card = pd.DataFrame({
        "Field": df.columns,
        "Distinct values": [df[c].nunique() for c in df.columns],
        "Distinct / records": [round(df[c].nunique() / n, 4) for c in df.columns],
    })
    card.to_csv(f"{TAB}/table16b_field_cardinality.csv", index=False)
    print("=== FIELD CARDINALITY ===")
    print(card.to_string(index=False))

    # ---- 2. label conditioning of the user-agent field ---------------------
    ct = pd.crosstab(df["user_agent"], y).rename(columns={0: "Benign", 1: "Attack"})
    for c in ("Benign", "Attack"):
        if c not in ct:
            ct[c] = 0
    ct = ct[["Benign", "Attack"]]
    ct.insert(0, "Length", [len(s) for s in ct.index])
    ct["Attack-exclusive"] = (ct["Benign"] == 0) & (ct["Attack"] > 0)
    ct = ct.sort_values(["Attack-exclusive", "Attack"], ascending=[False, False])
    ct.reset_index().to_csv(f"{TAB}/table16_label_conditioning.csv", index=False)

    excl = ct[ct["Attack-exclusive"]]
    n_excl = int(excl["Attack"].sum())
    print("\n=== USER-AGENT LABEL CONDITIONING ===")
    print(ct.to_string())
    print(f"\ndistinct user-agent strings: {df['user_agent'].nunique()} across {n:,} records")
    print(f"strings never seen in benign traffic: {len(excl)}")
    print(f"attack records carrying one: {n_excl} of {int(y.sum())} "
          f"({100 * n_excl / y.sum():.2f}% of attacks, {100 * n_excl / n:.2f}% of the dataset)")

    # ---- 3. grouping structure a random split could leak across ------------
    print("\n=== GROUPING STRUCTURE ===")
    for col in ("src_ip", "dst_ip", "timestamp"):
        k = df[col].nunique()
        rep = n - k
        print(f"{col:12s} distinct={k:6,}  records sharing a value with another: {rep:,}")
    print("\nNo host, session or episode identifier is present, so a grouped split "
          "cannot be constructed from these fields; the timestamp supports a "
          "chronological split (see src/temporal_split.py).")


if __name__ == "__main__":
    main()
