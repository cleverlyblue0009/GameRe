"""01 - Acquire CONDA, resolve the binary label mapping, build the splits.

CONDA (Weld et al., ACL-IJCNLP Findings 2021) ships three CSVs. Only two are
usable here:

  CONDA_train.csv  -> stratified 80/20 (seed 42) -> train / val
  CONDA_valid.csv  -> held-out TEST set
  CONDA_test.csv   -> UNLABELLED (blind CodaLab competition set) -> NOT USED

The binary toxic/non-toxic label is not given directly; it has to be derived
from the utterance-level `intentClass` and/or the token-level `slotClasses`.
This script enumerates candidate mappings and reports the counts each one
produces on the test set, so the choice is auditable rather than asserted.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

BASE_URL = "https://raw.githubusercontent.com/usydnlp/CONDA/main/data"
FILES = ("CONDA_train.csv", "CONDA_valid.csv", "CONDA_test.csv")

# The original study reported these test-set counts. The mapping we adopt must
# reproduce them exactly, or we stop and report.
TARGET_TOXIC, TARGET_NONTOXIC = 2345, 6628


def download():
    import urllib.request

    C.RAW.mkdir(parents=True, exist_ok=True)
    for fname in FILES:
        dest = C.RAW / fname
        if dest.exists() and dest.stat().st_size > 0:
            print("  [cached] " + fname)
            continue
        url = BASE_URL + "/" + fname
        print("  [get]    {} <- {}".format(fname, url))
        urllib.request.urlretrieve(url, dest)


def clean(df):
    """Drop rows with a missing/blank utterance (nothing to classify)."""
    before = len(df)
    df = df.dropna(subset=["utterance"]).copy()
    df = df[df["utterance"].astype(str).str.strip() != ""].copy()
    return df, before - len(df)


def candidate_mappings():
    """Every mapping we consider, so the report is honest about the search."""
    cands = []
    for r in range(1, 4):
        for combo in itertools.combinations(("E", "A", "I"), r):
            name = "intentClass in {" + ",".join(combo) + "}"
            cands.append((name, lambda d, c=combo: d["intentClass"].isin(c)))

    def has_slot(d, tags):
        return d["slotClasses"].fillna("").apply(
            lambda s: bool(set(str(s).split()) & set(tags))
        )

    cands += [
        ("token-level: slotClasses has T", lambda d: has_slot(d, {"T"})),
        ("token-level: slotClasses has T|C", lambda d: has_slot(d, {"T", "C"})),
        ("token-level: slotClasses has T|D", lambda d: has_slot(d, {"T", "D"})),
        ("token-level: slotClasses has T|D|S", lambda d: has_slot(d, {"T", "D", "S"})),
        (
            "intentClass != O  OR  slot has T",
            lambda d: (d["intentClass"] != "O") | has_slot(d, {"T"}),
        ),
        (
            "intentClass != O  AND  slot has T",
            lambda d: (d["intentClass"] != "O") & has_slot(d, {"T"}),
        ),
    ]
    return cands


def resolve_mapping(test):
    C.banner(
        "LABEL MAPPING SEARCH  (target: {} toxic / {} non-toxic)".format(
            TARGET_TOXIC, TARGET_NONTOXIC
        )
    )
    header = "{:<40}{:>7}{:>9}{:>7}  verdict".format(
        "candidate mapping", "toxic", "non-tox", "total"
    )
    print(header)
    print("-" * 75)

    report, matches = [], []
    for name, fn in candidate_mappings():
        y = fn(test).astype(int)
        tox, non = int(y.sum()), int(len(y) - y.sum())
        ok = (tox, non) == (TARGET_TOXIC, TARGET_NONTOXIC)
        if ok:
            matches.append(name)
        print(
            "{:<40}{:>7}{:>9}{:>7}  {}".format(
                name, tox, non, len(y), "<== EXACT MATCH" if ok else ""
            )
        )
        report.append(
            {
                "mapping": name,
                "toxic": tox,
                "non_toxic": non,
                "total": int(len(y)),
                "matches_target": ok,
            }
        )

    if not matches:
        print("\nFATAL: no candidate mapping reproduces the target counts.")
        print("Stopping as instructed - the label definition must be resolved first.")
        sys.exit(2)

    # Prefer the canonical phrasing when several equivalent forms match.
    canonical = "intentClass in {E,A,I}"
    chosen = canonical if canonical in matches else matches[0]
    print("\nADOPTED (primary): {}   i.e. toxic = intentClass is not 'O'".format(chosen))
    print("Equivalent matching forms: {}".format(matches))
    return chosen, report


def add_labels(df):
    df = df.copy()
    df["label"] = df["intentClass"].apply(C.label_primary).astype(int)
    # cheap to carry, and lets the E|I sensitivity analysis run without
    # re-deriving anything downstream
    df["label_EI"] = df["intentClass"].apply(C.label_ei).astype(int)
    return df


def describe(name, df):
    n = len(df)
    tox = int(df["label"].sum())
    tox_ei = int(df["label_EI"].sum())
    stats = {
        "split": name,
        "n": n,
        "toxic": tox,
        "non_toxic": n - tox,
        "toxic_pct": round(100.0 * tox / n, 2),
        "toxic_EI": tox_ei,
        "toxic_EI_pct": round(100.0 * tox_ei / n, 2),
        "intent_counts": df["intentClass"].value_counts().to_dict(),
    }
    print(
        "  {:<6} n={:>6}  toxic={:>5} ({:>5.2f}%)  non-toxic={:>5}   "
        "[E|I-only toxic={:>5} ({:>5.2f}%)]".format(
            name, n, tox, stats["toxic_pct"], n - tox, tox_ei, stats["toxic_EI_pct"]
        )
    )
    return stats


def main():
    C.set_seed()
    from sklearn.model_selection import train_test_split

    C.banner("01 - DATA: download")
    download()

    raw_train = pd.read_csv(C.RAW / "CONDA_train.csv")
    raw_test = pd.read_csv(C.RAW / "CONDA_valid.csv")
    raw_blind = pd.read_csv(C.RAW / "CONDA_test.csv")

    print("\nCONDA_train.csv rows : {}".format(len(raw_train)))
    print("CONDA_valid.csv rows : {}   (used as TEST)".format(len(raw_test)))
    print(
        "CONDA_test.csv  rows : {}   -> UNLABELLED, columns={}".format(
            len(raw_blind), list(raw_blind.columns)
        )
    )
    assert (
        "intentClass" not in raw_blind.columns
    ), "CONDA_test.csv unexpectedly has labels - re-check the source"
    print("  confirmed: CONDA_test.csv carries no intentClass -> excluded")

    trainfull, drop_tr = clean(raw_train)
    test, drop_te = clean(raw_test)
    print("\ndropped blank/NaN utterances: train={}, test={}".format(drop_tr, drop_te))

    test = add_labels(test)
    chosen, mapping_report = resolve_mapping(test)

    trainfull = add_labels(trainfull)

    C.banner("01 - DATA: splits")
    train, val = train_test_split(
        trainfull,
        test_size=0.20,
        random_state=C.SEED,
        stratify=trainfull["label"],
        shuffle=True,
    )
    train, val = train.copy(), val.copy()

    stats = [describe("train", train), describe("val", val), describe("test", test)]

    cols = ["Id", "utterance", "intentClass", "slotClasses", "label", "label_EI"]
    for name, df in (("train", train), ("val", val), ("test", test)):
        out = C.PROC / (name + ".csv")
        df[cols].to_csv(out, index=False)
        print("[saved] {} -> {}  ({} rows)".format(name, out.relative_to(C.ROOT), len(df)))

    # no leakage across splits
    ids = {
        k: set(v["Id"])
        for k, v in (("train", train), ("val", val), ("test", test))
    }
    assert not ids["train"] & ids["val"], "train/val Id overlap"
    print("\nintegrity: train/val Id sets disjoint OK")
    print(
        "note: train/val come from CONDA_train.csv and test from "
        "CONDA_valid.csv, so they are disjoint by construction"
    )

    C.save_json(
        {
            "target_counts": {"toxic": TARGET_TOXIC, "non_toxic": TARGET_NONTOXIC},
            "adopted_mapping": chosen,
            "adopted_mapping_rule": "toxic = 1 if intentClass != 'O' else 0",
            "sensitivity_mapping_rule": "toxic = 1 if intentClass in {E,I} else 0",
            "candidates": mapping_report,
            "dropped_blank_utterances": {"train": drop_tr, "test": drop_te},
            "rows_raw": {
                "CONDA_train.csv": len(raw_train),
                "CONDA_valid.csv": len(raw_test),
                "CONDA_test.csv": len(raw_blind),
            },
            "splits": stats,
            "split_config": {
                "test_size": 0.20,
                "seed": C.SEED,
                "stratify": "label",
                "source": "CONDA_train.csv",
            },
        },
        C.RESULTS / "01_data.json",
        "01_data",
    )


if __name__ == "__main__":
    main()
