"""05 - Clean test-set evaluation.

Reported per model:
  accuracy, macro-F1, weighted-F1, ROC-AUC, PR-AUC (average precision),
  recall on the toxic class, and the raw false-positive count on non-toxic.

Uncertainty: 95% percentile bootstrap CI on macro-F1, 1000 resamples, seed 42.
Resamples are drawn once and shared across models, so the intervals are
directly comparable and the paired McNemar tests below use the same items.

Significance: pairwise McNemar on the test predictions via statsmodels. We
pick the variant per pair by the standard rule and record which was used:
  - discordant pairs (b + c) < 25  -> exact binomial test
  - otherwise                      -> chi-square with continuity correction
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import predict as P

N_BOOT = 1000
DISCORDANT_EXACT_CUTOFF = 25
MODELS = ("LR", "NB", "RF", "DistilBERT")


def metrics(y, prob, threshold=0.5):
    from sklearn.metrics import (
        accuracy_score, average_precision_score, confusion_matrix,
        f1_score, recall_score, roc_auc_score,
    )

    pred = (np.asarray(prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "recall_toxic": float(recall_score(y, pred, pos_label=1)),
        "precision_toxic": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "fpr_non_toxic": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "n_non_toxic": int(tn + fp),
        "n_toxic": int(tp + fn),
    }


def bootstrap_macro_f1(y, pred, idx_matrix):
    """95% percentile CI over pre-drawn resample indices."""
    from sklearn.metrics import f1_score

    y = np.asarray(y)
    pred = np.asarray(pred)
    scores = np.empty(len(idx_matrix))
    for i, idx in enumerate(idx_matrix):
        yi = y[idx]
        # a resample containing a single class would make macro-F1 undefined
        if yi.min() == yi.max():
            scores[i] = np.nan
            continue
        scores[i] = f1_score(yi, pred[idx], average="macro")
    ok = scores[~np.isnan(scores)]
    return {
        "mean": float(ok.mean()),
        "std": float(ok.std(ddof=1)),
        "ci_low": float(np.percentile(ok, 2.5)),
        "ci_high": float(np.percentile(ok, 97.5)),
        "n_resamples": int(len(ok)),
        "n_degenerate_skipped": int(len(scores) - len(ok)),
    }


def mcnemar_pair(y, pred_a, pred_b):
    from statsmodels.stats.contingency_tables import mcnemar

    y = np.asarray(y)
    a_ok = np.asarray(pred_a) == y
    b_ok = np.asarray(pred_b) == y
    # 2x2: rows = A correct/incorrect, cols = B correct/incorrect
    n11 = int(np.sum(a_ok & b_ok))
    n12 = int(np.sum(a_ok & ~b_ok))   # b = A right, B wrong
    n21 = int(np.sum(~a_ok & b_ok))   # c = A wrong, B right
    n22 = int(np.sum(~a_ok & ~b_ok))
    discordant = n12 + n21
    use_exact = discordant < DISCORDANT_EXACT_CUTOFF
    res = mcnemar([[n11, n12], [n21, n22]], exact=use_exact,
                  correction=not use_exact)
    return {
        "table": {"both_correct": n11, "only_A_correct": n12,
                  "only_B_correct": n21, "both_wrong": n22},
        "discordant": discordant,
        "test": "exact binomial" if use_exact else
                "chi-square with continuity correction",
        "exact": bool(use_exact),
        "continuity_correction": bool(not use_exact),
        "statistic": float(res.statistic) if res.statistic is not None else None,
        "pvalue": float(res.pvalue),
        "significant_at_0.05": bool(res.pvalue < 0.05),
    }


def main():
    C.set_seed()
    C.announce_label_set()
    test = pd.read_csv(C.PROC / "test_prep.csv")
    y = test[C.label_col()].values
    raw = test["utterance"].astype(str).tolist()

    C.banner("05 - EVALUATE: predictions on the clean test set")
    print("  test n={}  toxic={}  non-toxic={}".format(
        len(y), int(y.sum()), int(len(y) - y.sum())))

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  WARNING: CUDA unavailable; DistilBERT inference on CPU.")
    probs = P.all_probas(raw, device=device)
    preds = {k: P.to_pred(v) for k, v in probs.items()}

    np.savez_compressed(
        C.CKPT / ("test_probs" + C.suffix() + ".npz"),
        y=y, ids=test["Id"].values,
        **{k: v for k, v in probs.items()},
    )
    print("  [saved] checkpoints/test_probs{}.npz (reused by 08/09)".format(C.suffix()))

    C.banner("05 - EVALUATE: main metrics (threshold 0.5)")
    rows = {}
    hdr = ("{:<12}{:>9}{:>10}{:>11}{:>9}{:>9}{:>10}{:>7}".format(
        "model", "acc", "macroF1", "weightF1", "ROC-AUC", "PR-AUC",
        "rec_tox", "FP"))
    print(hdr)
    print("-" * len(hdr))
    for m in MODELS:
        r = metrics(y, probs[m])
        rows[m] = r
        print("{:<12}{:>9.4f}{:>10.4f}{:>11.4f}{:>9.4f}{:>9.4f}{:>10.4f}{:>7}".format(
            m, r["accuracy"], r["macro_f1"], r["weighted_f1"], r["roc_auc"],
            r["pr_auc"], r["recall_toxic"], r["false_positives"]))
    print("\n  FP = false positives among {} non-toxic utterances".format(
        rows["LR"]["n_non_toxic"]))

    C.banner("05 - EVALUATE: 95% bootstrap CI on macro-F1 "
             "({} resamples, seed {})".format(N_BOOT, C.SEED))
    rng = np.random.default_rng(C.SEED)
    idx_matrix = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    boots = {}
    for m in MODELS:
        b = bootstrap_macro_f1(y, preds[m], idx_matrix)
        boots[m] = b
        print("  {:<12} macro-F1 {:.4f}  95% CI [{:.4f}, {:.4f}]  "
              "(width {:.4f})".format(
                  m, rows[m]["macro_f1"], b["ci_low"], b["ci_high"],
                  b["ci_high"] - b["ci_low"]))
    print("\n  Resample indices are shared across models "
          "(paired bootstrap), seed {}.".format(C.SEED))

    C.banner("05 - EVALUATE: pairwise McNemar")
    print("  rule: discordant < {} -> exact binomial; "
          "else chi-square + continuity correction\n".format(
              DISCORDANT_EXACT_CUTOFF))
    mc = {}
    print("{:<26}{:>12}{:>14}{:>12}  {}".format(
        "pair", "discordant", "statistic", "p-value", "test"))
    print("-" * 88)
    for a, b in itertools.combinations(MODELS, 2):
        r = mcnemar_pair(y, preds[a], preds[b])
        mc["{} vs {}".format(a, b)] = r
        stat = "n/a" if r["statistic"] is None else "{:.3f}".format(r["statistic"])
        print("{:<26}{:>12}{:>14}{:>12.3e}  {}{}".format(
            "{} vs {}".format(a, b), r["discordant"], stat, r["pvalue"],
            r["test"], "  *" if r["significant_at_0.05"] else ""))
    print("\n  * = significant at alpha=0.05")

    C.save_json(
        {
            "label_set": C.LABEL_SET,
            "label_rule": C.label_rule(),
            "n_test": int(len(y)),
            "threshold": 0.5,
            "metrics": rows,
            "bootstrap": {
                "n_resamples": N_BOOT, "seed": C.SEED,
                "method": "percentile, paired (shared resample indices)",
                "per_model": boots,
            },
            "mcnemar": {
                "rule": "exact binomial if discordant < {} else chi-square "
                        "with continuity correction".format(
                            DISCORDANT_EXACT_CUTOFF),
                "library": "statsmodels.stats.contingency_tables.mcnemar",
                "pairs": mc,
            },
            "device_distilbert": device,
        },
        C.RESULTS / ("05_evaluate" + C.suffix() + ".json"),
        "05_evaluate" + C.suffix(),
    )


if __name__ == "__main__":
    main()
