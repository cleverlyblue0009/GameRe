"""08 - Two-stage cascade: cheap LR first, DistilBERT only when unsure.

Logistic Regression scores every message. If P(toxic) falls inside an
uncertainty band, the message is escalated to DistilBERT and DistilBERT's
decision is used; otherwise LR's decision stands.

Bands swept: [0.4, 0.7] (default), [0.3, 0.8], [0.2, 0.9].

Reported per band, on clean text and under every script-06 attack:
  escalation rate, macro-F1 with 95% bootstrap CI, and the expected mean
  end-to-end latency with DistilBERT on CPU and on GPU:

      E[latency] = lat_LR + escalation_rate * lat_DistilBERT

which is the number a deployment actually budgets for. Latencies come from
results/07_latency.json (end-to-end, the primary metric).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import predict as P

BANDS = [(0.4, 0.7), (0.3, 0.8), (0.2, 0.9)]
DEFAULT_BAND = (0.4, 0.7)
N_BOOT = 1000


def cascade_pred(p_lr, p_bert, low, high):
    """LR decides unless it is inside the band, where DistilBERT takes over."""
    p_lr = np.asarray(p_lr)
    p_bert = np.asarray(p_bert)
    escalate = (p_lr >= low) & (p_lr <= high)
    pred = (p_lr >= 0.5).astype(int)
    pred[escalate] = (p_bert[escalate] >= 0.5).astype(int)
    return pred, escalate


def boot_ci(y, pred, idx_matrix):
    from sklearn.metrics import f1_score

    y = np.asarray(y)
    pred = np.asarray(pred)
    sc = []
    for idx in idx_matrix:
        yi = y[idx]
        if yi.min() == yi.max():
            continue
        sc.append(f1_score(yi, pred[idx], average="macro"))
    sc = np.asarray(sc)
    return {"ci_low": float(np.percentile(sc, 2.5)),
            "ci_high": float(np.percentile(sc, 97.5)),
            "mean": float(sc.mean())}


def load_latencies():
    """Pull end-to-end single-example means from script 07, if available."""
    path = C.RESULTS / "07_latency.json"
    if not path.exists():
        print("  WARNING: results/07_latency.json missing - latency columns "
              "will be null. Run src/07_latency.py first.")
        return None
    d = C.load_json(path)
    try:
        return {
            "LR": d["cpu_classical"]["LR"]["end_to_end_single"]["mean_ms"],
            "bert_cpu": d["cpu_distilbert"]["end_to_end_single"]["mean_ms"],
            "bert_gpu": d.get("gpu_distilbert", {})
                         .get("fp32", {})
                         .get("end_to_end_single", {})
                         .get("mean_ms"),
        }
    except (KeyError, TypeError):
        print("  WARNING: could not parse 07_latency.json")
        return None


def main():
    C.set_seed()
    C._stopwords()
    import torch
    from sklearn.metrics import f1_score

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    adv = __import__("06_adversarial")

    test = pd.read_csv(C.PROC / "test_prep.csv")
    y = test["label"].values
    raw = test["utterance"].astype(str).tolist()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vec, cmodels = P.load_classical()
    bert, tok, _ = P.load_bert(device=device)
    lat = load_latencies()

    rng = np.random.default_rng(C.SEED)
    idx_matrix = rng.integers(0, len(y), size=(N_BOOT, len(y)))

    # text variants: clean + each attack
    variants = {"clean": raw}
    for atk in adv.ATTACKS:
        texts, _ = adv.build_attacked(test, atk)
        variants[atk] = texts

    C.banner("08 - CASCADE: scoring every variant with LR and DistilBERT")
    scored = {}
    for name, texts in variants.items():
        p_lr = P.classical_proba(texts, vec, cmodels["LR"])
        p_bt = P.bert_proba(texts, bert, tok, device=device)
        scored[name] = (p_lr, p_bt)
        print("  scored: {}".format(name))

    C.banner("08 - CASCADE: results")
    out = {}
    for low, high in BANDS:
        tag = "[{}, {}]".format(low, high)
        out[tag] = {}
        print("\n  band {}{}".format(tag,
              "   <-- default" if (low, high) == DEFAULT_BAND else ""))
        hdr = ("    {:<13}{:>9}{:>10}{:>22}{:>13}{:>13}".format(
            "variant", "escal%", "macroF1", "95% CI", "E[ms] CPU", "E[ms] GPU"))
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for name in variants:
            p_lr, p_bt = scored[name]
            pred, esc = cascade_pred(p_lr, p_bt, low, high)
            rate = float(esc.mean())
            f1 = float(f1_score(y, pred, average="macro"))
            ci = boot_ci(y, pred, idx_matrix)
            e_cpu = e_gpu = None
            if lat:
                e_cpu = lat["LR"] + rate * lat["bert_cpu"]
                if lat["bert_gpu"] is not None:
                    e_gpu = lat["LR"] + rate * lat["bert_gpu"]
            print("    {:<13}{:>8.2f}%{:>10.4f}   [{:.4f}, {:.4f}]{:>13}{:>13}".format(
                name, 100 * rate, f1, ci["ci_low"], ci["ci_high"],
                "n/a" if e_cpu is None else "{:.3f}".format(e_cpu),
                "n/a" if e_gpu is None else "{:.3f}".format(e_gpu)))
            out[tag][name] = {
                "escalation_rate": rate,
                "n_escalated": int(esc.sum()),
                "macro_f1": f1,
                "bootstrap_ci": ci,
                "expected_ms_bert_cpu": e_cpu,
                "expected_ms_bert_gpu": e_gpu,
            }

    # reference points: LR alone and DistilBERT alone on each variant
    C.banner("08 - CASCADE: reference (no cascade)")
    ref = {}
    print("    {:<13}{:>12}{:>14}".format("variant", "LR only", "DistilBERT"))
    print("    " + "-" * 39)
    for name in variants:
        p_lr, p_bt = scored[name]
        a = float(f1_score(y, (p_lr >= 0.5).astype(int), average="macro"))
        b = float(f1_score(y, (p_bt >= 0.5).astype(int), average="macro"))
        ref[name] = {"LR_only_macro_f1": a, "DistilBERT_only_macro_f1": b}
        print("    {:<13}{:>12.4f}{:>14.4f}".format(name, a, b))

    C.save_json(
        {
            "bands": [list(b) for b in BANDS],
            "default_band": list(DEFAULT_BAND),
            "rule": ("LR decides unless low <= P_LR(toxic) <= high, in which "
                     "case DistilBERT decides"),
            "latency_model": "E[latency] = lat_LR + escalation_rate * lat_DistilBERT",
            "latency_source": ("results/07_latency.json, end-to-end "
                               "single-example means (primary metric)"),
            "latencies_used_ms": lat,
            "bootstrap": {"n_resamples": N_BOOT, "seed": C.SEED,
                          "method": "percentile, shared resample indices"},
            "results": out,
            "reference_no_cascade": ref,
        },
        C.RESULTS / "08_cascade.json",
        "08_cascade",
    )


if __name__ == "__main__":
    main()
