"""09 - Error analysis and model interpretation.

Two deliverables for the paper's qualitative section:

  1. The 15 most positive and 15 most negative Logistic Regression
     coefficients, mapped back from feature indices to the actual n-grams.
     Because the classical pipeline Porter-stems, the surviving features are
     stems; for each we also list the raw surface forms from the training
     split that produced that stem, so the table is readable.

  2. The 10 highest-confidence false positives and 10 highest-confidence
     false negatives for Random Forest and for DistilBERT, printed in full
     (raw utterance, gold intent class, predicted probability). "Highest
     confidence" means furthest on the wrong side of 0.5.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import predict as P

TOP_K = 15
N_ERRORS = 10


def stem_surface_map(train):
    """stem -> the raw surface forms that produce it (most common first)."""
    m = collections.defaultdict(collections.Counter)
    st = C._stemmer()
    for raw in train["utterance"].astype(str):
        pre = C.preprocess(raw, C.PreprocessConfig(
            lowercase=True, strip_urls_mentions=True, leetspeak=True,
            strip_punctuation=True, remove_stopwords=True, porter_stem=False))
        for tok in pre.split():
            m[st.stem(tok)][tok] += 1
    return m


def show_surface(feature, smap, k=3):
    """For a (possibly bigram) feature, show contributing surface forms."""
    parts = feature.split()
    out = []
    for p in parts:
        forms = smap.get(p)
        if not forms:
            out.append(p)
            continue
        top = [w for w, _ in forms.most_common(k)]
        out.append(p if top == [p] else "{}({})".format(p, "/".join(top)))
    return " ".join(out)


def lr_coefficients(vec, lr, smap):
    names = np.asarray(vec.get_feature_names_out())
    coef = lr.coef_.ravel()
    order = np.argsort(coef)
    neg_idx = order[:TOP_K]
    pos_idx = order[::-1][:TOP_K]

    C.banner("09 - ANALYSIS: Logistic Regression coefficients")
    print("  Positive coefficients push towards TOXIC; negative towards "
          "NON-TOXIC.")
    print("  Features are Porter stems; bracketed forms are the training-set")
    print("  surface words that map onto that stem.\n")

    print("  TOP {} POSITIVE (-> toxic)".format(TOP_K))
    print("  {:>5}  {:>9}  {:<16} {}".format("rank", "coef", "feature",
                                             "surface forms"))
    print("  " + "-" * 72)
    pos = []
    for r, i in enumerate(pos_idx, 1):
        print("  {:>5}  {:>9.4f}  {:<16} {}".format(
            r, coef[i], names[i], show_surface(names[i], smap)))
        pos.append({"rank": r, "feature": str(names[i]),
                    "coefficient": float(coef[i]),
                    "surface_forms": show_surface(names[i], smap)})

    print("\n  TOP {} NEGATIVE (-> non-toxic)".format(TOP_K))
    print("  {:>5}  {:>9}  {:<16} {}".format("rank", "coef", "feature",
                                             "surface forms"))
    print("  " + "-" * 72)
    neg = []
    for r, i in enumerate(neg_idx, 1):
        print("  {:>5}  {:>9.4f}  {:<16} {}".format(
            r, coef[i], names[i], show_surface(names[i], smap)))
        neg.append({"rank": r, "feature": str(names[i]),
                    "coefficient": float(coef[i]),
                    "surface_forms": show_surface(names[i], smap)})

    n_uni = sum(1 for f in names if " " not in f)
    print("\n  (vocabulary: {} features, {} unigrams / {} bigrams)".format(
        len(names), n_uni, len(names) - n_uni))
    return {"top_positive": pos, "top_negative": neg}


def error_analysis(test, probs, model_name):
    """Highest-confidence FPs and FNs, printed in full."""
    y = test["label"].values
    p = np.asarray(probs)
    pred = (p >= 0.5).astype(int)

    fp = np.where((pred == 1) & (y == 0))[0]
    fn = np.where((pred == 0) & (y == 1))[0]
    fp = fp[np.argsort(-p[fp])][:N_ERRORS]   # most confidently wrong
    fn = fn[np.argsort(p[fn])][:N_ERRORS]

    C.banner("09 - ANALYSIS: {} - highest-confidence errors".format(model_name))
    out = {"false_positives": [], "false_negatives": []}

    print("  FALSE POSITIVES (gold NON-toxic, predicted toxic) - "
          "{} shown of {}".format(len(fp), int(((pred == 1) & (y == 0)).sum())))
    for r, i in enumerate(fp, 1):
        row = test.iloc[i]
        print("   {:>3}. P(toxic)={:.4f}  intent={}  id={}".format(
            r, p[i], row["intentClass"], row["Id"]))
        print("        utterance : {}".format(repr(str(row["utterance"]))))
        print("        slots     : {}".format(str(row.get("slotClasses"))[:100]))
        print("        prep      : {}".format(repr(str(row["text_classical"]))))
        out["false_positives"].append({
            "rank": r, "id": int(row["Id"]), "p_toxic": float(p[i]),
            "intent_class": str(row["intentClass"]),
            "utterance": str(row["utterance"]),
            "slot_classes": str(row.get("slotClasses")),
            "preprocessed": str(row["text_classical"]),
        })

    print("\n  FALSE NEGATIVES (gold TOXIC, predicted non-toxic) - "
          "{} shown of {}".format(len(fn), int(((pred == 0) & (y == 1)).sum())))
    for r, i in enumerate(fn, 1):
        row = test.iloc[i]
        print("   {:>3}. P(toxic)={:.4f}  intent={}  id={}".format(
            r, p[i], row["intentClass"], row["Id"]))
        print("        utterance : {}".format(repr(str(row["utterance"]))))
        print("        slots     : {}".format(str(row.get("slotClasses"))[:100]))
        print("        prep      : {}".format(repr(str(row["text_classical"]))))
        out["false_negatives"].append({
            "rank": r, "id": int(row["Id"]), "p_toxic": float(p[i]),
            "intent_class": str(row["intentClass"]),
            "utterance": str(row["utterance"]),
            "slot_classes": str(row.get("slotClasses")),
            "preprocessed": str(row["text_classical"]),
        })

    # which gold intent classes do the errors concentrate in?
    all_fp = np.where((pred == 1) & (y == 0))[0]
    all_fn = np.where((pred == 0) & (y == 1))[0]
    fp_by = test.iloc[all_fp]["intentClass"].value_counts().to_dict()
    fn_by = test.iloc[all_fn]["intentClass"].value_counts().to_dict()
    print("\n  all FPs by gold intent: {}".format(fp_by))
    print("  all FNs by gold intent: {}".format(fn_by))
    out["fp_by_intent"] = fp_by
    out["fn_by_intent"] = fn_by
    out["n_fp_total"] = int(len(all_fp))
    out["n_fn_total"] = int(len(all_fn))
    return out


def main():
    C.set_seed()
    C._stopwords()

    train = pd.read_csv(C.PROC / "train_prep.csv")
    test = pd.read_csv(C.PROC / "test_prep.csv").fillna({"text_classical": ""})
    vec, cmodels = P.load_classical()

    print("building stem -> surface-form map from the training split...")
    smap = stem_surface_map(train)
    coefs = lr_coefficients(vec, cmodels["LR"], smap)

    # reuse the cached probabilities from script 05 when available
    cache = C.CKPT / "test_probs.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        probs = {k: z[k] for k in z.files if k not in ("y", "ids")}
        print("\nusing cached test probabilities from script 05")
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("\nno cache; recomputing test probabilities")
        probs = P.all_probas(test["utterance"].astype(str).tolist(),
                             device=device)

    errs = {}
    for name in ("RF", "DistilBERT"):
        if name not in probs:
            print("  skipping {} (no probabilities available)".format(name))
            continue
        errs[name] = error_analysis(test, probs[name], name)

    C.save_json(
        {
            "top_k": TOP_K,
            "n_errors_per_class": N_ERRORS,
            "lr_coefficients": coefs,
            "error_analysis": errs,
            "notes": (
                "Features are Porter stems from the classical pipeline. "
                "'Highest-confidence' errors are those furthest on the wrong "
                "side of the 0.5 threshold."
            ),
        },
        C.RESULTS / "09_analysis.json",
        "09_analysis",
    )


if __name__ == "__main__":
    main()
