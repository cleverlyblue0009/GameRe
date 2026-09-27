"""03 - Classical baselines: TF-IDF + Logistic Regression / MultinomialNB / RF.

Vectoriser (fitted on TRAIN ONLY, never on val or test):
  TfidfVectorizer(ngram_range=(1,2), max_features=20000,
                  sublinear_tf=True, min_df=2)

Hyperparameter selection - none of it touches the test set:
  LR  : C            -> 5-fold stratified CV on TRAIN (macro-F1)
  NB  : alpha        -> selected on the VAL split (macro-F1)
  RF  : max_depth    -> 5-fold stratified CV on TRAIN (macro-F1)

A note on the CV protocol: the vectoriser is fitted once on the full training
split and the CV folds then run over that fixed feature matrix. This is the
literal reading of "fit on train only" and leaks nothing from val or test. It
does allow a fold's features to have been counted using the whole training
split, which is standard practice and is recorded here for transparency.

Fitted objects land in checkpoints/ (gitignored) for scripts 05-09.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

# n_jobs used for *training* / CV. Script 07 separately measures RF inference
# latency at n_jobs=1 and n_jobs=-1 on the same fitted model.
TRAIN_N_JOBS = -1

TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    max_features=20000,
    sublinear_tf=True,
    min_df=2,
)

LR_C_GRID = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
NB_ALPHA_GRID = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
RF_DEPTH_GRID = [None, 10, 20, 30, 50]


def load(split):
    return pd.read_csv(C.PROC / (split + "_prep.csv")).fillna({"text_classical": ""})


def main():
    C.set_seed()
    C.announce_label_set()
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.naive_bayes import MultinomialNB

    train, val = load("train"), load("val")
    Xtr_txt, ytr = train["text_classical"].values, train[C.label_col()].values
    Xva_txt, yva = val["text_classical"].values, val[C.label_col()].values

    C.banner("03 - CLASSICAL: TF-IDF vectoriser (fit on TRAIN only)")
    t0 = time.perf_counter()
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    Xtr = vec.fit_transform(Xtr_txt)
    Xva = vec.transform(Xva_txt)
    fit_s = time.perf_counter() - t0
    print("  params           : {}".format(TFIDF_PARAMS))
    print("  train matrix     : {} x {}  (nnz={}, density={:.5f})".format(
        Xtr.shape[0], Xtr.shape[1], Xtr.nnz, Xtr.nnz / (Xtr.shape[0] * Xtr.shape[1])))
    print("  vocabulary size  : {}".format(len(vec.vocabulary_)))
    n_uni = sum(1 for f in vec.get_feature_names_out() if " " not in f)
    print("  unigrams/bigrams : {} / {}".format(n_uni, len(vec.vocabulary_) - n_uni))
    print("  fit time         : {:.2f}s".format(fit_s))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    results = {}

    # ---------------------------------------------------------------- LR
    C.banner("03 - CLASSICAL: Logistic Regression (C by 5-fold CV on TRAIN)")
    t0 = time.perf_counter()
    gs = GridSearchCV(
        LogisticRegression(
            penalty="l2", solver="lbfgs", class_weight="balanced",
            max_iter=2000, random_state=C.SEED,
        ),
        {"C": LR_C_GRID},
        scoring="f1_macro", cv=cv, n_jobs=TRAIN_N_JOBS, return_train_score=False,
    )
    gs.fit(Xtr, ytr)
    lr_s = time.perf_counter() - t0
    for c, m, sd in zip(
        gs.cv_results_["param_C"],
        gs.cv_results_["mean_test_score"],
        gs.cv_results_["std_test_score"],
    ):
        star = "  <== best" if c == gs.best_params_["C"] else ""
        print("    C={:<6} cv macro-F1 = {:.4f} +/- {:.4f}{}".format(c, m, sd, star))
    lr = gs.best_estimator_
    print("  best C           : {}".format(gs.best_params_["C"]))
    print("  val macro-F1     : {:.4f}".format(
        f1_score(yva, lr.predict(Xva), average="macro")))
    print("  train time       : {:.2f}s  (n_jobs={})".format(lr_s, TRAIN_N_JOBS))
    results["LR"] = {
        "best_params": {"C": float(gs.best_params_["C"])},
        "selection": "5-fold stratified CV on train, macro-F1",
        "grid": LR_C_GRID,
        "cv_scores": {str(c): float(m) for c, m in zip(
            gs.cv_results_["param_C"], gs.cv_results_["mean_test_score"])},
        "cv_best_score": float(gs.best_score_),
        "val_macro_f1": float(f1_score(yva, lr.predict(Xva), average="macro")),
        "train_seconds": round(lr_s, 2),
        "n_jobs": TRAIN_N_JOBS,
        "fixed": {"penalty": "l2", "solver": "lbfgs",
                  "class_weight": "balanced", "max_iter": 2000},
    }

    # ---------------------------------------------------------------- NB
    C.banner("03 - CLASSICAL: MultinomialNB (alpha selected on VAL)")
    t0 = time.perf_counter()
    nb_scores = {}
    for a in NB_ALPHA_GRID:
        m = MultinomialNB(alpha=a).fit(Xtr, ytr)
        s = f1_score(yva, m.predict(Xva), average="macro")
        nb_scores[a] = s
    best_alpha = max(nb_scores, key=nb_scores.get)
    for a, s in nb_scores.items():
        print("    alpha={:<6} val macro-F1 = {:.4f}{}".format(
            a, s, "  <== best" if a == best_alpha else ""))
    nb = MultinomialNB(alpha=best_alpha).fit(Xtr, ytr)
    nb_s = time.perf_counter() - t0
    print("  best alpha       : {}".format(best_alpha))
    print("  val macro-F1     : {:.4f}".format(nb_scores[best_alpha]))
    print("  total time       : {:.2f}s".format(nb_s))
    results["NB"] = {
        "best_params": {"alpha": float(best_alpha)},
        "selection": "val split, macro-F1",
        "grid": NB_ALPHA_GRID,
        "val_scores": {str(k): float(v) for k, v in nb_scores.items()},
        "val_macro_f1": float(nb_scores[best_alpha]),
        "train_seconds": round(nb_s, 2),
    }

    # ---------------------------------------------------------------- RF
    C.banner("03 - CLASSICAL: Random Forest (max_depth by 5-fold CV on TRAIN)")
    t0 = time.perf_counter()
    gs_rf = GridSearchCV(
        RandomForestClassifier(
            n_estimators=200, max_features="sqrt", class_weight="balanced",
            random_state=C.SEED, n_jobs=TRAIN_N_JOBS,
        ),
        {"max_depth": RF_DEPTH_GRID},
        scoring="f1_macro", cv=cv, n_jobs=1,  # inner RF already parallel
    )
    gs_rf.fit(Xtr, ytr)
    rf_s = time.perf_counter() - t0
    for d, m, sd in zip(
        gs_rf.cv_results_["param_max_depth"],
        gs_rf.cv_results_["mean_test_score"],
        gs_rf.cv_results_["std_test_score"],
    ):
        star = "  <== best" if d == gs_rf.best_params_["max_depth"] else ""
        print("    max_depth={:<6} cv macro-F1 = {:.4f} +/- {:.4f}{}".format(
            str(d), m, sd, star))
    rf = gs_rf.best_estimator_
    print("  best max_depth   : {}".format(gs_rf.best_params_["max_depth"]))
    print("  val macro-F1     : {:.4f}".format(
        f1_score(yva, rf.predict(Xva), average="macro")))
    print("  train time       : {:.2f}s".format(rf_s))
    print("  n_jobs (TRAIN)   : {}".format(TRAIN_N_JOBS))
    print("  n_jobs stored on fitted model: {}".format(rf.n_jobs))
    results["RF"] = {
        "best_params": {"max_depth": gs_rf.best_params_["max_depth"]},
        "selection": "5-fold stratified CV on train, macro-F1",
        "grid": [str(d) for d in RF_DEPTH_GRID],
        "cv_scores": {str(d): float(m) for d, m in zip(
            gs_rf.cv_results_["param_max_depth"],
            gs_rf.cv_results_["mean_test_score"])},
        "cv_best_score": float(gs_rf.best_score_),
        "val_macro_f1": float(f1_score(yva, rf.predict(Xva), average="macro")),
        "train_seconds": round(rf_s, 2),
        "n_jobs_train": TRAIN_N_JOBS,
        "fixed": {"n_estimators": 200, "max_features": "sqrt",
                  "class_weight": "balanced"},
    }

    # ---------------------------------------------------------------- save
    C.banner("03 - CLASSICAL: saving fitted pipelines")
    sfx = C.suffix()
    joblib.dump(vec, C.CKPT / ("tfidf_vectorizer" + sfx + ".joblib"))
    joblib.dump(lr, C.CKPT / ("model_lr" + sfx + ".joblib"))
    joblib.dump(nb, C.CKPT / ("model_nb" + sfx + ".joblib"))
    joblib.dump(rf, C.CKPT / ("model_rf" + sfx + ".joblib"))
    for f in ("tfidf_vectorizer", "model_lr", "model_nb", "model_rf"):
        p = C.CKPT / (f + sfx + ".joblib")
        print("  [saved] {}  ({:.1f} MB)".format(
            p.name, p.stat().st_size / 1024 ** 2))

    C.save_json(
        {
            "tfidf": {
                **{k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in TFIDF_PARAMS.items()},
                "fitted_on": "train split only",
                "vocab_size": len(vec.vocabulary_),
                "n_unigrams": n_uni,
                "n_bigrams": len(vec.vocabulary_) - n_uni,
                "fit_seconds": round(fit_s, 2),
            },
            "train_n_jobs": TRAIN_N_JOBS,
            "cv": {"n_splits": 5, "shuffle": True, "random_state": C.SEED,
                   "scoring": "f1_macro"},
            "models": results,
            "label_set": C.LABEL_SET,
            "label_rule": C.label_rule(),
            "protocol_note": (
                "No hyperparameter was selected using the test split. LR and RF "
                "use 5-fold stratified CV on train; NB alpha uses the val split."
            ),
        },
        C.RESULTS / ("03_classical" + C.suffix() + ".json"),
        "03_classical" + C.suffix(),
    )


if __name__ == "__main__":
    main()
