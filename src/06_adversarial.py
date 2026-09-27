"""06 - Adversarial robustness under character-level obfuscation.

THREAT MODEL (this is the important part)
-----------------------------------------
Attacks are applied to the **RAW** utterance, and the perturbed raw string is
then pushed through each model's own normal preprocessing. That is what a real
evader does: they type "f.u.c.k" into the chat box, and the defender's
pipeline -- leetspeak normalisation, punctuation stripping and all -- runs on
that. Perturbing already-preprocessed text would let the classical pipeline
"clean" an attack it never actually saw, which would overstate its robustness.

Only TOXIC test utterances are attacked (perturbing non-toxic text would not
model an evader). Non-toxic rows pass through untouched, so macro-F1 is always
computed over the full 8,973-row test set and stays comparable to script 05.

ATTACKS (all seeded, seed 42)
  leetspeak      per character of a profane-candidate token, with p=0.5:
                 a->4 e->3 i->1 o->0 s->$ t->7
  char_insert    a random punctuation character between internal characters
  whitespace     a space between internal characters
  swap           one adjacent transposition per token

"Profane-candidate" tokens: a token is attacked only if it is flagged by the
CONDA token-level annotation for that utterance (slotClasses tag T = toxicity,
or C = character/hero abuse) -- falling back, when annotation is missing, to
any alphabetic token of length >= 3 that is not a stopword. Using the
dataset's own toxicity annotation keeps the attack targeted at the words that
actually carry the label, instead of mangling the whole sentence.

ABLATION: the classical models are re-evaluated with leetspeak normalisation
OFF and punctuation stripping OFF to show how much each defence contributes.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import predict as P

LEET_ATTACK = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "$", "t": "7"}
LEET_P = 0.5
# Deliberately excludes the apostrophe: it is the one punctuation character the
# classical pipeline treats specially (kept inside contractions), so including
# it here would conflate "attack inserted a character" with "pipeline handles
# this character differently" and muddy the punctuation-OFF ablation.
INSERT_CHARS = ".-_*`~^"
N_AUDIT = 20
MODELS = ("LR", "NB", "RF", "DistilBERT")

# The preprocessing ablation is a property of the pipeline, not of the label
# definition, so it only needs running once. Set TOX_SKIP_ABLATION=1 to skip it
# (used for the E+I sensitivity run, where it would duplicate the primary run).
SKIP_ABLATION = os.environ.get("TOX_SKIP_ABLATION", "").strip() in ("1", "true", "yes")


# --------------------------------------------------------------------------
# choosing which tokens to attack
# --------------------------------------------------------------------------
def candidate_flags(utterance, slot_classes):
    """Return a bool per whitespace token: is this a profane candidate?

    Primary signal: CONDA's own token-level slot tags (T = toxicity,
    C = character/hero abuse). Fallback: alphabetic, length >= 3, not a
    stopword.
    """
    toks = str(utterance).split()
    tags = str(slot_classes).split() if isinstance(slot_classes, str) else []

    if len(tags) == len(toks) and tags:
        flags = [t in ("T", "C") for t in tags]
        if any(flags):
            return flags

    sw = C._stopwords()
    return [
        (tok.isalpha() and len(tok) >= 3 and tok.lower() not in sw) for tok in toks
    ]


# --------------------------------------------------------------------------
# attacks (operate on raw text)
# --------------------------------------------------------------------------
def atk_leetspeak(utterance, slots, rng):
    toks = str(utterance).split()
    flags = candidate_flags(utterance, slots)
    out = []
    for tok, flag in zip(toks, flags):
        if not flag:
            out.append(tok)
            continue
        out.append("".join(
            # index with the lowered char: membership was tested on ch.lower(),
            # and the leet substitutes (4/3/1/0/$/7) are caseless anyway
            LEET_ATTACK[ch.lower()]
            if (ch.lower() in LEET_ATTACK and rng.random() < LEET_P)
            else ch
            for ch in tok
        ))
    return " ".join(out)


def atk_char_insert(utterance, slots, rng):
    """Insert a random punctuation char between internal characters."""
    toks = str(utterance).split()
    flags = candidate_flags(utterance, slots)
    out = []
    for tok, flag in zip(toks, flags):
        if not flag or len(tok) < 3:
            out.append(tok)
            continue
        # between internal characters only: keep first and last char adjacent
        # to their neighbours untouched at the edges
        mid = [tok[0]]
        for ch in tok[1:-1]:
            mid.append(rng.choice(INSERT_CHARS))
            mid.append(ch)
        mid.append(rng.choice(INSERT_CHARS))
        mid.append(tok[-1])
        out.append("".join(mid))
    return " ".join(out)


def atk_whitespace(utterance, slots, rng):
    toks = str(utterance).split()
    flags = candidate_flags(utterance, slots)
    out = []
    for tok, flag in zip(toks, flags):
        if not flag or len(tok) < 3:
            out.append(tok)
            continue
        out.append(" ".join(tok))
    return " ".join(out)


def atk_swap(utterance, slots, rng):
    """One adjacent transposition per candidate token."""
    toks = str(utterance).split()
    flags = candidate_flags(utterance, slots)
    out = []
    for tok, flag in zip(toks, flags):
        if not flag or len(tok) < 4:
            out.append(tok)
            continue
        i = rng.randrange(1, len(tok) - 2) if len(tok) > 4 else 1
        lst = list(tok)
        lst[i], lst[i + 1] = lst[i + 1], lst[i]
        out.append("".join(lst))
    return " ".join(out)


ATTACKS = {
    "leetspeak": atk_leetspeak,
    "char_insert": atk_char_insert,
    "whitespace": atk_whitespace,
    "swap": atk_swap,
}


def build_attacked(test, attack_name):
    """Return the full test set's raw text with TOXIC rows perturbed."""
    rng = random.Random(C.SEED)
    fn = ATTACKS[attack_name]
    texts, changed = [], 0
    for _, row in test.iterrows():
        raw = str(row["utterance"])
        if row[C.label_col()] == 1:
            new = fn(raw, row.get("slotClasses"), rng)
            texts.append(new)
            changed += int(new != raw)
        else:
            texts.append(raw)
    return texts, changed


# --------------------------------------------------------------------------
def macro_f1(y, prob):
    from sklearn.metrics import f1_score

    return float(f1_score(y, (np.asarray(prob) >= 0.5).astype(int), average="macro"))


def audit(test, attack_name, vec, n=N_AUDIT):
    """Show raw -> perturbed -> what the TF-IDF vectoriser actually receives."""
    rng = random.Random(C.SEED)
    fn = ATTACKS[attack_name]
    tox = test[test[C.label_col()] == 1]
    rows = []
    print("\n  --- audit: {} (first {} toxic test items) ---".format(attack_name, n))
    shown = 0
    for _, row in tox.iterrows():
        raw = str(row["utterance"])
        pert = fn(raw, row.get("slotClasses"), rng)
        if shown >= n:
            continue
        prep = C.preprocess(pert, C.CLASSICAL)
        prep_clean = C.preprocess(raw, C.CLASSICAL)
        # which of the surviving terms are actually in the vocabulary?
        vocab = vec.vocabulary_
        kept = [t for t in prep.split() if t in vocab]
        kept_clean = [t for t in prep_clean.split() if t in vocab]
        print("    [{}] raw       : {!r}".format(shown + 1, raw[:90]))
        print("        perturbed : {!r}".format(pert[:90]))
        print("        -> prep   : {!r}".format(prep[:90]))
        print("        in-vocab  : {}  (clean was {})".format(kept, kept_clean))
        rows.append({
            "id": int(row["Id"]), "raw": raw, "perturbed": pert,
            "vectorizer_input": prep,
            "clean_vectorizer_input": prep_clean,
            "in_vocab_terms": kept, "in_vocab_terms_clean": kept_clean,
            "n_in_vocab": len(kept), "n_in_vocab_clean": len(kept_clean),
        })
        shown += 1
    return rows


def run_ablation(y, raw, test, vec, cmodels, abl):
    C.banner("06 - ADVERSARIAL: ablation (classical defences OFF)")
    print("  Re-running the classical models with preprocessing steps disabled.")
    print("  NOTE: the models were TRAINED with the full pipeline; here only the")
    print("  inference-time preprocessing changes. This isolates the")
    print("  contribution of each normalisation step as a defence, and the")
    print("  train/inference mismatch it introduces is itself part of the")
    print("  finding (reported as the clean-text column).\n")

    ablations = {
        "full_pipeline": C.CLASSICAL,
        "leetspeak_OFF": C.PreprocessConfig(
            True, True, False, True, True, True),
        "punctuation_OFF": C.PreprocessConfig(
            True, True, True, False, True, True),
        "both_OFF": C.PreprocessConfig(
            True, True, False, False, True, True),
    }
    hdr = "{:<18}{:<8}{:>10}".format("config", "model", "clean") + "".join(
        "{:>14}".format(a) for a in ATTACKS)
    print(hdr)
    print("-" * len(hdr))
    for cfg_name, cfg in ablations.items():
        abl[cfg_name] = {}
        for m in ("LR", "NB", "RF"):
            cf1 = macro_f1(y, P.classical_proba(raw, vec, cmodels[m], cfg))
            row = {"clean": cf1}
            cells = []
            for atk in ATTACKS:
                texts, _ = build_attacked(test, atk)
                f1 = macro_f1(y, P.classical_proba(texts, vec, cmodels[m], cfg))
                row[atk] = {"macro_f1": f1, "abs_drop": cf1 - f1}
                cells.append("{:.4f} ({:+.3f})".format(f1, f1 - cf1))
            abl[cfg_name][m] = row
            print("{:<18}{:<8}{:>10.4f}".format(cfg_name, m, cf1)
                  + "".join("{:>14}".format(c) for c in cells))
    return abl


def main():
    C.set_seed()
    C.announce_label_set()
    C._stopwords()
    import torch

    test = pd.read_csv(C.PROC / "test_prep.csv")
    y = test[C.label_col()].values
    raw = test["utterance"].astype(str).tolist()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vec, cmodels = P.load_classical()
    bert, tok, _ = P.load_bert(device=device)

    def all_probs(texts):
        out = {n: P.classical_proba(texts, vec, m) for n, m in cmodels.items()}
        out["DistilBERT"] = P.bert_proba(texts, bert, tok, device=device)
        return out

    C.banner("06 - ADVERSARIAL: clean baseline")
    clean = all_probs(raw)
    clean_f1 = {m: macro_f1(y, clean[m]) for m in MODELS}
    for m in MODELS:
        print("  {:<12} clean macro-F1 = {:.4f}".format(m, clean_f1[m]))

    C.banner("06 - ADVERSARIAL: attacks (applied to RAW toxic utterances)")
    results, audits = {}, {}
    for atk in ATTACKS:
        texts, changed = build_attacked(test, atk)
        pr = all_probs(texts)
        res = {}
        print("\n  attack = {}   (perturbed {}/{} toxic utterances)".format(
            atk, changed, int(y.sum())))
        for m in MODELS:
            f1 = macro_f1(y, pr[m])
            drop = clean_f1[m] - f1
            res[m] = {"macro_f1": f1, "clean_macro_f1": clean_f1[m],
                      "abs_drop": drop,
                      "rel_drop_pct": 100.0 * drop / clean_f1[m]}
            print("    {:<12} macro-F1 {:.4f}   drop {:+.4f}  "
                  "({:+.1f}%)".format(m, f1, -drop, -100.0 * drop / clean_f1[m]))
        results[atk] = {"n_perturbed": changed, "per_model": res}
        audits[atk] = audit(test, atk, vec)

    # ---------------------------------------------------------------- ablation
    abl = {}
    if SKIP_ABLATION:
        C.banner("06 - ADVERSARIAL: ablation SKIPPED")
        print("  TOX_SKIP_ABLATION set. The preprocessing ablation depends on")
        print("  the pipeline, not the label definition, so it is reported")
        print("  once from the primary run (results/06_adversarial.json).")
    else:
        run_ablation(y, raw, test, vec, cmodels, abl)
    C.banner("06 - ADVERSARIAL: summary")
    print("{:<14}{:>10}".format("model", "clean") + "".join(
        "{:>13}".format(a) for a in ATTACKS))
    print("-" * 66)
    for m in MODELS:
        print("{:<14}{:>10.4f}".format(m, clean_f1[m]) + "".join(
            "{:>13.4f}".format(results[a]["per_model"][m]["macro_f1"])
            for a in ATTACKS))
    print("\nabsolute drop from clean:")
    for m in MODELS:
        print("{:<14}{:>10}".format(m, "-") + "".join(
            "{:>13.4f}".format(-results[a]["per_model"][m]["abs_drop"])
            for a in ATTACKS))

    C.save_json(
        {
            "threat_model": (
                "Attacks applied to RAW toxic utterances, then passed through "
                "each model's normal preprocessing. Non-toxic rows untouched; "
                "macro-F1 computed over the full test set."
            ),
            "seed": C.SEED,
            "label_set": C.LABEL_SET,
            "label_rule": C.label_rule(),
            "attacks": {
                "leetspeak": "per char of candidate tokens, p=0.5: "
                             "a->4 e->3 i->1 o->0 s->$ t->7",
                "char_insert": "random punctuation from {!r} between internal "
                               "characters".format(INSERT_CHARS),
                "whitespace": "space between internal characters",
                "swap": "one adjacent transposition per candidate token",
            },
            "candidate_token_rule": (
                "CONDA token-level slot tag in {T, C} when the annotation "
                "aligns with the tokens; otherwise alphabetic, len>=3, "
                "non-stopword."
            ),
            "clean_macro_f1": clean_f1,
            "results": results,
            "ablation": abl,
            "ablation_skipped": SKIP_ABLATION,
            "ablation_note": (
                "SKIPPED for this run (TOX_SKIP_ABLATION set): the ablation "
                "depends on the preprocessing pipeline, not the label "
                "definition, so it is reported once from the primary run."
                if SKIP_ABLATION else
                "Models trained with the full pipeline; only inference-time "
                "preprocessing is ablated."
            ),
            "audit_samples": audits,
            "n_audit_per_attack": N_AUDIT,
        },
        C.RESULTS / ("06_adversarial" + C.suffix() + ".json"),
        "06_adversarial" + C.suffix(),
    )


if __name__ == "__main__":
    main()
