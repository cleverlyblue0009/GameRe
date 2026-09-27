"""02 - Preprocessing.

Two separate text pipelines, because the two model families want different
things:

CLASSICAL (for TF-IDF + LR/NB/RF), applied strictly in this order:
  1. lowercase
  2. strip URLs, [SEPA] turn separators and @player mentions
  3. leetspeak normalise   3->e 0->o 4->a 1->i @->a $->s
  4. strip punctuation     -> REPLACED BY A SPACE, apostrophes kept only
                              inside a word (contractions)
  5. remove NLTK English stopwords MINUS the gaming whitelist
     {gg, wp, noob, ez, gl, hf, glhf, op, nerf, buff}
  6. Porter stem

DISTILBERT: only steps 1 and 2. Its WordPiece tokeniser is designed to handle
punctuation and subwords itself, and stemming would fight the pretrained
vocabulary.

Every step is individually toggleable via common.PreprocessConfig, which is
what the script-06 ablation (leetspeak OFF / punctuation OFF) relies on.

NOTE ON PUNCTUATION (matters for the adversarial results in script 06):
punctuation is replaced by a space, not deleted. Replacing avoids silently
welding neighbouring words together ("you.suck" -> "you suck"). The cost is
that it does NOT defend against the character-insertion attack: "f.u.c.k"
becomes "f u c k" (four 1-char tokens) rather than "fuck". Deleting
punctuation instead would accidentally repair that attack. We keep the
realistic, standard choice and report the consequence honestly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

# the classical pipeline, one step at a time, for the stage-by-stage trace
STAGES = [
    ("0. raw", C.PreprocessConfig(False, False, False, False, False, False)),
    ("1. +lowercase", C.PreprocessConfig(True, False, False, False, False, False)),
    ("2. +strip url/mention", C.PreprocessConfig(True, True, False, False, False, False)),
    ("3. +leetspeak", C.PreprocessConfig(True, True, True, False, False, False)),
    ("4. +strip punctuation", C.PreprocessConfig(True, True, True, True, False, False)),
    ("5. +stopwords", C.PreprocessConfig(True, True, True, True, True, False)),
    ("6. +porter stem", C.PreprocessConfig(True, True, True, True, True, True)),
]

TRACE_SAMPLES = [
    "F.U.C.K you noob!!",
    "you SUCK n00b http://dota.com @bob",
    "don't be 'toxic' 3z game",
    "GG [SEPA] nice late game",
    "MY team is full of retards",
    "commend spec [SEPA] lel",
]


def stage_trace():
    """Show exactly what each step does - this is the audit trail for Sec. III."""
    C.banner("02 - PREPROCESSING: stage-by-stage trace (classical pipeline)")
    for text in TRACE_SAMPLES:
        print("\n  input: {!r}".format(text))
        for name, cfg in STAGES:
            print("    {:<24} {!r}".format(name, C.preprocess(text, cfg)))


def describe_stopwords():
    sw = C._stopwords()
    import nltk
    from nltk.corpus import stopwords

    full = set(stopwords.words("english"))
    print("\nNLTK english stopwords      : {}".format(len(full)))
    print("gaming whitelist re-admitted: {}".format(
        sorted(C.STOPWORD_WHITELIST & full)))
    print("  (whitelist terms not in NLTK list anyway: {})".format(
        sorted(C.STOPWORD_WHITELIST - full)))
    print("effective stopword count    : {}".format(len(sw)))
    return {
        "nltk_total": len(full),
        "whitelist": sorted(C.STOPWORD_WHITELIST),
        "whitelist_actually_in_nltk": sorted(C.STOPWORD_WHITELIST & full),
        "effective_count": len(sw),
    }


def process_split(name):
    df = pd.read_csv(C.PROC / (name + ".csv"))
    df["text_classical"] = df["utterance"].apply(lambda t: C.preprocess(t, C.CLASSICAL))
    df["text_bert"] = df["utterance"].apply(lambda t: C.preprocess(t, C.BERT))

    empty = int((df["text_classical"].str.strip() == "").sum())
    tok_counts = df["text_classical"].str.split().apply(len)
    vocab = set()
    for t in df["text_classical"]:
        vocab.update(t.split())

    out = C.PROC / (name + "_prep.csv")
    df.to_csv(out, index=False)
    stats = {
        "split": name,
        "n": len(df),
        "empty_after_classical": empty,
        "empty_pct": round(100.0 * empty / len(df), 3),
        "mean_tokens_classical": round(float(tok_counts.mean()), 2),
        "median_tokens_classical": int(tok_counts.median()),
        "max_tokens_classical": int(tok_counts.max()),
        "vocab_size_classical": len(vocab),
    }
    print(
        "  {:<6} n={:>6}  empty_after_prep={:>4} ({:.2f}%)  "
        "mean_tok={:>5.2f}  vocab={:>6}".format(
            name, len(df), empty, stats["empty_pct"],
            stats["mean_tokens_classical"], len(vocab),
        )
    )
    print("         -> {}".format(out.relative_to(C.ROOT)))
    return stats


def main():
    C.set_seed()
    # make sure the NLTK stopword corpus is present before we time anything
    C._stopwords()

    stage_trace()

    C.banner("02 - PREPROCESSING: stopword configuration")
    sw_stats = describe_stopwords()

    C.banner("02 - PREPROCESSING: applying to splits")
    stats = [process_split(s) for s in ("train", "val", "test")]

    C.banner("02 - PREPROCESSING: empty-output check")
    print("Utterances that preprocess to the empty string keep their row (the")
    print("vectoriser maps them to an all-zero vector). They are NOT dropped,")
    print("so every split keeps the exact size reported by script 01.")

    C.save_json(
        {
            "classical_order": [
                "lowercase",
                "strip_urls_mentions ([SEPA], @mentions, URLs)",
                "leetspeak (3->e 0->o 4->a 1->i @->a $->s)",
                "strip_punctuation (REPLACED BY SPACE; apostrophes kept "
                "inside words only)",
                "remove_stopwords (NLTK english minus gaming whitelist)",
                "porter_stem",
            ],
            "punctuation_policy": "replaced_by_space",
            "punctuation_policy_note": (
                "Punctuation is replaced by a space rather than deleted. This "
                "avoids welding words together, but means the character-"
                "insertion attack is NOT repaired by preprocessing "
                "('f.u.c.k' -> 'f u c k'). Deleting punctuation would "
                "accidentally defend against that attack; see the script 06 "
                "ablation."
            ),
            "bert_pipeline": ["lowercase", "strip_urls_mentions"],
            "leet_map": C.LEET_MAP,
            "stopwords": sw_stats,
            "splits": stats,
        },
        C.RESULTS / "02_preprocess.json",
        "02_preprocess",
    )


if __name__ == "__main__":
    main()
