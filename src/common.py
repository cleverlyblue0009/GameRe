"""Shared utilities for the CONDA toxicity-detection study.

Everything that more than one numbered script needs lives here: paths,
seeding, the label mapping, the classical preprocessing pipeline and
small JSON/IO helpers.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
CKPT = ROOT / "checkpoints"
FIGURES = ROOT / "figures"

for _d in (PROC, RESULTS, LOGS, CKPT, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# --------------------------------------------------------------------------
# label mapping
# --------------------------------------------------------------------------
# CONDA utterance-level intent classes (Weld et al. 2021):
#   O = Other / neutral
#   E = Explicit toxicity
#   I = Implicit toxicity
#   A = Action  (calls to "report"/"commend" a player)
#
# PRIMARY mapping reproduces the original study's test counts exactly
# (2,345 toxic / 6,628 non-toxic):  toxic = intentClass != 'O'
#
# SENSITIVITY mapping follows the dataset authors' own definition of the
# toxicity classes (the CONDA README ranks only E and I as toxicity):
#   toxic = intentClass in {E, I}
# See results/SUMMARY.md for why this distinction matters.
TOXIC_PRIMARY = ("E", "A", "I")
TOXIC_EI = ("E", "I")


def label_primary(intent: "str") -> int:
    return int(intent in TOXIC_PRIMARY)


def label_ei(intent: "str") -> int:
    return int(intent in TOXIC_EI)


# --------------------------------------------------------------------------
# which label definition is this run using?
# --------------------------------------------------------------------------
# Selected by the TOX_LABEL_SET environment variable so that a whole pipeline
# can be run twice without editing code:
#
#   python src/03_classical.py                      -> primary  (E + A + I)
#   TOX_LABEL_SET=ei python src/03_classical.py     -> sensitivity (E + I)
#
# Every artefact the sensitivity run writes is suffixed "_ei", so the two runs
# never overwrite each other.
LABEL_SETS = {
    "primary": {"column": "label", "suffix": "",
                "rule": "toxic = intentClass != 'O'  (E + A + I)"},
    "ei": {"column": "label_EI", "suffix": "_ei",
           "rule": "toxic = intentClass in {E, I}"},
}

LABEL_SET = os.environ.get("TOX_LABEL_SET", "primary").strip().lower()
if LABEL_SET not in LABEL_SETS:
    raise SystemExit(
        "TOX_LABEL_SET must be one of {}, got {!r}".format(
            sorted(LABEL_SETS), LABEL_SET)
    )


def label_col() -> str:
    """Name of the label column this run should train/evaluate against."""
    return LABEL_SETS[LABEL_SET]["column"]


def suffix() -> str:
    """Filename suffix keeping the two label runs' artefacts apart."""
    return LABEL_SETS[LABEL_SET]["suffix"]


def label_rule() -> str:
    return LABEL_SETS[LABEL_SET]["rule"]


def announce_label_set() -> None:
    print("label set : {}   ({})".format(LABEL_SET, label_rule()))
    print("label col : {}   artefact suffix: {!r}".format(
        label_col(), suffix()))


# --------------------------------------------------------------------------
# seeding
# --------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy and (if present) torch, deterministically."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
# CONDA joins several chat lines in one turn with a literal [SEPA] marker,
# and annotates addressed players with @handles.
SEPA_RE = re.compile(r"\[SEPA\]", re.I)
MENTION_RE = re.compile(r"@\w+")

LEET_MAP = {"3": "e", "0": "o", "4": "a", "1": "i", "@": "a", "$": "s"}

# Gaming terms that NLTK's stopword list would otherwise destroy, plus
# domain terms that carry the signal we are trying to classify.
STOPWORD_WHITELIST = {"gg", "wp", "noob", "ez", "gl", "hf", "glhf", "op", "nerf", "buff"}

# Punctuation handling: every punctuation character except the apostrophe is
# REPLACED BY A SPACE (not deleted). Replacing avoids silently welding two
# words together ("you.suck" -> "you suck", not "yousuck"). Apostrophes are
# kept only when they sit *inside* a word (contractions: "don't"), and are
# stripped when they hang off an edge ("'gg'" -> "gg").
PUNCT_KEEP_APOS = re.compile(r"[^\w\s']")
EDGE_APOS = re.compile(r"(?<!\w)'|'(?!\w)")


@dataclass(frozen=True)
class PreprocessConfig:
    """Toggle for each classical preprocessing step, applied in this order."""

    lowercase: bool = True
    strip_urls_mentions: bool = True
    leetspeak: bool = True
    strip_punctuation: bool = True
    remove_stopwords: bool = True
    porter_stem: bool = True

    def tag(self) -> str:
        on = [k for k, v in asdict(self).items() if v]
        return "+".join(on) if on else "raw"


# full classical pipeline
CLASSICAL = PreprocessConfig()
# DistilBERT sees only lowercase + URL/mention stripping
BERT = PreprocessConfig(
    lowercase=True,
    strip_urls_mentions=True,
    leetspeak=False,
    strip_punctuation=False,
    remove_stopwords=False,
    porter_stem=False,
)

_STEMMER = None
_STOPWORDS = None


def _stemmer():
    global _STEMMER
    if _STEMMER is None:
        from nltk.stem import PorterStemmer

        _STEMMER = PorterStemmer()
    return _STEMMER


def _stopwords() -> set:
    """NLTK English stopwords minus the gaming whitelist."""
    global _STOPWORDS
    if _STOPWORDS is None:
        import nltk
        from nltk.corpus import stopwords

        try:
            words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = set(stopwords.words("english"))
        _STOPWORDS = words - STOPWORD_WHITELIST
    return _STOPWORDS


def preprocess(text: str, cfg: PreprocessConfig = CLASSICAL) -> str:
    """Apply the configured preprocessing steps, in the documented order."""
    if text is None:
        return ""
    s = str(text)

    if cfg.lowercase:
        s = s.lower()

    if cfg.strip_urls_mentions:
        s = URL_RE.sub(" ", s)
        s = SEPA_RE.sub(" ", s)
        s = MENTION_RE.sub(" ", s)

    if cfg.leetspeak:
        s = s.translate(str.maketrans(LEET_MAP))

    if cfg.strip_punctuation:
        s = PUNCT_KEEP_APOS.sub(" ", s)   # punctuation -> space
        s = EDGE_APOS.sub("", s)          # drop non-contraction apostrophes

    toks = s.split()

    if cfg.remove_stopwords:
        sw = _stopwords()
        toks = [t for t in toks if t not in sw]

    if cfg.porter_stem:
        st = _stemmer()
        toks = [st.stem(t) for t in toks]

    return " ".join(toks)


# --------------------------------------------------------------------------
# io helpers
# --------------------------------------------------------------------------
def save_json(obj, path: Path, label: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default, ensure_ascii=False)
    print(f"[saved] {label or path.name} -> {path.relative_to(ROOT)}")


def load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def banner(msg: str) -> None:
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78)
