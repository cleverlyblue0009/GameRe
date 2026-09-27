"""Shared model loading and prediction.

Every entry point here takes **raw, unpreprocessed utterances** and applies
that model's own preprocessing internally. That is deliberate and it is the
whole reason scripts 06/07/08 are trustworthy:

  * script 06 perturbs RAW text and must then push it through the real
    pipeline, exactly as a deployed system would (attacking already-
    preprocessed text would be a fantasy threat model that flatters the
    classical models);
  * script 07 must be able to time the true end-to-end path, preprocessing
    included, not just the matrix multiply.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

CLASSICAL_NAMES = ("LR", "NB", "RF")
BERT_CKPT = C.CKPT / ("distilbert_best" + C.suffix() + ".pt")


# --------------------------------------------------------------------------
# classical
# --------------------------------------------------------------------------
def load_classical():
    """Return (vectorizer, {name: fitted_model})."""
    import joblib

    sfx = C.suffix()
    vec = joblib.load(C.CKPT / ("tfidf_vectorizer" + sfx + ".joblib"))
    models = {
        "LR": joblib.load(C.CKPT / ("model_lr" + sfx + ".joblib")),
        "NB": joblib.load(C.CKPT / ("model_nb" + sfx + ".joblib")),
        "RF": joblib.load(C.CKPT / ("model_rf" + sfx + ".joblib")),
    }
    return vec, models


def classical_preprocess(raw_texts, cfg=None):
    cfg = cfg or C.CLASSICAL
    return [C.preprocess(t, cfg) for t in raw_texts]


def classical_proba(raw_texts, vec, model, cfg=None):
    """RAW text -> preprocessing -> TF-IDF -> P(toxic). Full end-to-end path."""
    texts = classical_preprocess(raw_texts, cfg)
    X = vec.transform(texts)
    return model.predict_proba(X)[:, 1]


def classical_proba_from_matrix(X, model):
    """Model-only path, for the model-only latency figures."""
    return model.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------
# distilbert
# --------------------------------------------------------------------------
def load_bert(device="cuda", ckpt_path=None):
    """Rebuild the architecture from script 04 and load the best checkpoint."""
    import torch
    import torch.nn as nn
    from transformers import DistilBertModel, DistilBertTokenizerFast

    ckpt_path = Path(ckpt_path or BERT_CKPT)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            "DistilBERT checkpoint missing: {}. Run src/04_distilbert.py "
            "first.".format(ckpt_path)
        )

    class DistilBertCLSLinear(nn.Module):
        def __init__(self, name="distilbert-base-uncased", n_classes=2, dropout=0.1):
            super().__init__()
            self.bert = DistilBertModel.from_pretrained(name)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.bert.config.dim, n_classes)

        def forward(self, input_ids, attention_mask):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            return self.classifier(self.dropout(out.last_hidden_state[:, 0]))

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = DistilBertCLSLinear()
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    return model, tok, blob


def bert_proba(raw_texts, model, tok, device="cuda", batch_size=64,
               max_len=128, autocast_fp16=False):
    """RAW text -> lowercase/URL strip -> WordPiece -> P(toxic)."""
    import torch

    texts = [C.preprocess(t, C.BERT) for t in raw_texts]
    out = np.empty(len(texts), dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = tok(chunk, truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            if autocast_fp16 and str(device).startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(ids, mask)
            else:
                logits = model(ids, mask)
            p = torch.softmax(logits.float(), dim=-1)[:, 1]
            out[i : i + len(chunk)] = p.cpu().numpy()
    return out


# --------------------------------------------------------------------------
# convenience
# --------------------------------------------------------------------------
def all_probas(raw_texts, device="cuda", include_bert=True):
    """P(toxic) from every model, keyed by name. Raw text in."""
    vec, models = load_classical()
    probs = {
        name: classical_proba(raw_texts, vec, mdl)
        for name, mdl in models.items()
    }
    if include_bert:
        model, tok, _ = load_bert(device=device)
        probs["DistilBERT"] = bert_proba(raw_texts, model, tok, device=device)
    return probs


def to_pred(p, threshold=0.5):
    return (np.asarray(p) >= threshold).astype(int)
