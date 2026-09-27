"""04 - DistilBERT fine-tuning (GPU required).

Configuration, exactly as specified:
  backbone        distilbert-base-uncased
  head            a single Linear(hidden_size -> 2) on the [CLS] token
  max_len         128
  batch size      32
  optimiser       AdamW, lr 2e-5
  schedule        linear decay with 10% linear warmup
  epochs          3, early stopping on VAL macro-F1
  seed            42

On the head: HuggingFace's DistilBertForSequenceClassification actually stacks
pre_classifier(768->768) + ReLU + dropout + classifier(768->2). The spec asks
for a *linear* head on [CLS], so we implement that literally (one Linear
layer) rather than using the HF default. This is recorded here because it is a
deliberate, reportable deviation from the stock HF model.

Selection uses VAL only. The test split is never touched by this script.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 3
WARMUP_FRAC = 0.10
CKPT_PATH = C.CKPT / "distilbert_best.pt"


def build_model(device):
    import torch.nn as nn
    from transformers import DistilBertModel

    class DistilBertCLSLinear(nn.Module):
        """DistilBERT backbone + a single linear layer on the [CLS] token."""

        def __init__(self, name=MODEL_NAME, n_classes=2, dropout=0.1):
            super().__init__()
            self.bert = DistilBertModel.from_pretrained(name)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.bert.config.dim, n_classes)

        def forward(self, input_ids, attention_mask):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0]  # [CLS]
            return self.classifier(self.dropout(cls))

    return DistilBertCLSLinear().to(device)


def make_loader(df, tok, shuffle, batch_size=BATCH_SIZE, generator=None):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    enc = tok(
        df["text_bert"].fillna("").astype(str).tolist(),
        truncation=True, max_length=MAX_LEN, padding="max_length",
        return_tensors="pt",
    )
    ds = TensorDataset(
        enc["input_ids"], enc["attention_mask"],
        torch.tensor(df["label"].values, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0)


def evaluate(model, loader, device):
    import torch
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    preds, labels, losses = [], [], []
    import torch.nn as nn

    lossf = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for ids, mask, y in loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            logits = model(ids, mask)
            losses.append(lossf(logits, y).item())
            preds.append(logits.argmax(-1).cpu().numpy())
            labels.append(y.cpu().numpy())
    p = np.concatenate(preds)
    t = np.concatenate(labels)
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(t, p)),
        "macro_f1": float(f1_score(t, p, average="macro")),
    }


def main():
    import torch
    import torch.nn as nn
    from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup

    C.banner("04 - DISTILBERT: environment")
    if not torch.cuda.is_available():
        print("FATAL: CUDA unavailable. Refusing to train on CPU, as instructed.")
        print("Run src/00_env.py for the diagnosis.")
        sys.exit(1)
    device = torch.device("cuda")
    print("  device : {}".format(torch.cuda.get_device_name(0)))
    print("  torch  : {}  (cuda {})".format(torch.__version__, torch.version.cuda))

    C.set_seed()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train = pd.read_csv(C.PROC / "train_prep.csv")
    val = pd.read_csv(C.PROC / "val_prep.csv")
    print("  train  : {} rows   val: {} rows".format(len(train), len(val)))

    tok = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    g = torch.Generator()
    g.manual_seed(C.SEED)
    train_loader = make_loader(train, tok, shuffle=True, generator=g)
    val_loader = make_loader(val, tok, shuffle=False)

    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("  params : {:,} total / {:,} trainable".format(n_params, n_train))

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(WARMUP_FRAC * total_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    lossf = nn.CrossEntropyLoss()

    C.banner("04 - DISTILBERT: training")
    print("  batch={}  max_len={}  lr={}  epochs={}".format(
        BATCH_SIZE, MAX_LEN, LR, EPOCHS))
    print("  steps/epoch={}  total_steps={}  warmup_steps={} ({:.0%})".format(
        steps_per_epoch, total_steps, warmup_steps, WARMUP_FRAC))

    history, best_f1, best_epoch = [], -1.0, -1
    t_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, seen = 0.0, 0
        t0 = time.perf_counter()
        for step, (ids, mask, y) in enumerate(train_loader, 1):
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(ids, mask)
            loss = lossf(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
            if step % 100 == 0 or step == steps_per_epoch:
                print("    epoch {} step {:>4}/{}  train_loss={:.4f}  lr={:.2e}".format(
                    epoch, step, steps_per_epoch, running / seen,
                    sched.get_last_lr()[0]))
        train_loss = running / seen
        epoch_s = time.perf_counter() - t0

        m = evaluate(model, val_loader, device)
        print("  epoch {}: train_loss={:.4f}  val_loss={:.4f}  "
              "val_acc={:.4f}  val_macro_f1={:.4f}  ({:.1f}s)".format(
                  epoch, train_loss, m["loss"], m["accuracy"],
                  m["macro_f1"], epoch_s))
        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 5),
            "val_loss": round(m["loss"], 5),
            "val_accuracy": round(m["accuracy"], 5),
            "val_macro_f1": round(m["macro_f1"], 5),
            "seconds": round(epoch_s, 1),
        })

        if m["macro_f1"] > best_f1:
            best_f1, best_epoch = m["macro_f1"], epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                    "config": {
                        "model_name": MODEL_NAME, "max_len": MAX_LEN,
                        "batch_size": BATCH_SIZE, "lr": LR,
                        "head": "single Linear on [CLS]",
                    },
                },
                CKPT_PATH,
            )
            print("    -> new best val macro-F1, checkpoint saved")
        else:
            print("    -> no improvement over epoch {} ({:.4f})".format(
                best_epoch, best_f1))

    total_s = time.perf_counter() - t_start
    C.banner("04 - DISTILBERT: done")
    print("  best epoch      : {}  (val macro-F1 {:.4f})".format(best_epoch, best_f1))
    print("  total train time: {:.1f}s ({:.1f} min)".format(total_s, total_s / 60))
    print("  checkpoint      : {}  ({:.1f} MB)".format(
        CKPT_PATH.name, CKPT_PATH.stat().st_size / 1024 ** 2))
    print("  peak GPU memory : {:.2f} GiB".format(
        torch.cuda.max_memory_allocated() / 1024 ** 3))

    early_stopped = best_epoch < EPOCHS
    C.save_json(
        {
            "config": {
                "model_name": MODEL_NAME, "head": "single Linear(768->2) on [CLS]",
                "head_note": "deliberately simpler than HF's default "
                             "pre_classifier+ReLU+dropout+classifier head",
                "max_len": MAX_LEN, "batch_size": BATCH_SIZE, "lr": LR,
                "optimizer": "AdamW", "epochs": EPOCHS,
                "warmup_frac": WARMUP_FRAC, "warmup_steps": warmup_steps,
                "total_steps": total_steps, "grad_clip": 1.0, "seed": C.SEED,
                "scheduler": "linear decay after linear warmup",
            },
            "params_total": n_params,
            "history": history,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_f1,
            "early_stopped": early_stopped,
            "early_stopping_note": (
                "Best-on-val checkpoint is kept. With only 3 epochs there is no "
                "patience-based abort; 'early_stopped' means the best epoch was "
                "not the last."
            ),
            "total_train_seconds": round(total_s, 1),
            "peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
            "device": torch.cuda.get_device_name(0),
            "checkpoint": str(CKPT_PATH.relative_to(C.ROOT)),
        },
        C.RESULTS / "04_distilbert.json",
        "04_distilbert",
    )


if __name__ == "__main__":
    main()
