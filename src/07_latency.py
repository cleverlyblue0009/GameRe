"""07 - Inference latency.

PRIMARY metric is END-TO-END latency: raw string in -> preprocessing ->
vectoriser/tokeniser -> predict -> P(toxic) out. That is what a moderation
service actually pays per message. Model-only latency (feature matrix or
pre-tokenised tensor already in hand) is reported alongside it, because the
gap between the two is itself a finding: for the classical models the Python
preprocessing dominates, which a model-only number would hide completely.

Protocol
  warmup   100 single-example predictions (discarded)
  timed    1000 single-example predictions -> mean / p50 / p95 / p99
  batch    batch-32 throughput, plus amortised ms per example
  clock    time.perf_counter()

Random Forest is timed at BOTH n_jobs=1 and n_jobs=-1 on the *same fitted
model* (mutating only the n_jobs attribute), because thread-pool overhead can
make n_jobs=-1 slower than serial on single examples.

DistilBERT CPU uses the same 1000 timed single examples as everything else
(not a reduced count), and torch.get_num_threads() is recorded. DistilBERT GPU
is timed in eval mode under torch.inference_mode() with
torch.cuda.synchronize() before the timer stops, in fp32 and in fp16 autocast.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import predict as P

N_WARMUP = 100
N_TIMED = 1000
BATCH = 32


def stats(times_s):
    a = np.asarray(times_s, dtype=np.float64) * 1000.0  # -> ms
    return {
        "mean_ms": float(a.mean()),
        "std_ms": float(a.std(ddof=1)),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "n": int(a.size),
    }


def show(label, s):
    print("    {:<34} mean {:>8.3f}  p50 {:>8.3f}  p95 {:>8.3f}  "
          "p99 {:>8.3f} ms".format(label, s["mean_ms"], s["p50_ms"],
                                   s["p95_ms"], s["p99_ms"]))


def time_single(fn, samples, n_warmup=N_WARMUP, n_timed=N_TIMED):
    """Time fn on one example at a time, cycling through samples."""
    k = len(samples)
    for i in range(n_warmup):
        fn(samples[i % k])
    out = np.empty(n_timed)
    for i in range(n_timed):
        t0 = time.perf_counter()
        fn(samples[i % k])
        out[i] = time.perf_counter() - t0
    return out


def time_batch(fn, samples, batch=BATCH, n_warmup=10, n_timed=100):
    k = len(samples)
    for i in range(n_warmup):
        fn([samples[(i * batch + j) % k] for j in range(batch)])
    out = np.empty(n_timed)
    for i in range(n_timed):
        chunk = [samples[(i * batch + j) % k] for j in range(batch)]
        t0 = time.perf_counter()
        fn(chunk)
        out[i] = time.perf_counter() - t0
    per_batch = np.asarray(out) * 1000.0
    return {
        "batch_size": batch,
        "mean_batch_ms": float(per_batch.mean()),
        "p95_batch_ms": float(np.percentile(per_batch, 95)),
        "amortised_ms_per_example": float(per_batch.mean() / batch),
        "throughput_examples_per_s": float(batch / (per_batch.mean() / 1000.0)),
        "n_batches": int(len(per_batch)),
    }


def main():
    C.set_seed()
    C.announce_label_set()
    C._stopwords()
    import torch

    test = pd.read_csv(C.PROC / "test_prep.csv")
    raw = test["utterance"].astype(str).tolist()
    rng = np.random.default_rng(C.SEED)
    samples = [raw[i] for i in rng.permutation(len(raw))[: max(N_TIMED, 1000)]]

    vec, cmodels = P.load_classical()
    out = {"label_set": C.LABEL_SET, "protocol": {
        "n_warmup": N_WARMUP, "n_timed": N_TIMED, "batch_size": BATCH,
        "clock": "time.perf_counter",
        "primary_metric": "end_to_end",
        "primary_metric_note": (
            "end-to-end = raw string -> preprocessing -> vectoriser/tokeniser "
            "-> predict. model_only excludes preprocessing and "
            "vectorisation/tokenisation."
        ),
    }}

    # ---------------------------------------------------------------- CPU: classical
    C.banner("07 - LATENCY: classical models on CPU")
    cls_res = {}
    # pre-transform once for the model-only path
    prep_all = [C.preprocess(t, C.CLASSICAL) for t in samples]
    X_all = vec.transform(prep_all)

    for name in ("LR", "NB"):
        mdl = cmodels[name]
        print("  {}".format(name))
        e2e = time_single(
            lambda t: P.classical_proba([t], vec, mdl), samples)
        s_e2e = stats(e2e)
        show("end-to-end (single)", s_e2e)
        mo = time_single(
            lambda t, m=mdl: m.predict_proba(vec.transform([C.preprocess(t, C.CLASSICAL)]))[:, 1],
            samples)
        # true model-only: reuse a prebuilt row
        rows = [X_all[i] for i in range(min(len(samples), N_TIMED))]
        mo2 = time_single(lambda r, m=mdl: m.predict_proba(r)[:, 1], rows)
        s_mo = stats(mo2)
        show("model-only (single)", s_mo)
        b_e2e = time_batch(lambda ts: P.classical_proba(ts, vec, mdl), samples)
        print("    batch-{} throughput: {:.1f} ex/s  "
              "({:.3f} ms/example amortised)".format(
                  BATCH, b_e2e["throughput_examples_per_s"],
                  b_e2e["amortised_ms_per_example"]))
        cls_res[name] = {"end_to_end_single": s_e2e, "model_only_single": s_mo,
                         "end_to_end_batch": b_e2e}

    # ---------------------------------------------------------------- RF n_jobs
    C.banner("07 - LATENCY: Random Forest, n_jobs=1 vs n_jobs=-1 "
             "(same fitted model)")
    rf = cmodels["RF"]
    orig = rf.n_jobs
    rf_res = {}
    for nj in (1, -1):
        rf.n_jobs = nj
        print("  RF n_jobs={}".format(nj))
        e2e = time_single(lambda t: P.classical_proba([t], vec, rf), samples)
        s_e2e = stats(e2e)
        show("end-to-end (single)", s_e2e)
        rows = [X_all[i] for i in range(min(len(samples), N_TIMED))]
        mo = time_single(lambda r: rf.predict_proba(r)[:, 1], rows)
        s_mo = stats(mo)
        show("model-only (single)", s_mo)
        b = time_batch(lambda ts: P.classical_proba(ts, vec, rf), samples)
        print("    batch-{} throughput: {:.1f} ex/s  "
              "({:.3f} ms/example amortised)".format(
                  BATCH, b["throughput_examples_per_s"],
                  b["amortised_ms_per_example"]))
        rf_res["n_jobs={}".format(nj)] = {
            "end_to_end_single": s_e2e, "model_only_single": s_mo,
            "end_to_end_batch": b, "n_jobs": nj,
        }
    rf.n_jobs = orig
    print("  (restored fitted model n_jobs={})".format(orig))
    cls_res["RF"] = rf_res
    out["cpu_classical"] = cls_res

    # ---------------------------------------------------------------- DistilBERT CPU
    C.banner("07 - LATENCY: DistilBERT on CPU")
    n_threads = torch.get_num_threads()
    print("  torch.get_num_threads() = {}".format(n_threads))
    print("  torch.get_num_interop_threads() = {}".format(
        torch.get_num_interop_threads()))
    bert_cpu, tok, _ = P.load_bert(device="cpu")
    bert_cpu.eval()

    e2e = time_single(
        lambda t: P.bert_proba([t], bert_cpu, tok, device="cpu"), samples)
    s_e2e = stats(e2e)
    show("end-to-end (single)", s_e2e)

    # model-only: pre-tokenise, time just the forward pass
    enc_cache = [
        tok([C.preprocess(t, C.BERT)], truncation=True, max_length=128,
            padding=True, return_tensors="pt")
        for t in samples[:N_TIMED]
    ]

    def fwd_cpu(enc):
        with torch.inference_mode():
            return bert_cpu(enc["input_ids"], enc["attention_mask"])

    mo = time_single(fwd_cpu, enc_cache)
    s_mo = stats(mo)
    show("model-only (single)", s_mo)
    b = time_batch(lambda ts: P.bert_proba(ts, bert_cpu, tok, device="cpu",
                                           batch_size=BATCH), samples)
    print("    batch-{} throughput: {:.1f} ex/s  "
          "({:.3f} ms/example amortised)".format(
              BATCH, b["throughput_examples_per_s"],
              b["amortised_ms_per_example"]))
    out["cpu_distilbert"] = {
        "torch_num_threads": n_threads,
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "end_to_end_single": s_e2e, "model_only_single": s_mo,
        "end_to_end_batch": b,
    }
    del bert_cpu

    # ---------------------------------------------------------------- DistilBERT GPU
    if not torch.cuda.is_available():
        print("\nWARNING: CUDA unavailable - skipping GPU latency section.")
        out["gpu_distilbert"] = {"skipped": "cuda unavailable"}
    else:
        C.banner("07 - LATENCY: DistilBERT on GPU (fp32 and fp16 autocast)")
        bert, tok, _ = P.load_bert(device="cuda")
        bert.eval()
        gpu = {"device": torch.cuda.get_device_name(0)}

        for prec, fp16 in (("fp32", False), ("fp16_autocast", True)):
            print("  precision = {}".format(prec))

            def e2e_call(t, fp16=fp16):
                r = P.bert_proba([t], bert, tok, device="cuda",
                                 autocast_fp16=fp16)
                torch.cuda.synchronize()
                return r

            s_e2e = stats(time_single(e2e_call, samples))
            show("end-to-end (single)", s_e2e)

            enc_gpu = []
            for t in samples[:N_TIMED]:
                e = tok([C.preprocess(t, C.BERT)], truncation=True,
                        max_length=128, padding=True, return_tensors="pt")
                enc_gpu.append({k: v.to("cuda") for k, v in e.items()})

            def fwd(enc, fp16=fp16):
                with torch.inference_mode():
                    if fp16:
                        with torch.autocast("cuda", dtype=torch.float16):
                            o = bert(enc["input_ids"], enc["attention_mask"])
                    else:
                        o = bert(enc["input_ids"], enc["attention_mask"])
                torch.cuda.synchronize()
                return o

            s_mo = stats(time_single(fwd, enc_gpu))
            show("model-only (single)", s_mo)

            def batch_call(ts, fp16=fp16):
                r = P.bert_proba(ts, bert, tok, device="cuda",
                                 batch_size=BATCH, autocast_fp16=fp16)
                torch.cuda.synchronize()
                return r

            b = time_batch(batch_call, samples)
            print("    batch-{} throughput: {:.1f} ex/s  "
                  "({:.3f} ms/example amortised)".format(
                      BATCH, b["throughput_examples_per_s"],
                      b["amortised_ms_per_example"]))
            gpu[prec] = {"end_to_end_single": s_e2e,
                         "model_only_single": s_mo,
                         "end_to_end_batch": b}
        gpu["sync_note"] = ("torch.cuda.synchronize() called before stopping "
                            "every timer; eval mode + torch.inference_mode()")
        out["gpu_distilbert"] = gpu

    # ---------------------------------------------------------------- summary
    C.banner("07 - LATENCY: summary (END-TO-END single-example, primary)")
    hdr = "{:<28}{:>10}{:>10}{:>10}{:>14}".format(
        "configuration", "mean ms", "p95 ms", "p99 ms", "batch32 ex/s")
    print(hdr)
    print("-" * len(hdr))

    def line(label, d):
        print("{:<28}{:>10.3f}{:>10.3f}{:>10.3f}{:>14.1f}".format(
            label, d["end_to_end_single"]["mean_ms"],
            d["end_to_end_single"]["p95_ms"],
            d["end_to_end_single"]["p99_ms"],
            d["end_to_end_batch"]["throughput_examples_per_s"]))

    line("LR (CPU)", cls_res["LR"])
    line("NB (CPU)", cls_res["NB"])
    line("RF n_jobs=1 (CPU)", rf_res["n_jobs=1"])
    line("RF n_jobs=-1 (CPU)", rf_res["n_jobs=-1"])
    line("DistilBERT (CPU)", out["cpu_distilbert"])
    if "fp32" in out.get("gpu_distilbert", {}):
        line("DistilBERT GPU fp32", out["gpu_distilbert"]["fp32"])
        line("DistilBERT GPU fp16", out["gpu_distilbert"]["fp16_autocast"])

    C.save_json(out, C.RESULTS / ("07_latency" + C.suffix() + ".json"),
                "07_latency" + C.suffix())


if __name__ == "__main__":
    main()
