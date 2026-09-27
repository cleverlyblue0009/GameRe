"""10 - Paper figures.

Fig 2  macro-F1 per model on the clean test set, with 95% bootstrap CI bars.
Fig 3  grouped bars: macro-F1 drop per model per adversarial attack.
Fig 4  accuracy-latency frontier, log-scale x-axis. Points: LR, NB,
       RF(n_jobs=1), RF(n_jobs=-1), DistilBERT-CPU, DistilBERT-GPU (distinct
       marker) and the default cascade.

Every figure is written as both PDF (vector, for LaTeX) and PNG at 300 dpi,
using the viridis palette.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

DPI = 300
MODELS = ("LR", "NB", "RF", "DistilBERT")
ATTACKS = ("leetspeak", "char_insert", "whitespace", "swap")

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.constrained_layout.use": True,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,   # TrueType, so the PDF is editable/embeddable
    "ps.fonttype": 42,
})


def viridis(n):
    return [cm.viridis(x) for x in np.linspace(0.12, 0.88, n)]


def save(fig, stem):
    stem = stem + C.suffix()
    pdf = C.FIGURES / (stem + ".pdf")
    png = C.FIGURES / (stem + ".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=DPI)
    plt.close(fig)
    print("  [saved] {}  +  {}".format(pdf.name, png.name))


# --------------------------------------------------------------------------
def fig2(ev):
    m = ev["metrics"]
    b = ev["bootstrap"]["per_model"]
    vals = [m[k]["macro_f1"] for k in MODELS]
    lo = [vals[i] - b[k]["ci_low"] for i, k in enumerate(MODELS)]
    hi = [b[k]["ci_high"] - vals[i] for i, k in enumerate(MODELS)]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    cols = viridis(len(MODELS))
    bars = ax.bar(range(len(MODELS)), vals, yerr=[lo, hi], capsize=3.5,
                  color=cols, edgecolor="black", linewidth=0.5,
                  error_kw={"elinewidth": 0.9, "ecolor": "black"})
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Clean test-set performance (95% bootstrap CI)")
    lowest = min(b[k]["ci_low"] for k in MODELS)
    ax.set_ylim(max(0.0, lowest - 0.04), 1.0)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                "{:.3f}".format(v), ha="center", va="bottom", fontsize=7.5)
    save(fig, "fig2_macro_f1")


def fig3(adv):
    res = adv["results"]
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    n_m, n_a = len(MODELS), len(ATTACKS)
    width = 0.8 / n_m
    cols = viridis(n_m)
    x = np.arange(n_a)
    for i, mdl in enumerate(MODELS):
        drops = [res[a]["per_model"][mdl]["abs_drop"] for a in ATTACKS]
        ax.bar(x + i * width - 0.4 + width / 2, drops, width,
               label=mdl, color=cols[i], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", "\n") for a in ATTACKS])
    ax.set_ylabel("Macro-F1 drop from clean")
    ax.set_title("Degradation under character-level attacks")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    save(fig, "fig3_adversarial_drop")


def fig4(ev, lat, casc):
    """Accuracy-latency frontier, log-scale latency axis."""
    m = ev["metrics"]
    pts = []

    def e2e(d):
        return d["end_to_end_single"]["mean_ms"]

    cl = lat["cpu_classical"]
    pts.append(("LR (CPU)", e2e(cl["LR"]), m["LR"]["macro_f1"], "o", False))
    pts.append(("NB (CPU)", e2e(cl["NB"]), m["NB"]["macro_f1"], "s", False))
    pts.append(("RF n_jobs=1", e2e(cl["RF"]["n_jobs=1"]),
                m["RF"]["macro_f1"], "^", False))
    pts.append(("RF n_jobs=-1", e2e(cl["RF"]["n_jobs=-1"]),
                m["RF"]["macro_f1"], "v", False))
    pts.append(("DistilBERT (CPU)", e2e(lat["cpu_distilbert"]),
                m["DistilBERT"]["macro_f1"], "D", False))

    gpu = lat.get("gpu_distilbert", {})
    if "fp32" in gpu:
        pts.append(("DistilBERT (GPU)", e2e(gpu["fp32"]),
                    m["DistilBERT"]["macro_f1"], "*", True))

    if casc:
        band = "[0.4, 0.7]"
        c = casc["results"][band]["clean"]
        if c.get("expected_ms_bert_cpu"):
            pts.append(("Cascade {} (CPU)".format(band),
                        c["expected_ms_bert_cpu"], c["macro_f1"], "P", True))

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    cols = viridis(len(pts))
    for (label, xv, yv, marker, special), col in zip(pts, cols):
        ax.scatter(xv, yv, s=110 if special else 62, marker=marker,
                   color=col, edgecolor="black",
                   linewidth=1.0 if special else 0.5,
                   zorder=3, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("End-to-end latency per message (ms, log scale)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Accuracy-latency frontier")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    save(fig, "fig4_accuracy_latency")


def main():
    C.banner("10 - FIGURES")
    C.announce_label_set()
    sfx = C.suffix()
    need = {
        "05_evaluate{}.json".format(sfx): "script 05",
        "06_adversarial{}.json".format(sfx): "script 06",
        "07_latency{}.json".format(sfx): "script 07",
    }
    missing = [f for f in need if not (C.RESULTS / f).exists()]
    if missing:
        print("  cannot build figures; missing: {}".format(missing))
        print("  run the corresponding scripts first.")
        sys.exit(1)

    ev = C.load_json(C.RESULTS / "05_evaluate{}.json".format(sfx))
    adv = C.load_json(C.RESULTS / "06_adversarial{}.json".format(sfx))
    lat = C.load_json(C.RESULTS / "07_latency{}.json".format(sfx))
    casc_path = C.RESULTS / "08_cascade{}.json".format(sfx)
    casc = C.load_json(casc_path) if casc_path.exists() else None
    if casc is None:
        print("  note: 08_cascade.json absent - Fig 4 omits the cascade point")

    fig2(ev)
    fig3(adv)
    fig4(ev, lat, casc)
    print("\n  all figures written to {}/ as PDF + PNG at {} dpi "
          "(viridis)".format(C.FIGURES.name, DPI))


if __name__ == "__main__":
    main()
