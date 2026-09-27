# Toxicity detection on CONDA: classical baselines vs DistilBERT

A reproducible empirical study comparing TF-IDF classical classifiers against a
fine-tuned DistilBERT on in-game (Dota 2) chat toxicity, using the **CONDA**
dataset of Weld et al. (ACL-IJCNLP Findings 2021). The study measures clean
accuracy, robustness to character-level obfuscation, inference latency, and a
two-stage cascade that trades a little accuracy for a lot of throughput.

> **This is a fresh reimplementation, not a recovery.** An earlier version of
> this project was deleted along with its Claude Code transcripts. Nothing was
> replayed from history; every script here was rewritten from a written spec
> plus one surviving memory note. Where a number can be compared to the old
> paper, `results/SUMMARY.md` reports both side by side and flags differences
> rather than reconciling them silently.

## Dataset

CONDA ships three CSVs from <https://github.com/usydnlp/CONDA>:

| File | Rows | Use |
|---|---|---|
| `CONDA_train.csv` | 26,921 | stratified 80/20 (seed 42) → **train** / **val** |
| `CONDA_valid.csv` | 8,974 | held-out **test** |
| `CONDA_test.csv` | 8,974 | **unused** — unlabelled blind CodaLab set |

`CONDA_test.csv` carries no `intentClass` column, so it cannot be scored
locally; script 01 asserts this and excludes it.

### The binary label

CONDA has no toxic/non-toxic column. It has a 4-way utterance-level
`intentClass` and token-level `slotClasses`. Script 01 enumerates candidate
mappings and prints the counts each produces on the test set. Exactly one
reproduces the original study's 2,345 toxic / 6,628 non-toxic split:

```
toxic = 1  if intentClass != 'O'   (i.e. E ∪ A ∪ I)
toxic = 0  if intentClass == 'O'
```

after dropping one row whose `utterance` is blank (`Id` 2525, class `O`),
which is what turns 8,974 raw rows into the 8,973 reported.

**Caveat worth reading before citing any number here.** CONDA's own README
treats only `E` (explicit) and `I` (implicit) as the toxicity classes; `A` is
an *action* class, utterances asking others to report or commend a player. The
mapping above therefore labels 580 `A` utterances toxic, including 75 that
contain "commend"/"comment" — outright positive messages. Only 5.0% of `A`
utterances carry a token-level toxicity (`T`) tag, essentially the same rate as
the neutral class `O` (3.7%), against 80.6% for `E`. Every split therefore also
carries a `label_EI` column implementing `toxic = intentClass ∈ {E, I}` so the
sensitivity analysis can be run without re-deriving anything. See
`results/SUMMARY.md`.

Resulting splits:

| Split | n | toxic | % toxic | (`label_EI` % toxic) |
|---|---|---|---|---|
| train | 21,531 | 5,551 | 25.78 | 19.34 |
| val | 5,383 | 1,388 | 25.78 | 19.60 |
| test | 8,973 | 2,345 | 26.13 | 19.67 |

No hyperparameter is ever selected on the test split.

## Environment

Python 3.12, CUDA-enabled PyTorch. Script 00 is a hard gate: it refuses to
continue unless `torch.cuda.is_available()` is `True`, because a CPU-only torch
wheel silently installing itself is a known failure on this machine and would
turn script 04 into a multi-hour job.

> **Note on venv location.** The `C:` drive on the development machine was
> completely full (0 bytes free of 307 GB; the dominant consumers were game
> installs, not project data). The virtual environment, pip cache and pip temp
> directory were therefore placed on `E:`, which had 169 GB free. No user data
> was deleted. If you reproduce this elsewhere, a venv in `.venv/` works fine;
> nothing in the code depends on the venv's location.

```bash
# from the repo root
python -m venv .venv                       # or any location with ~8 GB free
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python -m pip install "transformers>=4.44" scikit-learn pandas \
    numpy statsmodels matplotlib nltk tqdm joblib
```

Exact hardware and library versions are recorded to `results/env.json`.

## Reproducing every table and figure

Run in order. Each script writes a JSON record to `results/` and is
independently re-runnable given its predecessors' outputs.

| Step | Command | Produces |
|---|---|---|
| 00 | `python src/00_env.py` | `results/env.json` — CUDA gate, CPU/GPU/RAM, versions |
| 01 | `python src/01_data.py` | `results/01_data.json`, `data/processed/{train,val,test}.csv` — label-mapping search, split sizes |
| 02 | `python src/02_preprocess.py` | `results/02_preprocess.json`, `*_prep.csv` — stage-by-stage preprocessing trace |
| 03 | `python src/03_classical.py` | `results/03_classical.json`, `checkpoints/*.joblib` — TF-IDF + LR/NB/RF |
| 04 | `python src/04_distilbert.py` | `results/04_distilbert.json`, `checkpoints/distilbert_best.pt` — GPU fine-tuning |
| 05 | `python src/05_evaluate.py` | `results/05_evaluate.json` — **main metrics table**, bootstrap CIs, **McNemar table** |
| 06 | `python src/06_adversarial.py` | `results/06_adversarial.json` — **adversarial drops**, **ablation**, audit samples |
| 07 | `python src/07_latency.py` | `results/07_latency.json` — **latency table** incl. RF `n_jobs` and GPU rows |
| 08 | `python src/08_cascade.py` | `results/08_cascade.json` — **cascade table** |
| 09 | `python src/09_analysis.py` | `results/09_analysis.json` — LR coefficients, error examples |
| 10 | `python src/10_figures.py` | `figures/fig{2,3,4}.{pdf,png}` |
| — | `python src/11_summary.py` | `results/SUMMARY.md` — every table, paper-ready, with old-vs-new columns |

### The sensitivity run (`E ∪ I`)

Because the primary label definition is contestable (see the `A`-class caveat
above), the whole pipeline runs a second time against
`toxic = intentClass ∈ {E, I}`. The label set is chosen by an environment
variable, and every artefact the second run writes is suffixed `_ei`, so the
two never collide:

```bash
# primary run (E ∪ A ∪ I) — the default
for s in 03_classical 04_distilbert 05_evaluate 06_adversarial \
         07_latency 08_cascade 09_analysis 10_figures; do
  python src/$s.py
done

# sensitivity run (E ∪ I)
export TOX_LABEL_SET=ei
for s in 03_classical 04_distilbert 05_evaluate 06_adversarial \
         07_latency 08_cascade 09_analysis 10_figures; do
  python src/$s.py
done
unset TOX_LABEL_SET

python src/11_summary.py    # reads both, writes one SUMMARY.md
```

Nothing is shared between the two runs but the raw text and the splits: each
gets its own TF-IDF fit, its own hyperparameter selection and its own
fine-tuned DistilBERT. `src/11_summary.py` reads both and reports, per
conclusion, whether it survives the label change — a claim that holds only
under the primary mapping is an artefact of labelling the action class toxic
and is flagged as label-dependent.

Which table comes from where:

- **Main metrics** (accuracy, macro-F1, weighted-F1, ROC-AUC, PR-AUC, toxic
  recall, false-positive count) → script 05.
- **Significance** — pairwise McNemar via `statsmodels`; the variant used per
  pair (exact binomial when discordant pairs < 25, otherwise chi-square with
  continuity correction) is recorded per pair → script 05.
- **Adversarial** — macro-F1 under each attack and the absolute drop from
  clean, plus the leetspeak-OFF / punctuation-OFF ablation → script 06.
- **Latency** — end-to-end (primary) and model-only, RF at `n_jobs=1` and
  `n_jobs=-1`, DistilBERT CPU and GPU (fp32 / fp16 autocast) → script 07.
- **Cascade** — escalation rate, macro-F1 with CI, expected latency → script 08.
- **Figures 2–4** → script 10.

## Design decisions that affect the numbers

These are choices a reader could reasonably have made differently; each is
recorded in the corresponding `results/*.json`.

1. **Punctuation is replaced by a space, not deleted.** This avoids welding
   words together (`you.suck` → `you suck`), but it means the
   character-insertion attack is *not* repaired: `f.u.c.k` becomes four
   one-character tokens, not `fuck`. Deleting punctuation would accidentally
   defend against that attack. The script 06 ablation quantifies this.
2. **The gaming stopword whitelist is currently a no-op.** None of
   `{gg, wp, noob, ez, gl, hf, glhf, op, nerf, buff}` appears in NLTK's
   English stopword list, so re-admitting them changes nothing (198 stopwords
   either way). Reported rather than quietly dropped.
3. **Attacks are applied to raw text**, then pushed through each model's real
   preprocessing. Perturbing already-preprocessed text would let the classical
   pipeline "clean" an attack it never saw and would overstate its robustness.
4. **DistilBERT's head is a single `Linear(768→2)` on `[CLS]`**, not
   HuggingFace's default `pre_classifier + ReLU + dropout + classifier`.
5. **End-to-end latency is the primary metric.** Model-only latency hides the
   Python preprocessing that dominates the classical models' real cost.
6. **The classical ablation changes inference-time preprocessing only**; the
   models remain those trained with the full pipeline, so the ablation
   includes a train/inference mismatch that is itself part of the finding.

## Layout

```
src/            00–11, numbered in execution order; common.py + predict.py shared
data/raw/       CONDA CSVs (gitignored; script 01 re-downloads them)
data/processed/ splits and preprocessed splits (gitignored)
checkpoints/    fitted sklearn pipelines + DistilBERT checkpoint (gitignored)
results/        one JSON per script, plus SUMMARY.md
figures/        fig2–fig4 as PDF + PNG at 300 dpi
logs/           run logs
```

`predict.py` is the reason scripts 06–08 can be trusted: every entry point
takes **raw** utterances and applies that model's own preprocessing
internally, so there is one code path for real inference, adversarial
evaluation and latency timing.

## Citation

```bibtex
@inproceedings{weld-etal-2021-conda,
    title = "{CONDA}: a {CON}textual Dual-Annotated dataset for in-game
             toxicity understanding and detection",
    author = "Weld, Henry and Huang, Guanghao and Lee, Jean and Zhang, Tongshu
              and Wang, Kunze and Guo, Xinghong and Long, Siqu and
              Poon, Josiah and Han, Caren",
    booktitle = "Findings of the ACL: ACL-IJCNLP 2021",
    year = "2021", pages = "2406--2416",
    url = "https://aclanthology.org/2021.findings-acl.213",
}
```
