# Training the model: bmDCA with Parallel Trajectory Tempering

Written for: whoever reruns or reviews this analysis.

> **Note — the command below is the abandoned `model_ptt_v2` run, kept for
> provenance.** It was executed in the old sibling `work/` directory and its
> paths are left exactly as they were run. The surviving model,
> `DCAmodel-serine_protease`, was trained with the same package and the same
> `--strategy ptt`, but not with these settings: its own training log
> (`$TRYP_DATA_ROOT/DCAmodel-serine_protease/adabmDCA.log` — see `docs/DATA.md`
> for where the data root is) records CUDA rather than MPS, 10 240 chains rather
> than 2 048, **no** sequence reweighting, and a 71 551-sequence train split of
> `iter_aln_v2.faa` with a 17 888-sequence held-out split, stopped on a
> validation plateau rather than on `--target`. What this document says about
> what PTT is, where to get it and what it writes applies to both runs; the
> command block does not. **The surviving model's own settings are recorded in
> [§ The surviving model](#the-surviving-model-dcamodel-serine_protease) at the
> foot of this file.**

## What "bmDCA with PTT" is, and where it comes from

**PTT = Parallel Trajectory Tempering.** It is a training strategy for
energy-based models: instead of a single set of persistent chains, it keeps a
ladder of models saved along the training trajectory — from an exactly
samplable independent-site profile at the bottom up to the current model — and
swaps configurations between the rungs. Because the bottom of the ladder has a
known normalisation, the ladder also yields an estimate of log *Z*, and with it
an online log-likelihood.

Why it matters here: the stopping rule of ordinary persistent-chain bmDCA is
the Pearson correlation of the two-site statistics *measured on its own
chains*. If those chains have not equilibrated, that number is optimistic and
training can stop early. That is exactly the failure the previous
`model_pcd_serine_protease` showed — its log reported `Pearson = 0.950`, while
an independent resampling of the same parameters measured 0.755 on the same
chains. PTT is designed to remove that failure mode.

Reference: N. Béreux, A. Decelle, C. Furtlehner, B. Seoane, *Equilibrium
Training of Energy-Based Models with Parallel Trajectory Tempering*,
arXiv:2607.27077 (2026).

## Where to get it — the trap

PTT is **not in any released version** of adabmDCA.

| | version | has PTT |
|---|---|---|
| PyPI `adabmDCA` (latest) | 1.0.0 | **no** |
| GitHub tag `v1.0.0` / `main` | 1.0.0 | **no** |
| GitHub `develop` branch | 2.0.0 (unreleased) | **yes**, and it is the default strategy |

The published documentation at <https://spqb.github.io/adabmDCApy/> is built
from `develop`, so the docs describe PTT while `pip install adabmDCA` gives you
a version that does not have it.

It is installed here into a **separate virtual environment**, pinned to a
commit, so that the `adabmDCA 1.0.0` the analysis repository depends on is left
untouched:

```bash
python3 -m venv --system-site-packages work/.venv-ptt
work/.venv-ptt/bin/python -m pip install \
    "git+https://github.com/spqb/adabmDCApy.git@02ccf95"
work/.venv-ptt/bin/adabmDCA --version     # 2.0.0
```

`02ccf95` is `develop` as of 2026-11-13. `develop` moves, so the pin matters.

Only *training* uses 2.0.0. Everything downstream reads the `params.dat` it
writes, whose format is byte-for-byte the one 1.0.0 already parses (the only
change between the two is that `save_params` can gzip its output).

## The command

```bash
cd work
./.venv-ptt/bin/adabmDCA train \
    -d ../iter_aln_v2.faa \
    -w data/weights_iter_aln_v2.dat \
    -o model_ptt_v2 \
    -m bmDCA --strategy ptt \
    --device mps --dtype float32 \
    --nchains 2048 --target 0.95 \
    --checkpoint-interval 50 --seed 0 --no-progress
```

Notes on the choices:

* **`--strategy ptt`** is the default on 2.0.0; it is written out so the log
  records it.
* **`-w`** reuses a weights file computed once. Sequence-identity reweighting
  over 89 439 sequences is an *O(M²)* comparison and takes about **10 minutes**
  on this machine — the single largest fixed cost of a run, and entirely
  avoidable on every rerun. The file gives **M_eff = 40 704.43**, and from it
  the pseudocount `1/M_eff = 2.457e-05`.
* **`--device mps`.** There is no CUDA on this machine (Apple silicon, 10
  cores, 24 GB). PTT has explicit MPS support: it accumulates in float32 on
  MPS and float64 elsewhere, and moves small diagnostic reductions to the CPU
  for float64. `--dtype float32` matches that.
* **`--nchains 2048`** is the 2.0.0 default. The previous work used 10 000, but
  that was on an RTX 3090; here the per-step cost would be prohibitive, and
  PTT's claim is equilibration rather than batch size. Larger samples for the
  *measurements* are drawn afterwards with `adabmDCA sample`, where they are
  cheap.
* **`--target 0.95`** is the same stopping rule as before, now resting on
  chains that PTT keeps at equilibrium.

Throughput measured on this machine: **~2.5–3 s per gradient step** (2048
chains, 10 sweeps per step).

## What it writes

| file | contents |
|---|---|
| `params.dat.gz` | the model, in the `h`/`J` text format 1.0.0 reads (gunzip before use) |
| `chains.fasta` | the 2048 persistent chains at the end of training |
| `weights.dat` | the sequence weights (identical to the input `-w` file) |
| `ptt.h5` | the PTT archive — the model ladder, for `adabmDCA sample --strategy ptt` |
| `history.csv` | per gradient step: Pearson, slope, log-likelihood, entropy, log *Z*, replicas, acceptance |
| `events.jsonl`, `adabmDCA.log`, `training.json` | the training log, ladder events and final summary |

`history.csv` logs one row per *gradient step*. PTT also runs mixing and
thermalisation rounds between gradient steps, during which no row is written,
so a gap of some minutes in that file is normal and not a stall. `events.jsonl`
is the file that shows what the ladder is doing.

## Not comparable with the earlier models

Energies carry a gauge, so nothing transfers between parameter sets. `mu` is
recalibrated on this model, and every number in [`RESULTS.md`](RESULTS.md) is recomputed
from it. Comparisons with the figures in the repository's `docs/` are
comparisons of *shape*, never of value — and in any case those used a different
alignment (L = 260) as well as a different model.

---

## The surviving model: `DCAmodel-serine_protease`

This is the model every figure in this repository is built from. The settings
below are transcribed from its own training log,
`$TRYP_DATA_ROOT/DCAmodel-serine_protease/adabmDCA.log`, which is the
authoritative record — not from the command block above, which belongs to a
different run.

| | |
|---|---|
| package | adabmDCA **2.0.0**, model `bmDCA`, seed 0 |
| trained | 2026-11-18, **6 h 44 m** (24,239 s) on **CUDA**, float32 |
| train split | `iter_aln_v2.train.fasta` — 71,551 sequences, L = 265, q = 21 |
| held-out split | `iter_aln_v2.test.fasta` — 17,888 sequences |
| reweighting | **none** (`no_reweighting: True`); clustering_seqid 0.8 recorded but unused |
| sampler | `metropolized_gibbs`, strategy **PTT**, 2 replicas, swap target acceptance 0.25 |
| chains | **10,240**, 10 sweeps per gradient step |
| optimizer | adaptive, learning rate 0.01, trust radius 0.01, centering on |
| regularisation | L2 = 0, effective pseudocount 1.3976e-05 |
| stopping | `validation_stop` (window 100, min gain 0), target_pearson 0.95, max 50,000 steps |
| **stopped at** | **20,396 gradient steps — `validation_plateau`, converged** |
| final Pearson | **0.9750** on training C_ij, **0.7607** on the held-out set |
| density | 1.0 (dense couplings, no decimation) |

Two things about this table are worth pausing on.

**The split is the point.** 71,551 + 17,888 = 89,439, exactly the rows of
`iter_aln_v2.faa`. Holding out 17,888 sequences is what makes figure F1 and
SI S1 predictions rather than consistency checks, and it is the only reason
this model replaced its predecessor. Never regenerate the split by guessing —
the figure code falls back to the whole alignment and says so, because a
fabricated split would quietly undermine the one figure meant to answer the
circularity objection.

**It was stopped by the validation set, not by `--target`.** Training ran to a
validation plateau at 20,396 steps with training Pearson 0.975 — below the
0.95 target only in the sense that it passed it and kept going, while the
held-out Pearson settled at 0.761. The gap between 0.975 and 0.761 is the
honest measure of how much of the fit is family structure and how much is the
training rows, and it is why the discriminator figure is scored on the test
split.

The validation data's own `pearson_to_training` is 0.7796 with slope 1.0174 —
i.e. the two splits are statistically the same distribution, so the held-out
set is a fair test rather than an unusual corner of the family.
