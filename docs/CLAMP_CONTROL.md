# The clamp-and-relax control

Written for: whoever picks this project up next.

> **Side experiment, not the main result.** The study this project reports is the
> λ ladder — see [`RESULTS.md`](RESULTS.md), which is what the talk presents.
> This document records a zero-parameter control run afterwards. It is kept
> because it is the only protocol here with nothing for an objective to game,
> and because it explains a pathology the main pipeline ran into. Nothing in the
> presentation depends on it.

Everything here comes from the `DCAmodel-serine_protease` model (trained on the
71,551-sequence train split of `iter_aln_v2`) and the results trees
`results/ladder/` and `results/clamp/`. Figures are in `figures/ladder/` and
`figures/clamp/`; rebuild either in seconds with `make_all.py`.

The question throughout: **take wild-type rat trypsin, put a serine where its
Asp189 is, and ask what the model thinks the rest of the protein should become.**
MSA column 201 is that site (label `189`); the wild type has `D`, chymotrypsins
have `S`.

---

## 1. The headline

Two protocols were run against each other.

**The λ ladder** (`results/ladder/`) steers sampling with a bias term
`H = β·E_DCA − λ·log P(S)`, scanning λ and β. It is the main pipeline and it
produces designs that score well.

**Clamp-and-relax** (`results/clamp/`) has no bias term at all. The critic column
is simply frozen at `S` and the chains sample `H = β·E_DCA` over everything else.
Zero free parameters, nothing for an objective to game.

Clamp-and-relax was added as the null control the ladder was missing. It is not
a null. It produces the single strongest result in the project — but only in one
corner of its own grid, and that corner is one the ladder never visits.

---

## 2. Clamp-and-relax: the full grid

`P(S | context)` is the model's conditional probability of serine at column 201
given everything else the chains settled on. Under the clamp it is **not** an
objective — nothing optimises it — so it is a clean diagnostic: it says how well
the context the model relaxed into actually supports the serine imposed on it.

Four conditions. `graft` = only the 79-column grafting window (165–243) may move;
`free` = all 265 columns may move. `clamp` = S imposed at 201; `relax` = the
wild-type D left in place, the matched control.

| condition | β | P(S) median | p90 | frac > 0.5 | hamming to WT | CDE |
|---|---|---|---|---|---|---|
| clamp_graft | 1 | 0.00415 | 0.092 | 1.4 % | 13 | 0.485 |
| clamp_graft | 1.5 | 0.00069 | 0.010 | 0.0 % | 9 | 0.462 |
| clamp_graft | 2 | 0.00065 | 0.001 | 0.0 % | 8 | 0.458 |
| clamp_graft | 3 | 0.00065 | 0.001 | 0.0 % | 8 | 0.456 |
| clamp_graft | 5 | 0.00063 | 0.001 | 0.0 % | 7 | 0.455 |
| relax_graft | 1 | 0.00061 | 0.002 | 0.0 % | 12 | 0.478 |
| relax_graft | 2 | 0.00049 | 0.001 | 0.0 % | 8 | 0.459 |
| relax_graft | 5 | 0.00052 | 0.001 | 0.0 % | 7 | 0.455 |
| **clamp_free** | **1** | **0.22790** | **0.913** | **34.2 %** | **80** | **0.636** |
| clamp_free | 2 | 0.00088 | 0.003 | 0.0 % | 38 | 0.403 |
| clamp_free | 5 | 0.00073 | 0.001 | 0.0 % | 36 | 0.393 |
| relax_free | 1 | 0.00105 | 0.011 | 0.5 % | 69 | 0.597 |
| relax_free | 2 | 0.00044 | 0.001 | 0.0 % | 38 | 0.402 |
| relax_free | 5 | 0.00048 | 0.001 | 0.0 % | 36 | 0.393 |

**One cell out of fourteen is alive.** Everything else sits on a floor of
≈ 0.0007, which is what `P(S)` looks like when the context is still trypsin.

### 2.1 The one live cell is real

`clamp_free` at β=1 beats its own matched control by **217×** in median `P(S)`,
and a third of its chains reach contexts that genuinely prefer serine against
0.5 % of control chains. Crucially both conditions drift about equally far from
the wild type (80 vs 69 substitutions), so this is not "the clamped run moved
more". The family is 10.4 % serine at column 201, so a run that merely wandered
toward the family average would raise `P(S)` on its own — `relax_free` is
exactly that run, and it does not.

The pocket agrees. Column 241 (label `226`), the site where trypsins and
chymotrypsins genuinely differ:

| | G | S | A | T |
|---|---|---|---|---|
| natural trypsins | 0.91 | 0.03 | 0.04 | — |
| **natural chymotrypsins** | 0.10 | **0.28** | **0.22** | **0.21** |
| clamp, all sites free, β=1 | 0.50 | **0.30** | 0.09 | — |
| no clamp, all sites free, β=1 | 0.95 | 0.03 | 0.02 | — |
| clamp, grafting window, β=1 | 0.67 | 0.11 | 0.08 | — |

Serine at 226 goes from 3 % in the control to **30 %** under the clamp, against a
natural-chymotrypsin frequency of 28 %. That column is essentially correct.

### 2.2 Two things kill it, independently

**Window size.** `clamp_graft` reaches `P(S)` = 0.004, fifty times less than
`clamp_free`. Of the 71 columns that move in `clamp_free`, **50 lie outside the
grafting segment**. The serine's support is paid for in residues the λ ladder
never allows to move. The 79-column window is too small, and that is a direct,
quantitative statement, not an impression.

**Cooling.** β=2 drops `P(S)` by 250× and puts column 226 back to G93. Every
colder rung is *further* from the chymotrypsins than β=1. Cooling is the
ladder's mechanism for sharpening designs, and here it destroys the effect.

### 2.3 What the model will not do

Across every β, the clamp leaves these untouched:

* **loop 185–188** stays `FLEG` at 89–100 % (natural chymotrypsins: G34/D18/G35/G30)
* **172** stays `Y` at 94–100 % (natural chymotrypsins: Y21/**W16**/F13)

Those are precisely the determinants Hedstrom, Szilagyi & Rutter (*Science* 1992,
255:1249) had to swap to get a functional switch. The model does not reach them.

### 2.4 A pathology that turned out to be an artifact

An earlier pilot steered on the log-odds `log P(S)/P(D)` and turned the pocket
into **glycine**. Under the clamp, glycine load over the grafting window is
17.6–19.4 %, against 15.7 % in natural trypsins — and the matched controls give
18.3–19.3 %, i.e. the same. So the glycine collapse was entirely an artifact of
a gameable objective, not a property of the model or of cold sampling. This is
the clearest argument in the project for keeping a zero-parameter control.

---

## 3. How to read this against the λ ladder

The ladder's designs score well on `P(S)`, but `P(S)` is what the ladder
optimises, so the number cannot also be the evidence. Clamp-and-relax provides
the reading that does not have that circularity, and it says:

* the coupling from column 201 to the rest of the protein is **real but weak and
  long-ranged** — it reaches 226 strongly, and reaches 50 columns outside the
  graft window collectively;
* it is **destroyed by cooling** and **starved by a 79-column window**;
* so a ladder run that is both cold and windowed is measuring its own bias term
  more than it is measuring the model.

That is the finding to carry forward.

---

## 4. Caveats — read before building on any of this

1. **Acceptance collapses when cold.** 0.034 at β=1, 0.008 at β=2. At β≥2 that is
   roughly 500 accepted moves over 78 sites. The β=2 and β=5 ensembles differ by
   two columns out of 265 and have nearly identical flip lists: they are **stuck
   near the wild type, not converged to a minimum**. Treat every cold rung as a
   lower bound on movement, not as an equilibrium result.

2. **The one positive result sits at the warmest, most diffuse rung**, which is
   also the one whose chains are least settled. `SI S6` (`figures/ladder/`) now
   checks drift from the traces — the worst traced rung moves ~20 % IQR — but the
   clamp-free β=1 condition deserves a longer run before it goes in a paper.

3. **`d_nat` is not comparable between `graft` and `free`.** `add_natural_neighbour.py`
   is called with each condition's own `--allowed-cols`, so the graft conditions
   are scored over 67 columns and the free conditions over ~250. The numbers in
   the results CSVs are correct; comparing them *across* window sizes is not.

4. **`clamp_free` had no matched control until after the fact.** `relax_free` was
   run separately and later. Both used the same seed, chains and step count, but
   they were not run in the same invocation.

5. **`P(S | context)` is a model-internal quantity.** It is not a prediction of
   catalytic activity, and nothing here has been assayed.

---

## 5. Reproducing

```bash
bash analysis/run_clamp.sh                   # all four conditions, ~18 min (M-series GPU)
bash analysis/run_clamp.sh clamp_graft       # one condition

cd analysis/figures_clamp && ../../.venv/bin/python make_all.py    # 3 figures, ~7 s
cd analysis/figures_ladder && ../../.venv/bin/python make_all.py   # 15 figures, ~34 s
```

Stage 1 writes ensembles; Stage 2 draws from them. They are separate on purpose:
Stage 2 is cheap and Stage 1 is not. See [`DATA.md`](DATA.md) for what must be
present before Stage 1 will start.
