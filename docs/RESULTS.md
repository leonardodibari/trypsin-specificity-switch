# Results

Written for: whoever picks this project up next.

This is the study presented as *"Trypsin → chymotrypsin: designing a specificity
switch with a Potts model"* (L. Di Bari, S. Rossi, 20 November 2026). **Every
data figure in that talk is produced by this repository**, from the committed
results, in about 34 seconds:

```bash
cd analysis/figures_ladder && ../../.venv/bin/python make_all.py
```

Output lands in `figures/ladder/` as 300 dpi PNG and vector PDF. The
slide-by-slide map is in [§7](#7-the-talk-slide-by-slide).

All numbers below were re-derived from `results/ladder/` while writing this
document; none were copied from the slides.

---

## 1. The question

Rat trypsin and chymotrypsin share a fold and a catalytic triad but cut
different substrates. Trypsin's specificity pocket has **aspartate at position
189**; chymotrypsin has **serine**. In this alignment that site is **MSA column
201** (label `189`).

The task: start from wild-type rat trypsin and sample sequences that prefer
serine at that site, under a Potts model of the family — then ask honestly how
much of a switch that actually is.

## 2. The model

A bmDCA Potts model (adabmDCA 2.1) over the serine-protease family:
**L = 265 columns, q = 21**, fitted to a **71,551-sequence train split** with
**17,888 sequences held out and never seen**. Fields say what each site likes,
couplings what pairs of sites like. No structures, no labels. Low energy means
"looks like a real member of the family".

Holding out a test set is the whole reason this model replaced its predecessor:
it makes the discriminator figure a *prediction* rather than a consistency check.

### It tells the two families apart on unseen sequences

Scored on the 17,888 held-out sequences (`fig01_discriminator`):

| score | AUC | AP |
|---|---|---|
| log P(S)/P(D) | **0.9835** | 0.9384 |
| −P(D \| context) | 0.9832 | 0.9227 |
| P(S \| context) alone | 0.9600 | 0.8750 |

`si01_controls` splits this by sequence novelty and the signal holds: AUC stays
above 0.97 even in the most unusual novelty bin. So the model has learnt
something about the family, not about the training rows.

### And it reproduces the statistics it was fitted on

A discriminator can score well on a model that generates badly, so this is
checked separately ([`METHODS.md`](METHODS.md) §7, `results/generative/`).
Resampling at $\lambda = 0$, $\beta = 1$ from natural sequences reproduces the
training $f_i$ and $f_{ij}$ at **100 % of the finite-sample ceiling** and $C_{ij}$
at **99.3 %**, and a start from *random* sequences 252 residues away converges to
the same place (94.4 %) — so that is the equilibrium, not an artefact of where
the chains began.

The same check found the one thing worth carrying into the design runs: a start
at the **wild type** does *not* equilibrate in 4,000 sweeps, and is slower than a
random start, because the wild type sits ~143 in energy below the family mean and
the chains must climb out of it. That barrier is the same one behind the cold
clamp runs sitting at 36–38 substitutions for every $\beta \geq 2$.

## 3. The method

Sample with

```
H = β·E_DCA − λ·log P(S | σ_\i*)
```

λ pays the chain to put a serine at the critic site; β cools the energy term.
**Gaps are an ordinary state and may move**, so chains can gain or lose
residues — which is why indels are tracked separately from substitutions
throughout.

### Scoring a design fairly: Q

A low energy can just mean a rigid, highly-constrained site. So designs are
scored with

```
Q = E_DCA − μ·CDE
```

where CDE is the context-dependent entropy of the surroundings and **μ = 422.62**
is fixed once, calibrated on 20,000 natural sequences, and never refitted. Q is what is left
after subtracting what the site would have scored anyway. **Lower Q is better**,
and it is what lets a cold run and a hot run be compared on a single axis.

## 4. What the ladder does

### Turning up the bias moves everything at once

Grafting window (79 columns, 165–243), β = 1, 1000 chains × 60,000 steps
(`fig02_corner_grafting`):

| λ | P(S) | substitutions | indels | % pure substitution | Q | % chains P(S) ≥ 0.80 |
|---|---|---|---|---|---|---|
| 0 | 0.001 | 12.0 | 0.16 | 87 % | −100.6 | 0 % |
| 10 | 0.744 | 15.3 | 0.32 | 76 % | −99.7 | 32.5 % |
| 30 | 0.862 | 15.8 | 0.36 | 73 % | −97.8 | 89.5 % |
| 100 | 0.935 | 16.8 | 0.52 | 63 % | −95.5 | **100 %** |
| 300 | 0.971 | 18.2 | 0.98 | 40 % | −91.6 | **100 %** |
| 1000 | 0.989 | 20.1 | 1.90 | 13 % | −86.5 | **100 %** |

**The bias works**: P(S) goes from 0.001 to 0.99, and from λ = 100 every chain
clears the 0.80 threshold. But it buys specificity at a steady price — more
mutations (12 → 22 total Hamming), worse Q, and a collapsing share of designs
that are pure substitutions rather than indels.

### Cooling makes better designs, but fewer of them

Cool the energy, never the bias (`fig03_cooling`). At matched serine (P(S) ≈ 0.75):

| condition | β | λ | Q | % unique designs |
|---|---|---|---|---|
| grafting | 1 | 10 | −99.7 | 100 % |
| grafting + cooling | 1.5 | 15 | −106.8 | 99.7 % |
| grafting + cooling | 2 | 20 | −108.2 | 89.2 % |
| grafting + cooling | 3 | 30 | −108.7 | 54.4 % |
| grafting + cooling | 5 | 50 | −108.8 | 20.5 % |
| grafting + cooling | 10 | 100 | −108.7 | 7.9 % |

**Cooling gains ≈ 9 in Q** (−99.7 → −108.8) at the same serine content, and
almost all of it arrives by β = 2 — after which the library collapses while Q
barely improves. Cooling also drives indels to essentially zero (`indel_to_wt`
0.32 → 0.00), so cold designs are pure point mutants.

### Restricting the sites is what concentrates them

The mutation count is set by *how many columns may move*, not by λ
(`fig08_sector_enrichment`, `fig04_set_overlap`). Six candidate site sets were
compared inside the grafting segment:

| set | columns | definition |
|---|---|---|
| full segment | 79 | every column of the grafting segment |
| consensus-22 | 22 | where the two families' consensus differs most, relaxed |
| coupling-20 | 20 | most strongly coupled to site 189 |
| red sector | 16 | sector columns inside the segment |
| 1992 construct | 14 | the columns the Hedstrom constructs mutate |
| consensus-8 | 8 | the 8 columns where the consensus differs most |

**The six sets hardly agree** — six different guesses at which sites matter,
with modest overlap. Of these, 8, 22 and 79 were sampled.

`consensus-8` reaches P(S) = 0.90 at λ = 100 with **7.5 substitutions**, against
16.8 for the full 79-column window. Fewer columns means a far smaller library
though: 41.9 % unique at λ = 100, falling to 12.6 % at λ = 1000.

### The loops are left alone

Gaps may move, and they essentially do not (`si03_gaps`). The mean gap count
stays at **≈ 42 of 265 columns** — the wild type's own value — across every bias
and every temperature (range over all 48 runs: 39.0 to 42.6). The designs keep
the wild type's gap pattern; the surface loops are not being rewritten.

## 5. The grid, and what was chosen

**48 runs**: site sets {8, 22, 79} × β {1, 1.5, 2, 3, 5, 10} × λ {0, 10, 30, 100,
300, 1000} and their matched-`λ/β` equivalents. The full pool is `si02_table_full`;
the shortlist is `fig05_table`.

**No run wins everything.** The column winners are spread across conditions:
fewest mutations → `consensus-8, β=1`; best indel purity → the cold runs; best
diversity → `grafting, β=1`. Three families, three islands (`fig06_corner_selected`):

* **grafting** — most serine, worst quality, fully diverse
* **cooling** — best quality, fewest distinct designs
* **consensus-8** — only ~8 mutations

The selection in `fig05_table` is *hand-picked, not scored*: a few from every
corner, so the experiment decides rather than a scoring rule. That choice is
deliberate and is documented in `analysis/figures_ladder/selection.py`.

## 6. What this does **not** show

This is the part to keep in view (`fig07_pca`, `si05_naturals_vs_designs`,
`si07_construct_context`).

* **The designs stay inside the trypsins.** On a PCA map built from natural
  sequences only, every design sits beside the wild type. Against 20,000 natural
  sequences (13,324 trypsins, 1,993 chymotrypsins, 4,683 other), Hamming
  distance to the wild type runs **81–196** (p1–p99) for the naturals and
  **6–28** for the designs. The designs occupy a tight patch next to the
  starting point.

  > The talk quotes "120 to 190" on slide 23; the p1–p99 range the pipeline
  > reports is 81–196. Use the figure's own numbers — the conclusion is
  > unchanged and if anything understated.
* **They close at most ~5 % of the gap to chymotrypsin.** `si07_construct_context`
  walks the Hedstrom construct ladder and shows the same thing for the
  constructs that were actually built and assayed:

  | construct | P(S) | → train | → test | gap closed |
  |---|---|---|---|---|
  | wild-type trypsin | 0.0005 | 0.0 % | 19.6 % | 0.0 % |
  | + loops 185–188, 221–225 | 0.0035 | 4.5 % | 23.8 % | 1.2 % |
  | + 172 | 0.0052 | 4.9 % | 24.2 % | 0.0 % |
  | + 216, 226 | 0.0504 | 5.3 % | 24.5 % | −1.2 % |

  **The real constructs never leave the trypsins either**, which is the honest
  frame for our own designs.
* **A high P(S) is a rewired neighbourhood, not a new enzyme.**
* **Fewer columns means a smaller library.**
* **Nothing here has been measured in a lab.**

### Would a different bias change the picks?

`si04_logodds_rerank` re-ranks the pool by log-odds instead of P(S): same
ranking, different best hundred. Steering *directly* on log-odds was also tried
and does not work — it turns the pocket into glycine. See
[`CLAMP_CONTROL.md`](CLAMP_CONTROL.md) §2.4 for why that is a property of a
gameable objective rather than of the model.

### Are these equilibrium results?

`si06_equilibration` traces every run and checks drift over the second half.
All five conditions are traced and every observable is flat by the end.

> ⚠ **Read this figure before trusting a cold rung.** It was silently rendering
> empty placeholders until November 2026 (it resolved traces through an absolute
> path baked into `summary.json`, which the repository reorganisation
> invalidated). It now reads the 54 traces that were on disk all along. The
> worst traced rung moves ~20 % IQR in P(S) — small, but not zero. Acceptance
> also falls steeply with β (0.029–0.098 at β=1, 0.008–0.027 at β=2,
> 0.0015–0.0057 at β=10), so on the coldest rungs "converged" and "stuck" are
> worth distinguishing before a result leans on them.

## 7. The talk, slide by slide

Every slide carrying data, and the script that makes its figure. Verified by
matching the embedded image dimensions in the PDF against `figures/ladder/`.

| slide | title | figure |
|---|---|---|
| 6 | The model tells the two families apart | `fig01_discriminator` |
| 7 | Turning up the bias moves everything at once | `fig02_corner_grafting` |
| 8 | Cooling makes better designs, but fewer of them | `fig03_cooling` |
| 9 | Restricting the sites is what concentrates them | `fig08_sector_enrichment` |
| 11 | The candidate sets hardly agree | `fig04_set_overlap` |
| 12 | The loops are left alone | `si03_gaps` |
| 14 | The runs we kept | `fig05_table` |
| 15 | Three families, three islands | `fig06_corner_selected` |
| 17 | The designs stay inside the family | `fig07_pca` |
| 20 | S1 · the signal survives on unseen sequences | `si01_controls` |
| 21 | S2 · every run we sampled | `si02_table_full` |
| 22 | S4 · would a different bias change the picks? | `si04_logodds_rerank` |
| 23 | S5 · designs against the whole family | `si05_naturals_vs_designs` |
| 24 | S6 · the chains reached equilibrium | `si06_equilibration` |
| 25 | S7 · what high P(S) does and does not mean | `si07_construct_context` |

Slides 1–5, 10, 13, 16, 18–19 are title, schematic and summary slides built in
Keynote; they carry no generated figure. The briefs that specify them are in
[`presentation/`](presentation/).

Rebuild any single figure with, e.g.:

```bash
cd analysis/figures_ladder && ../../.venv/bin/python make_all.py fig07
CAVEATS=1 ../../.venv/bin/python make_all.py     # with the caveat text under each figure
```

## 8. Reproducing from scratch

Stage 1 samples (expensive, GPU), Stage 2 draws (cheap). The stages of
`run_ladder.sh` are **not independent** — `derive` reads this model's consensus
columns out of the cooled grafting run, so `consensus` cannot be sampled until
`graft` and `derive` have run.

```bash
bash analysis/run_ladder.sh          # graft, derive, consensus, derive_large, consensus_large (~1.5 h)
bash analysis/run_ladder_sectors.sh  # the all-sites-free run + sector annotation
cd analysis/figures_ladder && ../../.venv/bin/python make_all.py
```

See [`DATA.md`](DATA.md) for what must be present before Stage 1 will start, and
the caveat there about the designed ensembles being gitignored.
