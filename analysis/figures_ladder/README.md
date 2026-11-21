# Figure pipeline — trypsin → chymotrypsin specificity switch

Every figure for the talk, in slide order, built from one model and one style.
Output goes to `figures/ladder/` as 300 dpi PNG and vector PDF.

```bash
cd analysis/figures_ladder
../../.venv/bin/python make_all.py            # every figure, ~12 s
../../.venv/bin/python make_all.py fig07      # one figure
```

## The model

| | `serine_protease` |
|---|---|
| parameters | `$TRYP_DATA_ROOT/DCAmodel-serine_protease/params.dat.gz` |
| trained on | the 71,551-sequence **train split** |
| held-out set | 17,888 sequences — F1/S1 are **predictive** |
| μ | 422.62 |
| wild-type E / CDE | 82.9 / 0.445 |
| results | `results/ladder/` |
| figures | `figures/ladder/` |

The parameter file is far too large for git and lives outside the repository,
under `TRYP_DATA_ROOT`, which defaults to the repository's parent directory —
see `docs/DATA.md`. An earlier model, `pcd_v2`, was dropped in the Nov 2026
cleanup together with its results; `config.MODELS` is still a dict so a future
model can be added beside this one rather than replacing it in place.

**Nothing model-derived is portable between models** — which is why each one
gets its own results tree and its own figure directory. μ, the λ and β ladders,
the coupling ranking and the consensus column set are all properties of the model
that produced them, and each is recomputed per model (`config._COUPLING`,
`config.consensus_columns()`). The red sector is *not* model-derived — it is a
column list that ships with the alignment — so it carries over unchanged.

---

## The two stages — read this before swapping the model

The pipeline has an expensive half and a cheap half, and **they are separate on
purpose**.

| | what | cost | when |
|---|---|---|---|
| **STAGE 1** | `analysis/run_ladder.sh` (+ `analysis/run_ladder_sectors.sh`) | ~1.5 h, GPU | re-run **first** for a new model |
| **STAGE 2** | `make_all.py` (these scripts) | ~12 s | re-run after Stage 1 |

`run_ladder.sh` runs its stages in order and they are **not independent**:

```bash
bash analysis/run_ladder.sh graft            # A: grafting beta=1, B: grafting + cooling
bash analysis/run_ladder.sh derive           # read THIS model's consensus from its cold run
bash analysis/run_ladder.sh consensus        # C: the tight consensus set, beta=1
bash analysis/run_ladder.sh derive_large     # the wider consensus set, from the warm lambda=1000 run
bash analysis/run_ladder.sh consensus_large  # D: the wider set, beta=1 and cooled
bash analysis/run_ladder_sectors.sh          # the all-sites-free run + sector annotation (F8)
```

`derive` must run between them: the consensus columns are read from the cooled
grafting run, so conditions C and D cannot be sampled until B exists.

**The failure mode this guards against:** point `config.MODEL_PARAMS` at a new
model, run Stage 2 only, and you get figures in which the *new* model's scores
are drawn over the *old* model's designed sequences. Nothing looks wrong. Every
number is plausible. Every number is meaningless.

So `config.check_run()` stamps each ensemble directory with a fingerprint of the
parameter file it was sampled from, and **refuses to build** if that no longer
matches. The fingerprint is size + sha256 of the first and last 1 MB —
deliberately not mtime, because copying a file changes mtime without changing
the model, and a check that cries wolf gets switched off.

### Swapping in a new model

1. Edit `MODEL_PARAMS` in `config.py`. Nothing else hard-codes a path.
2. Re-run **Stage 1**. The sampling scripts write fresh ensembles.
3. Re-run `make_all.py`.

Caches (`.cache/`) are keyed by the fingerprint, so they miss cleanly and
recompute rather than silently serving the old model's numbers.

---

## Module layout

| file | holds |
|---|---|
| `config.py` | every path, the critic column, the site sets, the four conditions, the fingerprint contract |
| `style.py` | the palette, rcParams, the notation strings, `save()` |
| `data.py` | loaders, the critic conditional, per-run metrics, diversity |
| `selection.py` | the "best 2" rule |
| `metrics.py` | ROC / PR / AUC / average precision |
| `fig0*.py`, `si0*.py` | one figure each, standalone and importable |
| `make_all.py` | runs them in talk order |

**No script defines a colour or a path of its own.** That is what makes the set
read as one piece of work.

### Notation, used identically everywhere

- the critic site is **i\*** — 3TGI residue 189, MSA column 201
- **P(S | σ_\i\*)** and **P(D | σ_\i\*)** — the conditional at i\* given the rest
- the log-odds is **log P(S | σ_\i\*) / P(D | σ_\i\*)**
- the measured literature construct is always **"1992 construct"**, never
  "Hedström 17" — D189S plus loops 185–188 and 221–225 (Hedstrom, Szilagyi &
  Rutter, *Science* 1992, 255:1249–1253)

---

## The logical flow, figure by figure

### F1 — `fig01_discriminator` · why P(S | σ_\i\*) is the right score

**Shows.** (a, b) the two conditionals split by the residue each sequence
actually carries at i\*; (c) ROC and (d) precision-recall for chymotrypsins
against trypsins.

**With the serine-protease model this is a HELD-OUT result**, evaluated on the
17,888 sequences excluded from training: **AUC 0.960 / 0.983 / 0.983** and
**AP 0.875 / 0.923 / 0.938** for P(S), −P(D) and the log-odds. The training
curve is drawn thin and dashed beside it (log-odds AUC 0.997, AP 0.989) so the
**generalisation gap is visible rather than asserted**. It is real — some of the
earlier single-number claim was memorisation — but 0.983 on unseen sequences is
a prediction, not a consistency check.

The split is matched **by sequence, not by name**: the split FASTAs were written
with renamed headers and the alignment's own names are not unique (85,069
distinct over 89,439 rows), so name matching misses 328 test sequences and is
ambiguous where it hits. All 89,439 sequences are distinct, so sequence matching
is exact — 17,888 of 17,888.

**Say.** The conditional at i\* does not depend on the residue at i\*, so scoring
a natural sequence is a leave-one-site-out test: the model reads a sequence's
specificity off its context alone. The 1992 series climbs monotonically
(0.005 → 0.050 → 0.071 → 0.215) but never leaves the trypsin bulk — grafting
alone does not flip the conditional.

**Objection.** *"Isn't that circular?"* With the serine-protease model, no — the
evaluation set was excluded from training. The follow-up — *"fine, but is it
better than just looking up the nearest known sequence?"* — is **S1**.

### F2 — `fig02_corner_grafting` · what the bias buys and what it costs

**Shows.** 5×5 corner over the grafting segment at β=1, all six λ together.

**Say.** λ moves every observable together: P(S) 0.01 → 0.97, Hamming to WT
19 → 40, E_DCA 175 → 255, distance to the nearest natural S-carrier 33% → 46%.
**Q rises from −96 to −86 — i.e. gets worse.** The switch is bought with energy,
mutational load and distance from the natural manifold.

**Objection.** *"So the designs are worse than the wild type?"* At β=1, yes, on
Q. That is exactly what F3 fixes.

### F3 — `fig03_cooling` · cooling buys plausibility

**Shows.** The β / λ̃ scan, regenerated in the shared style from
`cooling/beta_scan_summary.csv`.

**Say.** Cooling costs essentially nothing in P(S) (every β reaches ~0.95–0.97)
and buys a lot in Q: **no β=1 run reaches Q(wild type) = −102.97**; the bar is
first cleared at β=2. The price is diversity — at β=10, λ̃=100 only 142 of 1000
chains are distinct.

**Objection.** *"A library of 142 distinct sequences isn't a library."* Correct,
and it is why the selection rule tie-breaks on diversity.

### F4 — `fig04_set_overlap` · the candidate site sets barely agree

**Shows.** Jaccard and containment between the five sets, with the overlap a
random set of the same size would give.

**Say.** No set contains another. Only **column 235 (3TGI 221)** is common to all
four small sets. The 1992 construct is the most isolated: 4 shared columns with
the red sector against 2.8 expected by chance — essentially no agreement. The
model-derived sets agree with each other more than any agrees with what was
actually built.

### F5 — `fig05_table` · the selected designs

**Shows.** A hand-picked handful of runs, with **the best value in every column
in bold** (↑ / ↓ in the header says which direction is better).

**Say.** There is no scoring rule. Five desiderata pull against each other and
any single formula would bury the trade-off under arbitrary weights, so the rows
are chosen by eye from the full pool (S2) and the bolding shows what each one
wins. **No run wins more than two columns** — which is the point. Reading across:
consensus-19 is the cheapest to build (19 sites, 18.5 mutations), cooling at
β=10 is the most plausible (Q = −115.6), grafting at λ=1000 has the highest P(S)
(0.971), grafting at λ=300 keeps the widest library (pairwise 31.2).

**Columns worth explaining.** `mean Q_μ` is plausibility, lower is better, and
the wild type sits at −103.0. **`pairwise / Hamming`** is library spread
normalised by distance travelled — raw pairwise distance is not comparable
across runs that moved different distances from the wild type, since 3 mutations
apart is tight at 20 mutations out and scattered at 4. **~1 means chains differ
from each other about as much as they differ from the wild type; ~0 means the
library has collapsed to a single sequence.** It ranges from 1.06 (grafting,
λ=10) down to 0.12 (cooling, β=10) and is the clearest single read on collapse.

**To change the selection:** edit `selection.SELECTED`. Nothing else needs
touching; a run named there that does not exist fails loudly and lists what is
available.

### F6 — `fig06_corner_selected` · the selected designs side by side

**Shows.** 4×4 corner over the selected runs, with the wild-type Q as a dashed
reference on the Q axes.

**Say.** The three clusters are the three strategies. Cooling (orange) is tight
and plausible; consensus-19 (green) is nearly as plausible with a third of the
mutations and far more diversity; grafting at β=1 (blue) is far out in both
mutation count and Q.

### F7 — `fig07_pca` · designs and naturals in one projection

**Shows.** Two components of the natural alignment in one-hot space, everything
else projected in. (a) naturals + wild type + 1992 series; (b) naturals +
selected designs. Marginals on both components.

**Say.** PC1 separates trypsins from chymotrypsins. The designs sit in a tight
cluster near the wild type, **not** inside the chymotrypsin cloud — they are
trypsins carrying a chymotrypsin-like S1 site, which is what was asked for, but
it is worth saying out loud.

**Objection.** *"PC1+PC2 are only 3.8% of the variance."* True, and stated on the
axes. The projection is a view, not a summary.

### F8 — `fig08_sector_enrichment` · the red sector is targeted, not diluted

**Shows.** Red-sector enrichment vs λ, grafting segment against all-sites-free.

**Say.** Unbiased ensembles **avoid** the red sector (ratio 0.65 / 0.70). Once
λ ≥ 10 both put ~1.2–1.3× more mutations on it than its size predicts — and
freeing every site does not dilute that.

**Objection.** *"The two curves have different denominators."* Yes — red is 20.3%
of mutable sites under grafting and 6.4% when everything is free. Enrichment is
dimensionless so the **shapes** are comparable; the magnitudes are not. Caveated
on the figure.

---

## Supplementary

### S1 — `si01_controls` · how much of F1 is memorisation

F1 answers *"is it circular?"*. This answers the follow-up: *"is it better than
looking up the nearest known sequence?"* All of it on the **held-out** split.

| held-out score | AUC | AP |
|---|---|---|
| Potts P(S \| σ_\i\*) | 0.959 | 0.875 |
| **Potts log-odds** | **0.983** | 0.935 |
| 1-NN lookup | **0.983** | **0.953** |
| class profile, no couplings | 0.965 | 0.888 |

- **The couplings buy something real.** The log-odds beats the no-epistasis
  class profile by +0.018 AUC. That is the quantity to quote for "the model uses
  epistasis", and it is modest.
- **It ties explicit lookup** on AUC and loses slightly on AP — and both
  baselines are handicapped *in their favour*: they read the class labels and may
  use the model's training sequences, while the conditional reads neither.
- **It does not degrade on novel sequences.** Held-out sequences sit a median
  29.8 % of L from anything in training. In the two populated novelty bins the
  log-odds scores 0.993 (10–25 %) and 0.977 (> 25 %), against 1-NN's 0.991 and
  0.977. The near-duplicate bins are empty and are dropped rather than drawn as
  `nan`.
- Fair summary: *the task is easy for anything with access to the alignment; the
  conditional matches explicit lookup on sequences it has never seen, without
  reading any labels, and beats a no-epistasis baseline.*

**Score the log-odds, not P(S).** An earlier version compared the baselines
against P(S) alone — the weaker of the model's two discriminants — which made
the model look beaten (0.959 vs 0.983) when the right comparison is a tie.

### S2 — `si02_table_full` · the pool behind F5
All 22 sampled runs in scan order, best-in-column bold over the whole pool. This
is the table to read when re-choosing the selection: down a column shows what
the scan can reach on that axis, across a row shows what one setting costs on
the others. The column winners are spread over seven different runs, which is
the concrete reason a single score would have been misleading.

### S3 — `si03_gaps` · gap drift by condition
Wild type has 42 gap columns of 265. At β=1 strong bias **fills** them
(mean 41.8 → 36.9 at λ=1000). Cooling reverses it: at β≥5 essentially every
chain sits at exactly 42. consensus-19 barely moves (41.7 → 42.0).

### S5 — `si05_naturals_vs_designs` · the designs against the whole family
The comparison F2 cannot make. Natural sequences sit **20–214 mutations** from rat
trypsin while the designs sit at **12–51**, so on F2's axes only 0.1% of naturals
are visible and **no natural chymotrypsin at all** — which is why the naturals are
here and not there. With the axes opened up (1st–99th percentile of the naturals)
the designs become a tight cluster hard against the wild type, and that is the
result: **the designs are a neighbourhood of the wild type, not a walk across the
family.** The literature series sits in the same corner.

Two things to say when showing it. `Hamming to wild type` is a *design*
coordinate, so a natural sequence's position on it measures family distance, not
quality — F7 is the proper family comparison. And **in this figure alone colour
means natural class**, so all designs are one colour and the condition is carried
by marker shape; everywhere else colour means condition.

### S4 — `si04_logodds_rerank` · should the bias use the log-odds?
The sampler steers on `log P(S | σ_\i*)` (`src/ensemble.py`). **The two scores
disagree substantially about which chains are best: median top-100 overlap 36%,
median Spearman 0.53.** Panel (a) shows why — the relationship is tight at low
P(S) and fans out as P(S) → 1, exactly where the designs live. When P(S)
saturates, `log P(S)` loses dynamic range while the log-odds keeps resolving.

This is evidence **for** testing a log-odds bias with the next model. It was not
changed here because switching invalidates every calibrated number (μ, the λ
grid, the 0.80 gate) at once.

---

### S6 — `si06_equilibration` · did the chains equilibrate?
Four observables (E_DCA, Hamming to wild type, Q_μ inverted so up is better, and
P(S)) against sampling step, per condition. Line = ensemble mean over 1000
chains, band = interquartile range *across chains*, so it shows the ensemble's
spread rather than the error on the mean.

Every traced run is flat well before the end. **Drift is measured as the movement
of the mean over the final doubling of steps, as a percentage of the ensemble's
own interquartile spread** — not as a percentage of the observable's range, which
is the wrong yardstick for a near-constant series (the unbiased λ=0 run moves
P(S) over a total range of 0.006, so a drift of 0.0007 reads as "10% of range"
and looks like a failure). The largest drift anywhere is 10.8% of one IQR, which
in absolute terms is 0.0007 in P(S). The worst absolute drifts are 0.28 in
E_DCA, 0.07 in Hamming and 0.3 in Q.

Q carries no band and is scaled by range instead: it is linear in E and CDE so
its *mean* is exact from theirs, but the trace stores only per-observable
quartiles, and the quartile of a sum is not the sum of the quartiles.

### S7 — `si07_construct_context` · why the 1992 constructs score so low
The obvious explanation — the model has never seen anything like them — is
**wrong**, and the figure rules it out before giving the real one.

(a) The split is balanced: 17% of trypsins, 20% of chymotrypsins and 29% of the
rest are held out, so neither class is under-represented.

(b) The constructs are **not** unfamiliar. Each sits 4.5–5.3% of L from the
nearest *training* sequence — and that neighbour is a trypsin. The wild type is
itself a training sequence (distance 0).

(c) The explanation. The wild type sits **82 residues** from the nearest natural
chymotrypsin. The 1992 constructs close **none** of that gap (−1.2% to +1.2%);
12–14 grafted mutations against an 82-residue separation. So the model scores
them as trypsins because, by sequence, they *are* trypsins.

**The same panel indicts our own designs.** They close at most ~5% of that gap.
`P(S | σ_\i*)` is high because the LOCAL context at i\* has been rewired, not
because the protein has moved toward the chymotrypsin family. State this
explicitly whenever a P(S) number is shown: it is what the quantity does and
does not say.

## Gaps are proposed as moves — read every mutation count with this in mind

Every run passes `--allow-gap-moves`, so the gap is an ordinary state: the
sampler may **delete** a wild-type residue or **fill** a wild-type gap. Those are
indels, not point mutations, and an experimentalist cannot build them the same
way. F5 and S2 therefore split the count into `mean subst.`, `mean indels` and
`% chains indel-free`.

**The aggregate hides it; the per-chain figure does not.** Indels are 0–5 % of
all *mutations* but affect a large share of *chains*:

| design | % chains carrying an indel |
|---|---:|
| grafting β=1, λ=300 | **60 %** |
| consensus-22 β=1, λ=300 | **44 %** |
| grafting β=1, λ=30 | 27 % |
| cooled runs at λ̃=30 | **0.1 %** |

This is a real argument for the cooled runs that the old single-Hamming column
could not make: they are essentially **pure substitution libraries**.

**It also reaches the consensus sets.** A column qualifies as "rewritten in ≥ X %
of chains" whether the change was a substitution or an indel, so
`derive_consensus.py` now prints the composition of every chosen column and
warns about any that qualify mainly on indel moves.

- **consensus-8 is clean**: all eight columns are 100 % substitution-driven.
- **consensus-22 contains one column that is not a substitution site at all** —
  **column 231**, 60.7 % rewritten, **0 % substitution, 60.7 % insertion**: the
  wild type has a *gap* there and the sampler fills it. Two more lean partly on
  deletions (204 at 37 %, 238 at 14 %).

So consensus-22 cannot be handed over as a list of 22 positions to mutate
without saying that 231 is an insertion and 204/238 are partly deletions.

## Known gaps — state these before someone finds them

1. **The 1992 construct is reconstructed** from the column positions the
   literature implicates, grafted from rat chymotrypsin — not the sequences
   reported in the paper. Labelled as such wherever it is plotted.
2. **The red sector has 17 columns, not 16.** Column 2 (3TGI residue 18)
   lies outside the grafting segment, so `config.site_sets()["red sector"]` is
   the red sector ∩ segment — which is what the design runs actually allowed.
   The all-sites-free sector analysis scores all 17, so F8's two curves have
   different red denominators.
3. **The palette validator was NOT run** — `node` is not available here.
   The categorical slots are the documented reference palette, whose first three
   are validated for all-pairs use (scatter, PCA). F6/F7/S4 use four condition
   colours, one more than that, so marker shape and direct labels carry identity
   as well as colour. Re-run
   `node <dataviz-skill>/scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#6250d6" --mode light`
   if you get the chance.
4. **`summary.json`'s `mean_P_S` is NOT the conditional.** It is the
   fraction of chains carrying S at i*, which is 1.0 for every cooled run. The
   quantity every figure uses is the β = 1 conditional recomputed from the
   sequences by `data.run_frame` — at β = 5, λ̃ = 10 that is **0.756**, not 1.0.
   Read the ladders from the recomputed value, never from the stored summary.
5. **The step count is measured, not assumed.** Saturation of E,
   Hamming, Q and P(S) puts the slowest warm case (λ = 0, acceptance 0.029) at
   ~8,500 steps and the slowest cold case (β = 5, acceptance 0.0025) at ~11,400.
   Production uses 60,000 — a > 5× margin — and S6 re-checks every run.
