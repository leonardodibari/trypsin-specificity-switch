# Trypsin → chymotrypsin specificity switch by biased MCMC

Redesigning the substrate pocket of a serine protease *in silico*: this repository
samples protein sequences that start from wild-type rat trypsin and drift toward a
chymotrypsin-like specificity, under a Potts (bmDCA) model of the serine-protease
family trained with [adabmDCA 2.1](https://github.com/spqb/adabmDCApy).

A formal write-up of the method, with the equations and the command behind every
number, is in [`docs/METHODS.md`](docs/METHODS.md).

It answers two different questions with the same machinery:

* **One trajectory at a time** — what route does a chain take from the wild type to
  a switched pocket, and what does it pay along the way?
* **A thousand chains at once** — what does the biased *distribution* of sequences
  look like, and how do its designs compare with natural sequences of the family?

---

## Contents

1. [The problem, in one page](#the-problem-in-one-page)
2. [How the sampler works](#how-the-sampler-works)
3. [Installation](#installation)
4. [What you need to supply](#what-you-need-to-supply)
5. [Quick start](#quick-start)
6. [The scripts](#the-scripts)
7. [Specifying sites](#specifying-sites)
8. [Sample quality, and the calibration of mu](#sample-quality-and-the-calibration-of-mu)
9. [Results from a real run](#results-from-a-real-run)
10. [Conventions and gotchas](#conventions-and-gotchas)
11. [Repository layout](#repository-layout)
12. [Implementation notes](#implementation-notes)

---

## The problem, in one page

Trypsin and chymotrypsin are serine proteases with nearly the same fold and
catalytic machinery, but they cut different substrates. Trypsin cleaves after
positively charged residues; chymotrypsin after bulky hydrophobic ones. Much of
that difference is decided by a handful of residues lining the **S1 specificity
pocket**, and above all by one position — residue **189** in the standard
chymotrypsinogen (PDB 3TGI) numbering. Trypsin carries an aspartate there, whose
negative charge attracts lysine and arginine side chains; chymotrypsin carries a
serine, and the pocket becomes hydrophobic.

So "switching specificity" has a concrete sequence-level proxy: **make position
189 want to be a serine instead of an aspartate**, by changing the *other* residues
around it.

A Direct Coupling Analysis (DCA) model gives us a way to ask whether a sequence
"wants" that. Trained on a large alignment of the serine-protease family, a Potts
model assigns every sequence a statistical energy

```
E_DCA(seq) = - sum_i h_i(s_i) - 0.5 * sum_{i,j} J_ij(s_i, s_j)
```

where low energy means "typical of the family". From the same parameters one can
read off, for any site `i`, the conditional distribution of what residue belongs
there *given everything else in the sequence*:

```
P(a | site i, rest of the sequence) = softmax( h_i + sum_{j != i} J_ij(., s_j) )
```

Evaluated at the critic site, `P(S | 189, seq)` is the model's answer to "given the
rest of this protein, how chymotrypsin-like is this pocket?" For wild-type rat
trypsin the model is emphatic: `P(D) = 0.979` against `P(S) = 0.008`.

**The task this repository automates:** mutate a limited set of pocket positions
until `P(S | 189)` is high, while keeping the sequence a plausible member of the
family. Those two goals pull against each other, and the trade-off between them is
the object of study.

---

## How the sampler works

Sequences are sampled by Metropolis–Hastings under a modified Hamiltonian:

```
H(seq) = beta * E_DCA(seq) - lambda * log P(S | critic_site, seq)
```

* **`E_DCA`** is the Potts energy above — the "stay plausible" term.
* **`log P(S | critic_site)`** is the "switch the pocket" term.
* **`lambda` is the bias strength.** The larger it is, the harder the chain is
  pulled toward serine at the critic site, and the more DCA energy it is willing to
  pay. `lambda = 0` is plain Potts sampling, with no design pressure at all.
* **`beta`** is an inverse temperature. It scales the energy *and* sharpens the
  conditional, so `P` is always the conditional of `exp(-beta E)`.

A move substitutes one residue at one allowed site with another non-gap residue,
drawn uniformly. The proposal is therefore symmetric and the acceptance is simply
`min(1, exp(-dH))`.

Two properties of this construction are worth knowing, because the code relies on
both:

* `P(S | critic_site, seq)` conditions on *the rest of the sequence*, so it does not
  depend on the residue currently sitting at the critic site. A move at the critic
  site changes `E_DCA` only. (The pocket is switched by its context, not by
  stamping an S into position 189.)
* Alignment gaps are inert: a gap column is never mutated, and a gap is never
  proposed as a replacement residue.

The two sampling modes differ only in what they report:

| Mode | Script | Stops when | Reports |
|---|---|---|---|
| Single chain | `run_mcmc.py`, `run_sweep.py` | `P(S) >= threshold` (default 0.80) or `--max-steps` | the whole trajectory |
| Ensemble | `run_ensemble.py` | after `--n-steps`, no stopping rule | the final sequence of every chain |

---

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/leonardodibari/trypsin-specificity-switch.git
cd trypsin_specificity_switch
pip install -e .
```

That pulls in `adabmDCA` (version 1.0.0 on PyPI — this is the adabmDCA 2.1
software), plus torch, numpy, pandas, matplotlib and joblib.

To run the test suite (26 tests, about 10 seconds, no model file needed — it builds
a small random Potts model on the fly):

```bash
pip install -e ".[dev]"
pytest
```

A GPU is optional. Everything runs on CPU; `--device cuda` moves the ensemble
sampler onto a GPU when one is available, which is worth it for large batches.

---

## What you need to supply

| Input | Flag | Notes |
|---|---|---|
| **Potts model parameters** | `--model` | The `h`/`J` text file written by adabmDCA (or a directory containing exactly one). **Not included** — these files run to hundreds of megabytes. |
| **Wild-type sequence** | `--wt` | FASTA holding the *aligned* (gapped) row, same length `L` as the model. Included: `data/wt_trypsin.fasta`. |
| **Natural alignment** | `--natural` | Only for `calibrate_mu.py`. The family alignment the model was trained on. **Not included.** |
| **Residue-number mapping** | `--mapping` | *Optional.* CSV mapping PDB 3TGI residue labels to MSA columns. See [Specifying sites](#specifying-sites). |

What *is* included, in `data/`, are three aligned sequences of length 260 taken
from the study alignment:

* `wt_trypsin.fasta` — rat anionic trypsin (TRY2_RAT), the starting point;
* `chymotrypsin_bovine.fasta`, `chymotrypsin_rat.fasta` — chymotrypsins, used as
  donors when building a chimeric "target" sequence for reference.

The `.gitignore` excludes `*.dat` and `*.cache.pt` so a large parameter file cannot
be committed by accident. Keep it outside the repository and point `--model` at it.

**Model loading cache.** Parsing a real parameter file takes about 30 seconds. Pass
`--cache-model` and the loaded tensors are dumped next to the parameter file as
`<params>.cache.pt`, cutting subsequent loads to a few seconds. The cache is
invalidated automatically if the text file is newer.

---

## Quick start

With a parameter file at `../bmDCA_params.dat`, first check that the sites resolve
as you expect — this loads the model and exits without sampling:

```bash
python scripts/run_mcmc.py \
    --wt data/wt_trypsin.fasta \
    --model ../bmDCA_params.dat \
    --output /tmp/out \
    --critic-col 197 --allowed-cols 180-220 \
    --list-sites --cache-model
```

It prints the model dimensions, the critic site with the wild-type residue found
there, and the mutable columns. Then run one chain:

```bash
python scripts/run_mcmc.py \
    --wt data/wt_trypsin.fasta \
    --model ../bmDCA_params.dat \
    --output results/ \
    --critic-col 197 --allowed-cols 180-220 \
    --lam 50 --threshold 0.80 \
    --max-steps 200000 --record-every 50 --seed 42 \
    --cache-model
```

---

## The scripts

All scripts live in `scripts/` and share a common set of options (`--wt`,
`--model`, `--beta`, `--alphabet`, `--device`, `--cache-model`, …). Every one of
them responds to `--help`.

### `run_mcmc.py` — one chain, recorded step by step

Runs a single chain and records its state every `--record-every` accepted moves,
stopping when `P(S)` first reaches `--threshold` or when `--max-steps` proposals
have been made.

Writes to `<output>/lam<value>/`:

| File | Contents |
|---|---|
| `trajectory.csv` | one row per recorded step: `step, msa_col_changed, aa_from, aa_to, E_DCA, logP_S, H_eff` (step 0 is the wild type) |
| `sequences.fasta` | the recorded sequences, headers `>step_{n}`; `--ungapped-fasta` strips alignment gaps |
| `summary.json` | settings, acceptance rate, the step at which the threshold was first crossed, final state |

### `run_sweep.py` — many chains over many lambdas, in parallel

Same chains, run for a list of `--lambdas` with `--n-runs` independent replicates
each (replicate `i` uses seed `--seed + i`), distributed over cores with joblib
(`--n-jobs`). Each `(lambda, run)` pair lands in
`<output>/lam<value>/run<i>/` with the three files above, and
`<output>/sweep_summary.csv` collects one row per chain.

```bash
python scripts/run_sweep.py \
    --wt data/wt_trypsin.fasta --model ../bmDCA_params.dat \
    --output results/ --critic-col 197 --allowed-cols 180-220 \
    --lambdas 10 20 50 --threshold 0.80 \
    --max-steps 200000 --record-every 25 \
    --n-runs 4 --seed 1 --n-jobs 12 --cache-model
```

Each worker loads the model once and runs torch single-threaded by default
(`--torch-threads`), which is both faster and far cheaper in CPU time than letting
every chain try to use every core.

### `plot_trajectories.py` — the trajectory figure

Produces a 4 × 3 figure, one column per lambda, from a sweep's output:

1. `P(S | critic site)` against accepted step — each chain a thin line, with the
   median and its inter-quartile band, the threshold as a dashed rule, and a star
   where a chain first crosses it;
2. `E_DCA` against accepted step;
3. `E_DCA` against `P(S)`, one point per recorded step, with the wild type and a
   chimeric target marked;
4. `E_DCA - plot_lambda * log P(S)` against accepted step — the effective energy
   evaluated at **one chosen lambda for every column**, independent of the lambda
   each chain was sampled at, so differently biased chains are comparable on a
   single scale.

```bash
python scripts/plot_trajectories.py \
    --output-dir results/ --lambdas 10 20 50 --plot-lambda 20 \
    --critic-col 197 --wt data/wt_trypsin.fasta --model ../bmDCA_params.dat \
    --target-source data/chymotrypsin_bovine.fasta --allowed-cols 180-220 \
    --cache-model --save trajectories.png
```

The reference target is either supplied ready-made with `--target` or built with
`--target-source`, which copies the chymotrypsin residues onto the wild type at the
allowed positions, skipping gaps. The median line is drawn only while at least half
the chains are still running, so a lone survivor is never dressed up as a median.

### `run_ensemble.py` — a thousand chains per lambda

Where a single chain shows a route, an ensemble shows the distribution. Every chain
starts from the wild type, runs `--n-steps` proposed moves and contributes its
**final** sequence. The chains advance together as one tensor — one site is drawn
for the whole batch and scored with a single matrix product — so a thousand chains
cost barely more than one, and `--device cuda` puts the batch on a GPU.

```bash
python scripts/run_ensemble.py \
    --wt data/wt_trypsin.fasta --model ../bmDCA_params.dat \
    --output ensembles/ --critic-col 197 --allowed-cols 180-220 \
    --lambdas 0 10 50 100 300 1000 --n-chains 1000 --n-steps 60000 \
    --trace-every 1000 --trace-schedule log --trace-lambdas 10 100 1000 \
    --device cuda --cache-model
```

Writes, per lambda, to `<output>/lam<value>/`:

| File | Contents |
|---|---|
| `final_sequences.fasta` | the final sequence of every chain, aligned, one record per chain |
| `observables.csv` | `E_DCA`, `hamming_to_wt`, `P_S` and `CDE` for each chain's final sequence |
| `summary.json` | settings, acceptance rate, batch means |
| `trace.csv` | the four observables (mean and quartiles across chains) and the mean accepted moves per chain, at each recorded step |

Tracing costs a full all-site pass per recorded point, so `--trace-lambdas` limits
it to the two or three lambdas worth watching. `--trace-schedule log` spaces the
same number of points logarithmically, which is what resolves the approach to
equilibrium — nearly everything happens within the first hundred accepted moves.
Recording never touches the random number stream, so a traced run and an untraced
one with the same seed produce identical chains.

### `calibrate_mu.py` — fix the sample-quality weight on natural data

See [the next section](#sample-quality-and-the-calibration-of-mu). Writes
`mu.json` (read directly by the plotting scripts), `natural_observables.csv` and a
calibration figure.

```bash
python scripts/calibrate_mu.py \
    --natural path/to/family_alignment.faa --model ../bmDCA_params.dat \
    --output calibration/ --max-sequences 20000 \
    --wt data/wt_trypsin.fasta --critic-col 197 --cache-model
```

### `plot_ensemble.py` — the pairwise scatter matrix

Crosses every pair of the four per-sequence quantities — energy, Hamming distance
to the wild type, `P(S | critic site)` and sample quality — as a lower-triangular
scatter matrix with the marginal distributions on the diagonal. Each lambda gets
its own shade along a single hue (lambda is an ordered magnitude, not a category),
and the natural alignment, if supplied, sits behind as a grey cloud.

```bash
python scripts/plot_ensemble.py \
    --ensemble-dir ensembles/ --mu calibration/mu.json \
    --natural-observables calibration/natural_observables.csv \
    --max-points 1000 --save pairs.png
```

`--point-size`, `--natural-point-size` and `--font-scale` control the marks and
type; `--quantities E_DCA P_S Q` drops the Hamming column, whose scale is dominated
by the family's distance from any single sequence.

### `substitute_critic.py` — write the serine in, and rescore

The bias acts on the *conditional* at the critic site, which does not depend on the
residue sitting there, so a chain's final sequence still carries whatever residue
it last accepted — at `lambda = 0`, almost always the wild-type aspartate. This
script sets the critic column to `S` in every final sequence of every lambda and
recomputes the four observables, so that a figure drawn from them describes
sequences that really are serine at the pocket. Only the sampled sequences are
touched; the natural alignment and the calibration of `mu` are left alone.

```bash
python scripts/substitute_critic.py \
    --ensemble-dir ../results/ensembles --output ../results_serine/ensembles \
    --wt data/wt_trypsin.fasta --model ../bmDCA_params.dat \
    --critic-col 197 --device cuda --cache-model
```

It writes the same `final_sequences.fasta` / `observables.csv` / `summary.json`
layout as `run_ensemble.py`, so every plotting script reads the result directly.
`observables.csv` gains one column, `critic_residue_before`, holding the residue
that was replaced.

### `plot_critic_deltas.py` — what that substitution costs

Two panels of histograms, one per lambda: the per-chain change in energy and in
sample quality between a chain's own final sequence and the same sequence with the
serine written in. Chains that already ended on `S` differ by exactly zero and are
left out, with their share in the legend; `--include-unchanged` puts them back.

```bash
python scripts/plot_critic_deltas.py \
    --ensemble-dir ../results/ensembles --substituted-dir ../results_serine/ensembles \
    --mu ../results/calibration/mu.json --save docs/critic_substitution_deltas.png
```

### `add_natural_neighbour.py` — how close is a design to something nature has?

Adds a per-sequence column: the distance to the nearest sequence of the natural
alignment **that already carries the target residue at the critic site** (8728 of
the 82 021 here). Energy says whether the model finds a design plausible; this
says whether the family contains anything like it.

`--allowed-cols` restricts the comparison to the columns the sampler could move
and `--as-percentage` reports it as a share of them. That is the version worth
plotting: outside those columns a design *is* the wild type, so counting the rest
adds the same large constant to every design and hides what separates them.

```bash
python scripts/add_natural_neighbour.py \
    --ensemble-dir ../results_serine/ensembles \
    --natural ../iter_aln_dedup.faa --critic-col 197 \
    --allowed-cols 180-220 --wt data/wt_trypsin.fasta --as-percentage \
    --natural-observables ../results/calibration/natural_observables.csv --device cuda
```

The column goes into each `observables.csv`, so `plot_ensemble.py --quantities
E_DCA hamming_to_wt P_S Q min_hamming_natural_S_pct` turns the 4 × 4 scatter
matrix into a 5 × 5 one. With `--natural-observables` the natural cloud gets the
column too, each sequence's own row excluded from its own minimum.

### `run_generative_check.py` — does resampling reproduce the training statistics?

The generative check of the model itself, not a design run: `lambda = 0`,
`beta = 1`, every non-gap column mutable. Chains are followed as they equilibrate
and compared at each checkpoint with the **reweighted** statistics of the training
alignment (weighted by `--weights`, over every sequence in it; the chains
themselves are unweighted, being i.i.d. draws from the model) — the Pearson coefficients of `f_i`, of `f_ij` and of the connected
correlations `C_ij` (off-diagonal blocks only), all computed with adabmDCA's own
`get_freq_single_point`, `get_freq_two_points` and `get_correlation_two_points`.

```bash
python scripts/run_generative_check.py \
    --wt data/wt_trypsin.fasta --model ../bmDCA_params.dat \
    --natural ../iter_aln_dedup.faa --weights ../bmDCA_weights.dat \
    --output ../results_generative/ --critic-col 197 --allowed-cols 0-259 \
    --n-chains 1000 --n-steps 520000 --trace-every 8000 --trace-schedule log \
    --device cuda --cache-model
```

Unlike a design run, this sampler treats the **gap as an ordinary state**: it may
be proposed, and a gap column may mutate. That is not optional — the training
alignment is 11.6 % gaps, and chains that keep one gap pattern explore a subspace
of the space the model was fitted on and cannot reach the right equilibrium. Pass
`--no-gap-moves` to see what that costs: chains started at the wild type never
leave the trypsin basin (81 substitutions away after 2000 sweeps, against a family
that is 160 away) and `f_i` stalls at 0.79 instead of 0.98.

Two other options matter for reading the result. `--ceiling-repeats` measures the
**finite-sample ceiling** by drawing `--n-chains` sequences from the natural
alignment itself: a batch of 1000 estimates `f_ij` from 1000 samples, so even a
perfect sampler reaches only 0.85 on `C_ij`, and that — not 1 — is the target.
`--start` moves the chains' starting point: `natural` draws one natural sequence
per chain under the weights, `random` draws a uniformly random sequence per chain.
Running all three is how one tells a model that cannot reproduce its statistics
from chains that have not mixed — the wild type and natural sequences approach
equilibrium from below in energy, random sequences from far above it, and only a
mixed chain ends in the same place from all three.

### `plot_generative_check.py` — the convergence figure

Pearson coefficients against their ceilings on the top row, per-sequence
observables against the weighted natural means below. Several runs overlay, which
is how the starting points are compared.

```bash
python scripts/plot_generative_check.py \
    --run-dir ../results_gibbs ../results_gibbs_natural ../results_gibbs_random \
    --log-x --no-nogap --figsize 15 8.6 --save docs/generative_check.png
```

### `plot_ensemble_trace.py` — how the ensemble gets there

One panel per quantity, showing the batch mean as a line and the inter-quartile
range across chains as a band, against the mean number of **accepted** moves per
chain (`--x proposed` for proposed steps, `--log-x` for the log axis that exposes
the transient). Natural means are drawn as dotted rules.

```bash
python scripts/plot_ensemble_trace.py \
    --ensemble-dir ensembles/ --mu calibration/mu.json \
    --natural-observables calibration/natural_observables.csv \
    --log-x --save trace.png
```

Each plotting script writes a PNG wherever `--save` points; `docs/` holds the
example outputs from the run described below.

---

## Specifying sites

Two positions matter to every run: the **critic site** whose conditional is biased,
and the **allowed sites** a move may touch. Both can be given in either of two
ways.

**By MSA column** (0-based), which needs nothing but the model:

```
--critic-col 197 --allowed-cols 180-220
```

For the alignment shipped in `data/`, column 197 *is* trypsin Asp189: it is the `D`
of the `DSCQGDSGGP` motif, and the model's conditional there reads
`P(D) = 0.979`, `P(S) = 0.008` on the wild type — the expected trypsin signature,
and a useful check that your alignment and model agree.

**By PDB residue number**, which needs a mapping CSV with columns `pdb_residue` and
`msa_col`:

```
--mapping mapping_3TGI.csv --critic-site 189 --allowed-sites 160-228
```

`--allowed-sites` accepts an inclusive range (`160-228`, insertion codes such as
`188A` included), a comma-separated list (`18,160,188A,189`), the keyword `sho`
(the 17 S1-pocket positions of Sho et al.), or any mixture of those. Sites that are
a gap in the wild type are dropped with a warning.

`msa_col` is read as 0-based Python indexing; add `--mapping-base 1` for a file
written by a 1-based pipeline such as Julia. Everything downstream, including the
`msa_col_changed` column of `trajectory.csv`, is 0-based.

Add `--list-sites` to any run to print the resolution and exit without sampling.

---

## Sample quality, and the calibration of mu

Low energy alone is a poor measure of whether a designed sequence is reasonable: a
sequence can score well simply by sitting in a very constrained context. To correct
for that, each sequence also gets a **context-dependent entropy (CDE)** — the mean
over sites of the Shannon entropy of the model's conditional at that site, given
the rest of the sequence. A low CDE means the model considers the sequence's own
context highly constraining.

The **sample quality** combines the two:

```
Q_mu(seq) = E_DCA(seq) - mu * CDE(seq)
```

`mu` is not a free knob — it is calibrated on the natural alignment. A natural
family is a set of sequences the model should rate as uniformly good, so `mu` is
chosen to make `Q` as nearly constant as possible across them. That is the
minimiser of `Var(E - mu * CDE)`, which has the closed form
`Cov(E, CDE) / Var(CDE)` — the slope of energy against context entropy that the
family itself defines. `calibrate_mu.py` computes it and reports the fit.

On the serine-protease family used here (82 021 aligned sequences, 20 000 scored),
energy and context entropy turn out to be almost collinear, with
`corr(E, CDE) = 0.976`:

```
mu* = 307.6      sd(E) = 70.9  ->  sd(Q) = 15.3      95.4 % of the energy variance removed
```

Since CDE is a *mean* over the `L = 260` sites, `mu* / L = 1.18`: the quality is
`E - 1.18 * (total conditional entropy)` — a free energy at an effective
temperature of about 1.2 that the natural family itself sets.

---

## Results from a real run

Six lambdas, 1000 chains each, 60 000 proposed moves per chain, 36 mutable columns
(`--allowed-cols 180-220`, less the five columns that are gaps in the wild type),
critic column 197. About 35 minutes on one RTX 3090.

| lambda | ⟨E_DCA⟩ | ⟨Hamming⟩ | ⟨P(S)⟩ | max P(S) | chains ≥ 0.80 | ⟨CDE⟩ | ⟨Q_mu⟩ (sd) | acceptance |
|---|---|---|---|---|---|---|---|---|
| 0 | 224.9 | 5.4 | 0.005 | 0.028 | 0 % | 0.767 | −11.1 (4.5) | 0.039 |
| 10 | 254.8 | 9.6 | 0.152 | 0.378 | 0 % | 0.813 | 4.6 (5.2) | 0.080 |
| 50 | 279.4 | 15.0 | 0.515 | 0.655 | 0 % | 0.866 | 12.9 (5.3) | 0.062 |
| 100 | 291.6 | 17.6 | 0.641 | 0.730 | 0 % | 0.885 | 19.3 (5.0) | 0.050 |
| 300 | 314.6 | 22.2 | 0.772 | 0.805 | 1.4 % | 0.919 | 31.8 (4.5) | 0.027 |
| 1000 | 334.0 | 25.9 | 0.818 | 0.825 | **100 %** | 0.942 | 44.2 (3.1) | 0.011 |
| *natural* | *364.9* | *161.0* | *0.110* | *0.998* | *5.2 %* | *1.156* | *9.2 (15.3)* | — |

**The bias works, and the threshold is reachable.** At `lambda = 1000` every one of
the 1000 chains ends at `P(S) >= 0.80`, against `P(S) = 0.008` for the wild type.
But `P(S)` saturates there: the mean is 0.818 and the best chain reaches 0.825, so
on these 36 columns the model will not put much more than four fifths of the
conditional mass on serine, however hard it is pushed.

**Specificity is bought at a steady price.** Each step up in lambda raises `P(S)`
and pays in energy, in substitutions and in context entropy together. In the energy
against `P(S)` panel the six ensembles lie on *one* front rather than six separate
clouds: lambda slides along a single trade-off curve, it does not find a cheaper
route to the same specificity.

**The quality ceiling bites before the specificity ceiling.** Natural sequences have
`Q = 9.2 ± 15.3`. The `lambda = 10–50` ensembles sit comfortably inside that band
(4.6–12.9), `lambda = 100` is at its edge, and the `lambda = 1000` ensemble that
clears the threshold sits at 44.2 — more than two standard deviations above
anything the family does, with a context entropy (0.94) well below the natural mean
(1.16). Sequences that satisfy the specificity constraint are therefore
*model-plausible but not family-typical*: interesting as designs, not as members of
the natural ensemble.

**Writing the serine in changes almost nothing — and that is the point.** The
table above describes the sequences the chains ended on, which at low lambda still
carry the wild-type aspartate at position 189. Substituting `S` there and
rescoring (`substitute_critic.py`, figures `ensemble_pairs_serine.png` and
`critic_substitution_deltas.png`) leaves `P(S)` identical to machine precision —
the conditional does not depend on that residue — and moves the energy by at most
0.9 and the quality by at most 2.5, against a natural spread of 15.3. The sign is
informative: at `lambda = 0` the substitution is a real penalty (`+5.3` in energy,
every chain on the wrong side of zero), while from `lambda = 50` upward it *lowers*
the energy of the chains it touches, by up to 4.0 at `lambda = 1000` — those
contexts already wanted a serine and were simply caught on another residue. The
share of chains that needed no substitution at all rises from 0.7 % to 84.2 %
across the lambda range. Section 6.6 of [`docs/METHODS.md`](docs/METHODS.md) has
the full tables.

**The designs walk away from the natural sequences that already solve the
problem, and overtake them.** Over the 36 mutable columns, the distance to the
nearest natural S-carrier rises monotonically with lambda — median 22 % at
`lambda = 0` (the wild-type pocket itself), 53 % at `lambda = 1000`. A typical
natural sequence sits 39 % from its nearest S-carrying relative over those same
columns, so the weakly biased ensembles are closer to the natural S-carriers than
naturals are to each other, while from `lambda = 50` up the designs are further.
Past that point the bias is steering away from the arrangement the family uses,
not toward it.

**Equilibration is fast but not monotone.** Nearly everything happens within the
first ~100 accepted moves per chain, and at large lambda the ensemble overshoots
before settling: at `lambda = 1000` the mean energy peaks near 368 around 40
accepted moves — above the natural mean of 365 — then relaxes to its stationary
334, while the sample quality peaks near 83 and falls back to 44. The bias hauls
the sequence up the energy scale first, and the chain anneals part of that cost
away afterwards, once `P(S)` has nothing left to gain. Practically: a run truncated
after a few thousand proposed steps reports a materially worse energy than the
equilibrium one.

---

### Is the model itself any good?

A separate question from the designs, answered by `run_generative_check.py`:
resample the model with no bias at all (`lambda = 0`, `beta = 1`, every column
free, the gap an ordinary state) and compare with the reweighted statistics it
was trained on. 1000 chains, 520 000 proposed moves each, from three starting
points.

| sampler | start | sweeps | `f_i` | `f_ij` | `C_ij` | slope | ⟨E⟩ | ⟨Hamming to WT⟩ |
|---|---|---|---|---|---|---|---|---|
| Metropolis | wild type | 8000 | 0.990 | 0.978 | 0.562 | 0.55 | 449.9 | 169 |
| Metropolis | natural | 2000 | 0.993 | 0.983 | 0.601 | 0.59 | 442.5 | 167 |
| Gibbs | wild type | 2000 | 0.990 | 0.977 | 0.548 | 0.54 | 449.3 | 169 |
| Gibbs | natural | 2000 | 0.991 | 0.980 | 0.568 | 0.55 | 450.0 | 169 |
| Gibbs | random | 2000 | 0.990 | 0.978 | 0.540 | 0.52 | 453.9 | 169 |
| Gibbs | wild type | 2000 | 0.991 | 0.981 | 0.716 | 0.54 | 450.5 | 169 |
| Gibbs | random | 2000 | 0.991 | 0.981 | 0.711 | 0.54 | 451.1 | 169 |
| *target (ceiling at 1000 chains)* | — | — | *0.999* | *0.996* | *0.848* | *1.00* | *361.8* | *160* |
| *target (ceiling at 10 000 chains)* | — | — | *1.000* | *1.000* | *0.986* | *0.995* | *361.8* | *160* |

The last two Gibbs rows use **10 000** chains; every other row uses 1000. See
below for why that matters.

Two samplers and three starting points give one answer. The Gibbs runs — adabmDCA's
own `gibbs_step_uniform_sites`, the sampler the model was trained with — span 4.6
in energy while arriving from three directions: the wild type climbs from `E = 220`,
natural sequences rise from 362, and random sequences fall from 1197. The
Metropolis chains land in the same place. Equilibrium is near `E = 450`, about 88
above the weighted natural mean of 361.8.

Single- and two-site frequencies come close to their ceilings; the **connected
correlations reach 0.55 of a possible 0.85, with a slope of 0.54**, meaning the
resampled correlations are systematically about 45 % weaker than the natural ones.
The model has learnt the family's composition well and a little over half of its
correlation structure.

That shortfall is the model, not the measurement: quadrupling the Metropolis run
(2000 → 8000 sweeps) lifts `f_i` from 0.976 to 0.990 and leaves `C_ij` at 0.56,
the natural-start Gibbs run begins at `C_ij = 0.832` against a ceiling of 0.848
before the model's own dynamics carry it down to 0.57, and the random-start run
climbs to the same 0.54 from `C_ij = 0.0001` — the same equilibrium reached from
below as well as from above.

**Quote the slope, not the Pearson coefficient.** Repeating the check with 10 000
chains raises the `C_ij` ceiling from 0.848 to 0.986 and the measured `C_ij` from
0.548 to 0.716, so a third of the apparent shortfall at 1000 chains was noise in
the batch's own estimate — and the ratio to the ceiling is not stable either
(0.65 against 0.73), so dividing by it is not a valid correction. The slope does
not move: 0.539 at 1000 chains, 0.538 at 10 000, with a ceiling of 1.00 at both,
because adabmDCA regresses the chains' `C_ij` on the natural `C_ij` as predictor
and noise in the response does not bias that coefficient. The energy gap is
likewise unmoved (450.5 against 449.3). METHODS §7.5 derives this; the figure is
`docs/generative_check_10k.png`.

Read `docs/METHODS.md` §7 before quoting those numbers: the ceiling, the gap
treatment and the residual non-convergence of the wild-type run all bear on them.

---

### A second model: `model_pcd_serine_protease`

The whole analysis repeated on a second set of parameters for the same family.
Energies are gauge-dependent, so nothing transfers between models; `mu` is
recalibrated to **316.53** and every number below is recomputed. METHODS §9 has
the commands.

Generative check, 25 000 Gibbs chains, 4000 sweeps, from two opposite starts:

| start | `f_i` | `f_ij` | `C_ij` | slope | ⟨E⟩ |
|---|---|---|---|---|---|
| random (from E = 1199) | 0.993 | 0.985 | 0.762 | 0.595 | 373.5 |
| natural (from E = 289) | 0.993 | 0.985 | 0.761 | 0.593 | 373.6 |
| *ceiling at 25 000 chains* | *1.000* | *1.000* | *0.996* | *0.998* | *289.0* |

The two starts meet to 0.1 in energy and 0.0008 in `C_ij` from 910 apart, so this
is equilibrium. Same verdict as the first model on different parameters:
composition right, connected correlations about 40 % too weak.

Designs, the same six lambdas as §6, 1000 chains each, S written into the critic
site afterwards:

| λ | ⟨E⟩ | ⟨Hamming to WT⟩ | ⟨P_S⟩ | P_S > 0.8 | ⟨Q⟩ | median distance to nearest natural S-carrier |
|---|---|---|---|---|---|---|
| 0 | 155.0 | 6.4 | 0.005 | 0 % | −94.3 | 25.0 % |
| 10 | 179.8 | 10.2 | 0.138 | 0 % | −84.0 | 33.3 % |
| 50 | 203.6 | 15.4 | 0.480 | 0 % | −75.9 | 41.7 % |
| 100 | 216.8 | 18.1 | 0.611 | 0 % | −68.6 | 44.4 % |
| 300 | 242.1 | 23.0 | 0.748 | 0 % | −53.0 | 50.0 % |
| 1000 | 266.4 | 27.3 | 0.796 | 23.6 % | −34.9 | 55.6 % |
| *natural* | *289.6* | *161.0* | *0.110* | — | *−79.1 ± 16.5* | *38.9 %* |

The §6 trade-off again, and monotone in λ across the whole ladder: `P_S` is bought
with energy and with distance from the family. `docs/critic_substitution_deltas_pcd.png`
shows what writing S costs per chain: wholly positive at λ = 0, straddling zero at
λ = 10, wholly negative from λ = 50 up. Only λ = 1000 puts any chains above
`P_S` = 0.8, and by then ⟨Q⟩ has left the natural band. Figures:
`docs/mu_calibration_pcd.png`, `docs/ensemble_pairs_pcd.png`,
`docs/ensemble_trace_pcd.png`, `docs/generative_check_pcd.png`.

The model's own equilibrium sample, the 10 000 persistent chains in
`chains.fasta`, scored the same way, with the ceiling measured at the same M:

| | `f_i` | `f_ij` | `C_ij` | slope | ⟨E⟩ |
|---|---|---|---|---|---|
| training chains (10 000) | 0.993 | 0.986 | **0.755** | 0.576 | 376.4 |
| *ceiling at 10 000 chains* | *1.000* | *1.000* | *0.980* | *1.004* | *289.0* |

That matches the independent 25 000-chain run above (0.762, E = 373.5) from a
different sampler and the opposite starting point.

### Cooling the Potts term: what β buys

`--beta` multiplies the DCA energy; the critic's conditional is now always
evaluated at β = 1 (`CRITIC_BETA`), so cooling the energy no longer sharpens the
bias as a side effect. At β = 1 nothing changes, so every result above stands.
With the critic fixed the measure factorises exactly:

```
exp(-beta E + lambda log P_1)  =  exp(-beta [ E - lambda_tilde log P_1 ]),   lambda_tilde = lambda / beta
```

so `lambda_tilde` says *what* is optimised and β only says *how cold*. Nineteen
ensembles over the (β, λ) grid, all rescored at β = 1, give three answers:

* **β saturates by about 5.** At equal `lambda_tilde`, β = 5 and β = 10 agree to
  the third digit in every observable — the sampler has reached its
  zero-temperature limit and the ensemble depends on `lambda_tilde` alone.
* **Cooling at fixed bias changes almost nothing.** Along `lambda_tilde` = 1000,
  β = 1 → 3 moves the energy 266.4 → 264.3 and leaves `P_S` at 0.797, while the
  distinct sequences fall 466 → 53.
* **At matched energy — the framing the suggestion came in — cooling does help,
  most where it matters.** At the energy each cooled run reaches, against what
  β = 1 would have given there:

| cooled run | ⟨E⟩ | P_S cooled | P_S at β = 1 | gain | diversity cooled | at β = 1 |
|---|---|---|---|---|---|---|
| β = 5, `lambda_tilde` = 100 | 208.6 | 0.631 | 0.530 | **+0.101** | 2.4 | 11.2 |
| β = 5, `lambda_tilde` = 200 | 226.9 | 0.719 | 0.666 | +0.054 | 1.9 | 9.5 |
| β = 5, `lambda_tilde` = 300 | 241.6 | 0.763 | 0.746 | +0.018 | 2.0 | 7.5 |

  The gain grows as the energy falls toward the wild type's 143.7 — a fifth more
  specificity at the same energy, for a library four to five times less diverse.
  (Holding `lambda_tilde` fixed instead, the gain is only +0.02 in P_S.)

So: `lambda_tilde` places the designs, β sharpens them. Use `lambda_tilde` at β = 1 to place the designs against the natural band
(`lambda_tilde` = 50 gives Q = −75.9 with 1000 distinct sequences); a mild
β = 1.5–2 is defensible if every chain must carry S unaided. Beyond β ≈ 3 the
ensemble stops being a library.

Figures: `docs/beta_scan_pcd.png` (the grid), `docs/beta_matched_pairs_pcd.png`
(the 5×5 at matched `lambda_tilde` = 100, where the cold libraries are a tight knot
inside the warm one) and `docs/beta_matched_trace_pcd.png` (the same three along
the run). Detail in METHODS §10.

**One caveat on this model's training log.** It reports `Pearson = 0.950027` for
those very chains, against the 0.755 measured here — reproducing the log's `M_eff`
and pseudocount exactly, unchanged in float64. `params.dat`, `chains.fasta` and
`weights.dat` agree with each other; the logged 0.95 is the outlier. Since
`target_pearson` is the stopping rule, training may have stopped early. METHODS
§9.1 has the detail.


## Conventions and gotchas

* **Alphabet.** The default is adabmDCA's `protein` token set,
  `-ACDEFGHIKLMNPQRSTVWY`, with the **gap first** (index 0). Tokens come from
  `adabmDCA.alphabet.get_tokens`, encoding and decoding from
  `adabmDCA.fasta.encode_sequence`/`decode_sequence`, and FASTA input/output from
  `adabmDCA.Alignment`/`read_alignment`/`write_alignment`. Use `--alphabet` to
  override — for instance `ACDEFGHIKLMNPQRSTVWY-` for the gap-last convention used
  by some older pipelines. The same token string is used to read the parameter file
  and the sequences, so energies are invariant under the relabelling; there is a
  test for exactly this.
* **Parameters** are read by `adabmDCA.load_model`, which expects the `h`/`J` text
  records adabmDCA writes. Loading verifies that the self-couplings `J[i, :, i, :]`
  vanish, which the conditional and the substitution energy both assume.
* **MSA columns are 0-based** everywhere in the code and in every output file.
* **The wild-type FASTA must be the aligned row**, of exactly the model's length.
  An ungapped sequence will be rejected with a length error.
* **Energies are not comparable across models.** The DCA energy has a gauge; only
  differences within one model mean anything.

---

## Repository layout

```
data/        three aligned MSA rows: the wild type and two chimera donors
docs/        example figures produced by the plotting scripts
src/
  encoding.py     alphabet handling and FASTA I/O, over adabmDCA
  potts.py        model loading, energies, conditionals, substitution energies
  sites.py        PDB-to-MSA mapping, site selection, chimera construction
  mcmc.py         the single-chain biased sampler and its output files
  ensemble.py     the same moves, vectorised over a thousand chains (CPU or GPU)
  observables.py  energy, Hamming distance, P(S), context entropy, sample quality
  natural_stats.py  reweighted f_i / f_ij / C_ij of the alignment, and Pearson against them
scripts/
  run_mcmc.py             one chain
  run_sweep.py            lambdas x replicates, in parallel
  plot_trajectories.py    the 4x3 trajectory figure
  run_ensemble.py         many chains per lambda, scored
  calibrate_mu.py         mu from the natural alignment, with its figure
  substitute_critic.py    write the target residue into the critic site, rescore
  add_natural_neighbour.py  distance to the nearest natural sequence with that residue
  plot_ensemble.py        the pairwise scatter matrix
  plot_critic_deltas.py   the cost of that substitution, chain by chain
  run_generative_check.py   unbiased resampling against the training statistics
  plot_generative_check.py  the convergence figure for that check
  plot_ensemble_trace.py  the observables along the run
  _common.py              CLI options shared by the scripts
tests/       energy/conditional consistency and sampler correctness
```

The modules install at top level, so `from mcmc import run_mcmc` or
`from observables import context_dependent_entropy` works in a notebook once the
package is installed. The scripts also run straight from a checkout without
installing.

---

## Implementation notes

* **One expression, two uses.** `potts.site_logits(i, seq)` is adabmDCA's Gibbs
  expression, `bias[i] + couplings[i] · x`. Its softmax is the conditional, and the
  difference of two of its entries is the substitution energy. Because both come
  from the same quantity they cannot drift apart; the tests check each of them
  against energies recomputed by `adabmDCA.statmech.compute_energy`.
* **Incremental updates, verified.** Along a chain the energy is updated by a single
  substitution term and the critic-site logits by one coupling column; both are
  recomputed from scratch at the end of a chain as a guard against accumulated
  floating-point error.
* **Correctness of the batched sampler.** The ensemble sampler is tested against the
  *exact* biased Boltzmann distribution: on a small model with a mutable space small
  enough to enumerate, the histogram of final states is compared with
  `exp(-beta E) * P(S)^lambda` over every reachable state, at `lambda = 0` and
  `lambda = 3`.
* **Throughput.** A single chain runs at roughly 2500 proposed moves per second per
  CPU core. The batched sampler advances 1000 chains at about 200 steps per second
  on a GPU — that is 200 000 chain-moves per second, and kernel-launch latency, not
  arithmetic, is the limit, so larger batches are nearly free.
