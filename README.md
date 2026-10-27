# Trypsin → chymotrypsin specificity switch by biased MCMC

Redesigning the substrate pocket of a serine protease *in silico*: this repository
samples protein sequences that start from wild-type rat trypsin and drift toward a
chymotrypsin-like specificity, under a Potts (bmDCA) model of the serine-protease
family trained with [adabmDCA 2.1](https://github.com/spqb/adabmDCApy).

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
git clone <repository-url>
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

Six lambdas, 1000 chains each, 60 000 proposed moves per chain, 41 mutable columns
(`--allowed-cols 180-220`), critic column 197. About 35 minutes on one RTX 3090.

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
on these 41 columns the model will not put much more than four fifths of the
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
scripts/
  run_mcmc.py             one chain
  run_sweep.py            lambdas x replicates, in parallel
  plot_trajectories.py    the 4x3 trajectory figure
  run_ensemble.py         many chains per lambda, scored
  calibrate_mu.py         mu from the natural alignment, with its figure
  plot_ensemble.py        the pairwise scatter matrix
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
