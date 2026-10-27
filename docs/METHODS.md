# Biased Potts sampling for a trypsin → chymotrypsin specificity switch

*Methods draft. Every quantity below is defined once and then reused; each
subsection ends with the command that reproduces it. Numerical values are those
of the run shipped with this repository.*

---

## 1. Model and notation

A sequence is a vector $\sigma = (\sigma_1, \dots, \sigma_L)$ with
$\sigma_i \in \{1, \dots, q\}$ over the alphabet `-ACDEFGHIKLMNPQRSTVWY`
($q = 21$, gap first). The Potts model fitted to the serine-protease alignment by
adabmDCA 2.1 assigns the statistical energy

$$
E(\sigma) \;=\; -\sum_{i=1}^{L} h_i(\sigma_i) \;-\; \frac{1}{2}\sum_{i \neq j} J_{ij}(\sigma_i, \sigma_j),
$$

with fields $h \in \mathbb{R}^{L \times q}$ and couplings
$J \in \mathbb{R}^{L \times q \times L \times q}$, symmetric
($J_{ij}(a,b) = J_{ji}(b,a)$) and with vanishing diagonal blocks
($J_{ii} \equiv 0$, checked at load time). The model used here has $L = 260$,
$q = 21$.

Define the **site logits** of site $i$ in the context of $\sigma$,

$$
\ell_i(a \mid \sigma) \;=\; h_i(a) \;+\; \sum_{j \neq i} J_{ij}(a, \sigma_j),
$$

which is the quantity adabmDCA itself computes in its Gibbs step. Two objects
follow from it, and the implementation derives both from the same expression so
they cannot disagree:

$$
P_\beta(\sigma_i = a \mid \sigma_{\setminus i}) \;=\; \frac{e^{\beta \ell_i(a \mid \sigma)}}{\sum_{b} e^{\beta \ell_i(b \mid \sigma)}},
\qquad
\Delta E_{i: a \to b}(\sigma) \;=\; -\bigl[\ell_i(b \mid \sigma) - \ell_i(a \mid \sigma)\bigr].
$$

The first is the single-site conditional of the Boltzmann measure
$\propto e^{-\beta E}$; the second is the exact energy change of a single
substitution. Note that $\ell_i(\cdot \mid \sigma)$ does not depend on $\sigma_i$:
*the conditional at a site is a property of its context, not of its own residue.*

---

## 2. System and design target

The wild type $\sigma^{\mathrm{wt}}$ is rat anionic trypsin (TRY2_RAT) as its
aligned row of the family MSA (`data/wt_trypsin.fasta`, $L = 260$).

The **critic site** $i^\star$ is the S1-pocket position that discriminates the two
specificities — residue 189 in chymotrypsinogen (3TGI) numbering, Asp in trypsin
and Ser in chymotrypsin. In the alignment used here it is MSA column
$i^\star = 197$ (0-based), identifiable as the `D` of the `DSCQGDSGGP` motif. The
model reproduces the trypsin signature there:

$$
P(\mathrm{D} \mid i^\star, \sigma^{\mathrm{wt}}) = 0.979,
\qquad
P(\mathrm{S} \mid i^\star, \sigma^{\mathrm{wt}}) = 0.008,
\qquad
E(\sigma^{\mathrm{wt}}) = 219.70 .
$$

The **design target** is to raise $P(\mathrm{S} \mid i^\star, \sigma)$ by mutating
the surrounding pocket. Let $\mathcal{A}$ denote the set of mutable columns. Here
$\mathcal{A}$ is the window of MSA columns $180$–$220$ minus the five columns
(211–215) that are gaps in $\sigma^{\mathrm{wt}}$, leaving
$|\mathcal{A}| = 36$ mutable positions. Gap columns are never mutated and the gap
symbol is never proposed, so every sampled sequence has the gap pattern of the
wild type.

Reproduce the site resolution (loads the model and exits):

```bash
python scripts/run_mcmc.py \
    --wt data/wt_trypsin.fasta --model ../bmDCA_params.dat --output /tmp/out \
    --critic-col 197 --allowed-cols 180-220 --list-sites --cache-model
```

---

## 3. Sample quality and the calibration of $\mu$

### 3.1 Context-dependent entropy

Energy alone is a poor score for a designed sequence: a sequence can be assigned a
low energy merely by occupying a strongly constrained context. We therefore also
measure, for each sequence, the mean conditional entropy over sites — the
**context-dependent entropy**

$$
S_{\mathrm{CDE}}(\sigma) \;=\; \frac{1}{L}\sum_{i=1}^{L} H\bigl[P_\beta(\cdot \mid \sigma_{\setminus i})\bigr],
\qquad
H[p] = -\sum_{a=1}^{q} p_a \log p_a ,
$$

in nats. Small $S_{\mathrm{CDE}}$ means the model regards the sequence's own
context as highly determining.

### 3.2 Definition of the sample quality

The **sample quality** trades the two against each other with a single weight
$\mu$:

$$
Q_\mu(\sigma) \;=\; E(\sigma) \;-\; \mu\, S_{\mathrm{CDE}}(\sigma).
$$

### 3.3 Calibration on the natural family

$\mu$ is not a free parameter. A natural alignment is a set of sequences the model
ought to rate as uniformly acceptable, so we fix $\mu$ as the value that makes
$Q_\mu$ as nearly constant as possible over the natural family
$\mathcal{N} = \{\sigma^{(1)}, \dots, \sigma^{(N)}\}$:

$$
\mu^\star \;=\; \arg\min_{\mu} \operatorname{Var}_{\mathcal{N}}\bigl[E - \mu\, S_{\mathrm{CDE}}\bigr]
\;=\; \frac{\operatorname{Cov}_{\mathcal{N}}(E,\, S_{\mathrm{CDE}})}{\operatorname{Var}_{\mathcal{N}}(S_{\mathrm{CDE}})} ,
$$

the regression slope of energy on context entropy within the family. The residual
variance at the optimum is
$\operatorname{Var}(E)\,\bigl[1 - \rho^2\bigr]$ with
$\rho = \operatorname{corr}(E, S_{\mathrm{CDE}})$, so the fraction of energy
variance removed is $\rho^2$.

### 3.4 Result

On $N = 20{,}000$ sequences drawn without replacement (seed 0) from the family
alignment of 82,021 aligned sequences:

| quantity | value |
|---|---|
| $\rho = \operatorname{corr}(E, S_{\mathrm{CDE}})$ | $0.9765$ |
| $\mu^\star$ | $307.63$ |
| $\operatorname{sd}(E)$ | $70.95$ |
| $\operatorname{sd}(Q_{\mu^\star})$ | $15.30$ |
| variance removed | $95.4\,\%$ |
| $\langle E \rangle_{\mathcal{N}}$ | $364.9$ |
| $\langle S_{\mathrm{CDE}} \rangle_{\mathcal{N}}$ | $1.156$ |
| $\langle Q_{\mu^\star} \rangle_{\mathcal{N}}$ | $9.2 \pm 15.3$ |

Energy and context entropy are nearly collinear within the family. Because
$S_{\mathrm{CDE}}$ is a *mean* over sites, $\mu^\star / L = 1.183$, so

$$
Q_{\mu^\star}(\sigma) \;\simeq\; E(\sigma) - T_{\mathrm{eff}} \sum_{i=1}^{L} H\bigl[P(\cdot \mid \sigma_{\setminus i})\bigr],
\qquad T_{\mathrm{eff}} = 1.18 ,
$$

i.e. a free energy at an effective temperature the natural family itself sets.

```bash
python scripts/calibrate_mu.py \
    --natural ../iter_aln_dedup.faa --model ../bmDCA_params.dat \
    --output calibration/ --max-sequences 20000 --seed 0 \
    --wt data/wt_trypsin.fasta --critic-col 197 --cache-model
```

Writes `calibration/mu.json`, `calibration/natural_observables.csv` and the
calibration figure (§6.1). Runtime ≈ 13 s on 8 CPU threads.

---

## 4. Biased sampling

### 4.1 Target distribution

Sampling is performed under the modified Hamiltonian

$$
H_{\beta,\lambda}(\sigma) \;=\; \beta\, E(\sigma) \;-\; \lambda \log P_\beta\bigl(\mathrm{S} \mid i^\star, \sigma_{\setminus i^\star}\bigr),
$$

whose stationary distribution is

$$
\pi_{\beta,\lambda}(\sigma) \;\propto\; e^{-\beta E(\sigma)} \; P_\beta\bigl(\mathrm{S} \mid i^\star, \sigma\bigr)^{\lambda}.
$$

$\lambda$ is the bias strength: $\lambda = 0$ recovers plain Potts sampling at
inverse temperature $\beta$, and increasing $\lambda$ tilts the measure toward
contexts that favour serine at the critic site, at the price of energy. All runs
reported here use $\beta = 1$.

### 4.2 Move set and acceptance

One move picks a site $i \in \mathcal{A}$ uniformly, then a replacement residue
$b$ uniformly among the $q - 1 = 20$ non-gap symbols different from $\sigma_i$.
The proposal is symmetric, $g(\sigma \to \sigma') = g(\sigma' \to \sigma)$, so the
Metropolis–Hastings acceptance reduces to

$$
A(\sigma \to \sigma') \;=\; \min\bigl\{1,\; e^{-\Delta H}\bigr\},
\qquad
\Delta H \;=\; \beta\,\Delta E \;-\; \lambda\,\bigl[\log P'_{\mathrm{S}} - \log P_{\mathrm{S}}\bigr].
$$

Two identities make each move $O(L)$ rather than $O(L^2)$. First, $\Delta E$ comes
from the site logits of §1. Second, because $\ell_{i^\star}$ is linear in the
one-hot encoding, a substitution at $i \neq i^\star$ updates it in $O(q)$:

$$
\ell_{i^\star}(\cdot \mid \sigma') \;=\; \ell_{i^\star}(\cdot \mid \sigma) \;+\; J_{i^\star i}(\cdot, b) \;-\; J_{i^\star i}(\cdot, a),
$$

and a substitution *at* $i^\star$ leaves $\ell_{i^\star}$ unchanged, so such a move
alters the energy term only. Both running quantities are recomputed from scratch at
the end of a chain as a guard against accumulated floating-point error.

### 4.3 Two regimes

**(a) Single chains.** A chain is run until $P_{\mathrm{S}} \geq \theta$
(threshold, default $0.80$) or until a step budget is exhausted, recording its
state every $n$ accepted moves. This yields trajectories — the route taken and its
cost.

**(b) Ensembles.** $M$ independent chains start from $\sigma^{\mathrm{wt}}$, run a
fixed number of proposed moves with no stopping rule, and contribute their final
state. This samples $\pi_{\beta,\lambda}$ itself. Chains are advanced jointly as a
tensor: one site is drawn per step for the whole batch and all $M$ substitutions
are scored with a single matrix product (the strategy of adabmDCA's
uniform-site Gibbs sweep). Chains share only the sequence of visited sites, never a
residue or an acceptance, so they remain statistically independent.

### 4.4 Verification

The batched sampler is checked against the exact target rather than against itself.
On a model small enough that the reachable space
$\{\sigma : \sigma_i = \sigma^{\mathrm{wt}}_i \,\forall i \notin \mathcal{A}\}$ can
be enumerated, the empirical histogram of final states is compared with
$\pi_{\beta,\lambda}$ computed by summation, at $\lambda = 0$ and $\lambda = 3$;
the test requires total variation below $0.06$ and correlation above $0.98$
(`tests/test_ensemble.py`). The single-site conditional, the substitution energy
and the batched observables are each checked against values recomputed with
`adabmDCA.statmech.compute_energy` (`tests/test_model.py`).

```bash
pytest          # 26 tests, ~10 s, no model file required
```

---

## 5. Observables

For a sampled sequence $\sigma$ we record

$$
E(\sigma), \qquad
d_H(\sigma, \sigma^{\mathrm{wt}}) = \sum_{i=1}^{L} \mathbb{1}[\sigma_i \neq \sigma^{\mathrm{wt}}_i], \qquad
P_{\mathrm{S}}(\sigma) = P_\beta(\mathrm{S} \mid i^\star, \sigma), \qquad
Q_{\mu^\star}(\sigma),
$$

that is: plausibility, distance travelled, specificity achieved, and corrected
plausibility.

---

## 6. Runs and figures

### 6.1 Calibration figure

Four panels: the energy histogram of the natural alignment; the
$E$ vs $S_{\mathrm{CDE}}$ scatter with the fitted slope $\mu^\star$; the scan of
$\operatorname{Var}(E - \mu S_{\mathrm{CDE}})$ against $\mu$ with its minimum; and
the distributions of $E$ and $Q_{\mu^\star}$ on a common scale. Produced by the
command of §3.4 as `calibration/mu_calibration.png` (copy in
`docs/mu_calibration.png`).

### 6.2 Single-chain sweep and trajectory figure

Four independent chains for each of $\lambda \in \{10, 20, 50\}$, up to $2\times10^5$
proposed moves, recording every 25 accepted moves.

```bash
python scripts/run_sweep.py \
    --wt data/wt_trypsin.fasta --model ../bmDCA_params.dat \
    --output results/ --critic-col 197 --allowed-cols 180-220 \
    --lambdas 10 20 50 --threshold 0.80 \
    --max-steps 200000 --record-every 25 \
    --n-runs 4 --seed 1 --n-jobs 12 --cache-model

python scripts/plot_trajectories.py \
    --output-dir results/ --lambdas 10 20 50 --plot-lambda 20 \
    --critic-col 197 --wt data/wt_trypsin.fasta --model ../bmDCA_params.dat \
    --target-source data/chymotrypsin_bovine.fasta --allowed-cols 180-220 \
    --cache-model --save docs/example_sweep.png
```

The figure is $4 \times 3$, one column per $\lambda$: (i) $P_{\mathrm{S}}$ against
accepted step, per chain, with median and inter-quartile band and the threshold
rule; (ii) $E$ against accepted step; (iii) $E$ against $P_{\mathrm{S}}$ with the
wild type and a chimeric reference marked; (iv) the effective energy
$E - \lambda_{\mathrm{plot}} \log P_{\mathrm{S}}$ evaluated at a *single* chosen
$\lambda_{\mathrm{plot}}$ for all columns, so chains sampled under different biases
are placed on one scale. Runtime ≈ 1.5 min on 12 cores.

### 6.3 Ensembles

$M = 1000$ chains per $\lambda$, $6 \times 10^4$ proposed moves each,
$\lambda \in \{0, 10, 50, 100, 300, 1000\}$, $\beta = 1$, $|\mathcal{A}| = 36$.

```bash
python scripts/run_ensemble.py \
    --wt data/wt_trypsin.fasta --model ../bmDCA_params.dat \
    --output ensembles/ --critic-col 197 --allowed-cols 180-220 \
    --lambdas 0 10 50 100 300 1000 --n-chains 1000 --n-steps 60000 \
    --trace-every 1000 --trace-schedule log --trace-lambdas 10 100 1000 \
    --seed 0 --device cuda --cache-model
```

Runtime ≈ 35 min on one RTX 3090 (≈ 200 batch-steps/s, i.e. $2\times10^5$
chain-moves/s). Each $\lambda$ yields `final_sequences.fasta` (the 1000 final
sequences), `observables.csv`, `summary.json`, and for traced $\lambda$ a
`trace.csv` of batch means and quartiles at 55 log-spaced checkpoints. Recording
does not consume random numbers, so traced and untraced runs with the same seed
produce bit-identical chains (verified by direct comparison).

**Stationary values** (mean over the 1000 final sequences; natural row for
reference):

| $\lambda$ | $\langle E\rangle$ | $\langle d_H\rangle$ | $\langle P_{\mathrm{S}}\rangle$ | $\max P_{\mathrm{S}}$ | $P_{\mathrm{S}}\!\geq\!0.8$ | $\langle S_{\mathrm{CDE}}\rangle$ | $\langle Q_{\mu^\star}\rangle$ (sd) | acceptance |
|---|---|---|---|---|---|---|---|---|
| 0 | 224.9 | 5.4 | 0.005 | 0.028 | 0 % | 0.767 | −11.1 (4.5) | 0.039 |
| 10 | 254.8 | 9.6 | 0.152 | 0.378 | 0 % | 0.813 | 4.6 (5.2) | 0.081 |
| 50 | 279.4 | 15.0 | 0.515 | 0.655 | 0 % | 0.866 | 12.9 (5.3) | 0.062 |
| 100 | 291.6 | 17.6 | 0.641 | 0.730 | 0 % | 0.885 | 19.3 (5.0) | 0.050 |
| 300 | 314.6 | 22.2 | 0.772 | 0.805 | 1.4 % | 0.919 | 31.8 (4.5) | 0.027 |
| 1000 | 334.0 | 25.9 | 0.818 | 0.825 | **100 %** | 0.942 | 44.2 (3.1) | 0.011 |
| natural | 364.9 | 161.0 | 0.110 | 0.998 | 5.2 % | 1.156 | 9.2 (15.3) | — |

Three observations.

**(i) The constraint is satisfiable, but saturates.** At $\lambda = 1000$ all 1000
chains satisfy $P_{\mathrm{S}} \geq 0.80$, against $0.008$ for the wild type. The
ensemble mean is $0.818$ and the best single chain reaches $0.825$: on
$\mathcal{A}$ the attainable conditional mass on serine has a ceiling near $0.82$,
independent of further increase in $\lambda$.

**(ii) The cost is a single front.** $\langle E \rangle$, $\langle d_H \rangle$ and
$\langle S_{\mathrm{CDE}} \rangle$ all increase monotonically with
$\langle P_{\mathrm{S}} \rangle$. In the $(P_{\mathrm{S}}, E)$ plane the six
ensembles fall on one curve rather than six separate clouds, so $\lambda$ moves the
ensemble *along* a trade-off front rather than discovering a cheaper one.

**(iii) The quality constraint binds first.** Natural sequences have
$Q_{\mu^\star} = 9.2 \pm 15.3$. The $\lambda \in \{10, 50\}$ ensembles lie inside
that band ($4.6$, $12.9$), $\lambda = 100$ at its edge, while the $\lambda = 1000$
ensemble that satisfies the design constraint sits at $44.2$, more than two
standard deviations above the family, with $S_{\mathrm{CDE}} = 0.94$ against the
natural $1.16$. Designs that switch the pocket are therefore model-plausible but
not family-typical.

### 6.4 Pairwise figure

```bash
python scripts/plot_ensemble.py \
    --ensemble-dir ensembles/ --mu calibration/mu.json \
    --natural-observables calibration/natural_observables.csv \
    --max-points 1000 --save docs/ensemble_pairs.png
```

Lower-triangular scatter matrix of the four observables of §5, one point per final
sequence, marginals on the diagonal, one shade per $\lambda$ along a single hue
(an ordered magnitude), with the natural alignment behind in grey.

### 6.5 Relaxation figure

```bash
python scripts/plot_ensemble_trace.py \
    --ensemble-dir ensembles/ --mu calibration/mu.json \
    --natural-observables calibration/natural_observables.csv \
    --log-x --save docs/ensemble_trace.png
```

One panel per observable: batch mean and inter-quartile range across the 1000
chains against the mean number of *accepted* moves per chain, on a logarithmic
axis.

Relaxation is fast but **non-monotone**. Writing $t$ for accepted moves per chain,
the large-$\lambda$ ensembles overshoot and then anneal:

| $\lambda$ | $\max_t \langle E\rangle$ | at $t$ | stationary $\langle E\rangle$ | $\max_t \langle Q\rangle$ | at $t$ | stationary $\langle Q\rangle$ |
|---|---|---|---|---|---|---|
| 10 | 255.9 | 415 | 254.8 | 5.1 | 415 | 4.6 |
| 100 | 309.5 | 48 | 291.6 | 37.6 | 27 | 19.3 |
| 1000 | 368.8 | 47 | 334.0 | 82.3 | 38 | 44.2 |

At $\lambda = 1000$ the mean energy peaks at $368.8$ — above the natural mean of
$364.9$ — after only ~47 accepted moves, then relaxes to $334.0$; the sample
quality peaks at $82.3$ and falls to $44.2$. The bias first drives the sequence up
the energy scale to gain $P_{\mathrm{S}}$, and the chain subsequently recovers part
of that cost once $P_{\mathrm{S}}$ has saturated. Consequently a run truncated
after a few thousand proposed moves reports a materially worse energy than the
equilibrium ensemble. A second effect is visible in the same figure: per *accepted*
move, $\lambda = 1000$ rises later than $\lambda = 100$, because its acceptance is
a fifth as large ($0.011$ vs $0.050$) and only strongly favourable moves survive.

---

## 7. Reproducibility

* **Seeds.** Ensemble $\lambda_k$ uses seed `--seed + k`; sweep replicate $i$ uses
  `--seed + i`. Identical seeds reproduce chains exactly (tested).
* **Determinism of the trace.** Checkpointing draws no random numbers; the
  log-spaced and linear-spaced runs of §6.3 produced bit-identical
  `final_sequences.fasta`.
* **Model cache.** `--cache-model` stores the parsed tensors beside the parameter
  file (`<params>.cache.pt`), reducing load time from ≈ 32 s to ≈ 3 s; the cache is
  invalidated when the text file is newer.
* **Indexing.** MSA columns are 0-based throughout, in code and in all outputs. A
  1-based mapping file is read with `--mapping-base 1`.
* **Gauge.** DCA energies are gauge-dependent; only differences computed within one
  parameter set are meaningful, and no comparison here crosses models.
