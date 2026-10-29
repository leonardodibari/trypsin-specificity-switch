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
plausibility. A fifth, used in §6.7, places a design relative to the family rather
than to the wild type: the distance to the nearest natural sequence that already
carries the target residue at the critic site,

$$
d^{\mathrm{S}}_{\min}(\sigma) = \frac{1}{|\mathcal{A}|}
\min_{\tau \in \mathcal{N}_{\mathrm{S}}} \sum_{i \in \mathcal{A}} \mathbb{1}[\sigma_i \neq \tau_i],
\qquad \mathcal{N}_{\mathrm{S}} = \{\tau \text{ natural} : \tau_{i^\star} = \mathrm{S}\},
$$

restricted to the mutable columns $\mathcal{A}$ and expressed as a fraction of
them, since a design equals the wild type everywhere else.

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

### 6.6 Writing the serine in: the substituted ensembles

The bias of §4.1 acts on the *conditional* $P(\mathrm{S} \mid i^\star, \sigma)$,
which by construction does not depend on $\sigma_{i^\star}$: the chain is pushed to
build a context that wants a serine, not to place one. A chain's final sequence
therefore still carries whatever residue it last accepted at $i^\star$, and at
$\lambda = 0$ that is almost always the wild-type aspartate. To describe sequences
that *are* chymotrypsin-like at the critic site, $\sigma_{i^\star}$ is set to S in
every final sequence and the four observables of §5 are recomputed:

```bash
python scripts/substitute_critic.py \
    --ensemble-dir ../results/ensembles \
    --output ../results_serine/ensembles \
    --wt data/wt_trypsin.fasta --model ../bmDCA_params.dat \
    --critic-col 197 --device cuda --cache-model

python scripts/plot_ensemble.py \
    --ensemble-dir ../results_serine/ensembles --mu ../results/calibration/mu.json \
    --natural-observables ../results/calibration/natural_observables.csv \
    --max-points 1000 \
    --title "Final sequences of 1000 chains per lambda, with S written into the critic site" \
    --save docs/ensemble_pairs_serine.png
```

The substitution is applied to the sampled sequences only. The natural alignment
and the calibration of $\mu^\star = 307.6$ are those of §3 unchanged, so the grey
cloud and the quality scale are common to both figures.

Writing $\sigma' = \sigma$ with $\sigma'_{i^\star} = \mathrm{S}$:

| $\lambda$ | $\sigma_{i^\star} = \mathrm{S}$ already | $\langle E\rangle$ | $\langle \Delta E\rangle$ | $\langle d_H\rangle$ | $\langle S_{\mathrm{CDE}}\rangle$ | $\langle Q_{\mu^\star}\rangle$ (sd) | $\langle \Delta Q\rangle$ |
|---|---|---|---|---|---|---|---|
| 0 | 0.7 % | 230.1 | **+5.27** | 6.4 | 0.776 | −8.7 (4.2) | +2.47 |
| 10 | 16.5 % | 255.5 | +0.70 | 10.2 | 0.822 | 2.6 (4.9) | −2.02 |
| 50 | 52.1 % | 278.5 | −0.86 | 15.2 | 0.869 | 11.2 (4.8) | −1.70 |
| 100 | 64.2 % | 290.7 | −0.91 | 17.7 | 0.887 | 17.9 (4.5) | −1.37 |
| 300 | 76.3 % | 313.8 | −0.84 | 22.2 | 0.920 | 30.8 (3.9) | −1.02 |
| 1000 | 84.2 % | 333.4 | −0.63 | 25.9 | 0.942 | 43.5 (2.5) | −0.71 |

$P_{\mathrm{S}}$ is unchanged to machine precision at every $\lambda$
($\max_{\mathrm{chains}} |\Delta P_{\mathrm{S}}| = 0$), which is the numerical
confirmation of the invariance argued above; the $P_{\mathrm{S}}$ column of §6.3
therefore applies to the substituted ensembles as it stands, including the 100 %
of chains at $\lambda = 1000$ that satisfy the design constraint. The Hamming
distance moves by at most one, upward where the chain carried the wild-type D and
downward where it had already drifted off it.

**The serine is essentially free wherever the bias has done its work.** The per
chain differences are in `docs/critic_substitution_deltas.png`:

```bash
python scripts/plot_critic_deltas.py \
    --ensemble-dir ../results/ensembles \
    --substituted-dir ../results_serine/ensembles \
    --mu ../results/calibration/mu.json \
    --save docs/critic_substitution_deltas.png
```

Chains that already ended on S have $\Delta E = \Delta Q = 0$ exactly, and their
share rises from 0.7 % at $\lambda = 0$ to 84.2 % at $\lambda = 1000$; the
histograms are drawn over the remainder, whose differences are not degenerate.
Over those chains:

| $\lambda$ | median $\Delta E$ | mean $\Delta E$ | median $\Delta Q$ | mean $\Delta Q$ |
|---|---|---|---|---|
| 0 | +5.37 | +5.31 | +2.52 | +2.48 |
| 10 | +1.31 | +0.84 | −2.55 | −2.42 |
| 50 | −1.36 | −1.80 | −2.90 | −3.55 |
| 100 | −2.06 | −2.53 | −2.97 | −3.84 |
| 300 | −3.34 | −3.55 | −3.26 | −4.30 |
| 1000 | −3.68 | −3.98 | −3.42 | −4.47 |

At $\lambda = 0$ the substitution is a genuine penalty: the unbiased ensemble sits
in a trypsin context, and forcing a serine into it costs $+5.3$ in energy and
$+2.5$ in quality, with the $\Delta E$ distribution entirely to the right of zero.
From $\lambda = 50$ upward the sign reverses — the substitution *lowers* the energy
of the chains it touches, by up to $-4.0$ on average at $\lambda = 1000$. Those are
chains whose context already prefers serine (that is what $P_{\mathrm{S}} > 0.5$
means) but which happened to be caught on another residue at the moment sampling
stopped; writing the S in moves them onto the residue their own context favours.
The ensemble-level shift is correspondingly small, at most $-0.9$ in $\langle E
\rangle$, because the majority of chains do not move at all.

**The conclusions of §6.3 are unaffected.** $\langle Q \rangle$ changes by at most
$2.5$ against a natural spread of $15.3$: the $\lambda = 1000$ ensemble still sits
at $43.5$, more than two standard deviations above the natural family, and the
ordering of the ensembles along every axis is preserved. Installing the serine is
not what makes these designs atypical — the context built to accept it is.

### 6.7 Distance to the natural sequences that already have the serine

Energy says how plausible the model finds a design; it does not say whether
anything like it exists. The fifth quantity of the pairwise figure answers the
second question: the Hamming distance from each design to the closest member of
the natural alignment **restricted to the sequences that already carry S at the
critic column** -- 8728 of the 82 021 sequences, 10.6 %.

The distance is measured **over the mutable columns only**, and reported as a
percentage of them. Outside those 36 columns a design is the wild type by
construction, so including the other 224 would add one large constant -- the wild
type's own distance from the family, about 70 substitutions -- to every design
alike, and bury the differences between them under it. Within the pocket the
quantity is the one that matters: of the positions the sampler was free to
choose, what fraction did it set differently from the closest natural sequence
that already has the serine?

```bash
python scripts/add_natural_neighbour.py \
    --ensemble-dir ../results_serine/ensembles \
    --natural ../iter_aln_dedup.faa --critic-col 197 \
    --allowed-cols 180-220 --wt data/wt_trypsin.fasta --as-percentage \
    --natural-observables ../results/calibration/natural_observables.csv --device cuda

python scripts/plot_ensemble.py \
    --ensemble-dir ../results_serine/ensembles --mu ../results/calibration/mu.json \
    --natural-observables ../results/calibration/natural_observables.csv \
    --max-points 1000 --quantities E_DCA hamming_to_wt P_S Q min_hamming_natural_S_pct \
    --title "Final sequences of 1000 chains per lambda, with S written into the critic site" \
    --save docs/ensemble_pairs_serine.png
```

The distance is computed as the number of compared columns minus the number of
agreements, one matrix product per chunk of queries. A natural sequence that is
itself in the subset would be at distance zero from its own copy, so its own row
is excluded from its minimum. Values are multiples of $1/36$, which is why the
panels show a staircase.

| $\lambda$ | 0 | 10 | 50 | 100 | 300 | 1000 | *natural* |
|---|---|---|---|---|---|---|---|
| median | 22.2 % | 33.3 % | 41.7 % | 44.4 % | 50.0 % | 52.8 % | *38.9 %* |
| range | 16.7–36.1 | 22.2–47.2 | 30.6–52.8 | 36.1–52.8 | 44.4–55.6 | 47.2–58.3 | *0–69.4* |

**The designs walk away from the natural solution, and overtake it.** The wild
type's own pocket is already 22 % away from the nearest natural S-carrier -- 8 of
36 positions -- and every step up in $\lambda$ adds to that, monotonically, up to
52.8 % at $\lambda = 1000$: of the positions it was free to choose, the most
strongly biased ensemble sets more than half differently from the closest natural
sequence that solves the same problem.

The crossing point is what makes this more than a restatement of
$\langle d_H \rangle$. A typical natural sequence sits 38.9 % from its nearest
S-carrying relative over these same columns, so the $\lambda \leq 10$ ensembles
are *closer* to the natural S-carriers than natural sequences typically are to
each other, while from $\lambda \approx 50$ upward the designs are further. The
bias does not steer the pocket toward the arrangement the family uses; past a
certain strength it steers away from it, and the ensembles that satisfy the
specificity constraint are the ones furthest from any natural precedent -- the
same conclusion the sample quality reaches in §6.3, by an independent route that
never mentions the model.

---

## 7. Does unbiased resampling reach the training statistics?

Everything above takes the model as given. This section asks whether the model
reproduces the statistics it was fitted on, which is the generative check of a
Potts model and is independent of the design question.

### 7.1 What is compared

A bmDCA model is fitted by matching the reweighted one- and two-point frequencies
of the training alignment. If the fit succeeded *and* the sampler equilibrates,
sequences resampled at $\lambda = 0$, $\beta = 1$ must reproduce

$$f_i(a), \qquad f_{ij}(a,b), \qquad C_{ij}(a,b) = f_{ij}(a,b) - f_i(a) f_j(b) \;\; (i \neq j),$$

the last being the connected correlations, whose diagonal blocks are excluded
because $f_{ii}(a,b) = f_i(a) \delta_{ab}$ is reproduced by construction and would
flatter any sampler. Weights come from `bmDCA_weights.dat`, one per alignment row: they are
reciprocals of integers between 1 and 480, i.e. the usual
$w_a = 1 / |\{b : \mathrm{id}(a,b) > \theta\}|$ reweighting that downweights dense
clusters of close homologues. They sum to
$M_{\mathrm{eff}} = 25\,488$, 31.1 % of $M = 82\,021$; Kish's effective sample
size $(\sum w)^2 / \sum w^2 = 33\,847$ is a different and larger number, and is
not what "M_eff" means here.

**Every quantity in this section is weighted by those weights and computed over
all 82 021 sequences**: the frequencies $f_i$ and $f_{ij}$, and the natural means
of §7.4. The chains' own frequencies $p_i$, $p_{ij}$ are *un*weighted, as they must
be -- the chains are i.i.d. draws from the model, carrying no weights. The
frequencies and the Pearson coefficients are computed by
`adabmDCA.stats.get_freq_single_point`, `get_freq_two_points` and
`get_correlation_two_points`, the last of which extracts the off-diagonal blocks
itself. No pseudo-count is used.

The finite-sample ceiling of §7.3 draws its 1000 natural sequences with
probability proportional to $w$, so the draw follows the same distribution the
reference frequencies describe, and then compares them unweighted, exactly as a
batch of chains is compared. (The draw is without replacement, so inclusion
probabilities are only approximately proportional to $w$; at 1.2 % of the
alignment the distortion is negligible.)

**Sections 3 and 6 do not use these weights.** The calibration of $\mu$, the
natural observables behind the grey cloud, the natural band
$Q = 9.2 \pm 15.3$ and the natural median of §6.7 were all computed *unweighted*
on a random subsample of 20 000 sequences, before the weights file was available.
That is an inconsistency between the two halves of this document, and it is worth
stating that it was checked rather than assumed: recomputing them under the
weights moves $\mu^\star$ from 307.63 to 307.45 ($-0.1$ %), the natural band from
$9.22 \pm 15.30$ to $9.17 \pm 15.31$, the mean natural energy from 364.9 to 363.1
and the natural median of §6.7 not at all (38.9 %). No conclusion in §6 depends on
the difference.

Note that matching $f_i$ and $f_{ij}$ *implies* matching the mean energy, since
$E$ is linear in exactly those quantities. The mean energy of a batch is therefore
not an independent check but a sensitive read-out of the same one.

```bash
python scripts/run_generative_check.py \
    --wt data/wt_trypsin.fasta --model ../bmDCA_params.dat \
    --natural ../iter_aln_dedup.faa --weights ../bmDCA_weights.dat \
    --output ../results_generative_gaps/ --critic-col 197 --allowed-cols 0-259 \
    --n-chains 1000 --n-steps 520000 --trace-every 8000 --trace-schedule log \
    --seed 0 --device cuda --cache-model
```

1000 chains, $5.2 \times 10^5$ proposed moves each (2000 sweeps over all 260
columns), ≈ 10 min on one RTX 3090.

### 7.2 The gap must be an ordinary state

The design sampler of §4 never proposes a gap and never mutates a gap column, so a
chain keeps the gap pattern it starts from. That is the right choice for a design
run -- the wild type's alignment is not what is being redesigned -- but it is
fatal here: the training alignment is 11.6 % gaps, and a sampler confined to one
gap pattern explores a subspace of the space the model was fitted on. It cannot
reach the right equilibrium, and the failure is not subtle. Running the check both
ways, at identical settings (`--no-gap-moves` for the restricted version):

| start | gap | $f_i$ | $f_{ij}$ | $C_{ij}$ | slope | $\langle E\rangle$ | $\langle d_H^{\mathrm{WT}}\rangle$ |
|---|---|---|---|---|---|---|---|
| wild type | inert | 0.788 | 0.657 | 0.105 | 0.04 | 313.9 | 81 |
| wild type | ordinary state | **0.976** | **0.951** | **0.558** | 0.65 | 435.9 | 152 |
| natural | inert | 0.969 | 0.944 | 0.616 | 0.62 | 433.0 | 156 |
| natural | ordinary state | **0.993** | **0.983** | **0.601** | 0.59 | 442.5 | 167 |

With the gap inert, chains started at the wild type never leave the trypsin basin:
81 substitutions from the wild type after 2000 sweeps, against a family that is
160 away. Freeing the gap raises that to 152 and lifts the single-site agreement
from 0.79 to 0.98. **All results below use the free gap**; the restricted rows are
kept only as the demonstration that the restriction, not the model, was what the
earlier numbers measured.

### 7.3 The finite-sample ceiling

A batch of $M$ chains estimates $f_{ij}$ from $M$ samples, and that estimate is
noisy. The Pearson coefficient a *perfect* sampler could reach at $M = 1000$ is
therefore below 1, and for the connected correlations, whose entries are small,
well below it. Drawing 1000 sequences from the natural alignment itself under the
same weights measures this directly (`--ceiling-repeats`, five draws):

| | $f_i$ | $f_{ij}$ | $C_{ij}$ | slope of $C_{ij}$ |
|---|---|---|---|---|
| ceiling at $M = 1000$ | 0.9988 | 0.9960 | **0.848** ± 0.008 | 1.000 ± 0.029 |

Every coefficient must be read against that row: 0.85, not 1, is what success
looks like for $C_{ij}$ with a thousand chains. The row is specific to $M = 1000$
and rises with $M$ (§7.5 measures it at 10 000), so a Pearson coefficient is only
meaningful beside the ceiling for its own batch size.

### 7.4 Result

Runs identical but for where the chains start: all at the wild type, one natural
sequence each drawn under the weights (`--start natural`; with the gap a free state
these are used unmodified), or one uniformly random sequence each over all 21
states (`--start random`).

| sampler | start | sweeps | $f_i$ | $f_i$ ng | $f_{ij}$ | $f_{ij}$ ng | $C_{ij}$ | $C_{ij}$ ng | slope | $\langle E\rangle$ | $\langle d_H^{\mathrm{WT}}\rangle$ | $\langle S_{\mathrm{CDE}}\rangle$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Metropolis | wild type | 2000 | 0.976 | 0.979 | 0.951 | 0.955 | 0.558 | 0.501 | 0.65 | 435.9 | 151.7 | 1.317 |
| Metropolis | wild type | 8000 | 0.990 | 0.990 | 0.978 | 0.976 | 0.562 | 0.494 | 0.55 | 449.9 | 168.7 | 1.369 |
| Metropolis | natural | 2000 | 0.993 | 0.993 | 0.983 | 0.982 | 0.601 | 0.536 | 0.59 | 442.5 | 167.1 | 1.342 |
| **Gibbs** | **wild type** | **2000** | **0.990** | **0.989** | **0.977** | **0.975** | **0.548** | **0.478** | **0.54** | **449.3** | **168.9** | **1.364** |
| **Gibbs** | **natural** | **2000** | **0.991** | **0.991** | **0.980** | **0.979** | **0.568** | **0.499** | **0.55** | **450.0** | **168.8** | **1.367** |
| **Gibbs** | **random** | **2000** | **0.990** | **0.990** | **0.978** | **0.976** | **0.540** | **0.472** | **0.52** | **453.9** | **169.4** | **1.381** |
| *target* | — | — | *0.999* | *0.999* | *0.996* | *0.995* | *0.848* | *0.826* | *1.000* | *361.8* | *160.4* | *1.147* |

**Five runs, two samplers, three starting points, one answer.** The three Gibbs
runs span mean energies of 449.3, 450.0 and 453.9 and mean distances from the wild
type of 168.9, 168.8 and 169.4 -- and they reach that common point from three
different directions. The wild-type chains climb from $E = 220$, the natural chains
rise from 362, and the random chains fall from 1197, a starting ensemble with
$C_{ij} = 0.0001$ and 248 substitutions from the wild type, carrying none of the
family's structure whatever. All three traces fluctuate without trend over the last
decade of steps. The Metropolis runs land in the same place, 449.9 from the wild
type at 8000 sweeps, which is the agreement between an exact conditional sampler
and a Metropolis chain that the equilibrium requires and that neither sampler
could fake alone.

The random start matters because it is the only one that is not already inside the
family. Chains begun at the wild type or at natural sequences are begun at a point
the model likes, so a failure to mix would hold them near it -- and would hold both
near it in the same direction. Random sequences begin 743 in energy *above* the
equilibrium and arrive at it from above. There is no remaining reading under which
the ensemble is merely stuck where it started.

Gibbs reaches that equilibrium about four times faster per site update, as the
figure shows, but it does not reach a *different* one. Nothing in the answer
depended on the sampler.

**That equilibrium is not the natural alignment.** The weighted natural mean is
361.8, some 88 below where every run settles, and since $E$ is linear in $f_i$ and
$f_{ij}$ this is the frequency comparison restated on a scale with units. The one-
and two-point frequencies come close to what 1000 chains permit -- $f_i$ at 0.990
of a possible 0.999, $f_{ij}$ at 0.978 of 0.996 -- but the connected correlations
fall far short of theirs, and the **slope of 0.54** rather than 1 says what the
shortfall is: the resampled correlations are systematically *weaker* than the
natural ones by about 46 %. The model has learnt the single-site composition of
the family well, most of the pair composition, and about half of the correlation
structure that is the whole point of a Potts model.

The slope, not the Pearson coefficient, is the quantity to quote. Section 7.5
shows why: `_get_slope` regresses the chains' $C_{ij}$ on the natural $C_{ij}$
with the *natural* values as the predictor, so noise in the chains' estimate
leaves it unbiased, and it is the same at 1000 chains and at 10 000. The Pearson
coefficient is attenuated by that noise and moves with $M$.

Three controls show that this is the model and not the measurement. Quadrupling the
Metropolis run (2000 to 8000 sweeps) lifted $f_i$ from 0.976 to 0.990 and left
$C_{ij}$ at 0.56, so the shortfall is not a convergence artefact -- more sampling
fixes the composition and does not touch the correlations. And the natural-start
Gibbs run begins, after twelve site updates, at $C_{ij} = 0.832$ against a ceiling
of 0.848: the data itself sits where it should, and it is the model's own
dynamics, not the statistic, that carries the ensemble away from it. The random
start closes the last gap, arriving at $C_{ij} = 0.54$ from $C_{ij} = 0.0001$:
the equilibrium is reached from below as well as from above, and sits in the same
place either way.

**What this establishes.** The earlier caveats are closed. The gap is a free state
(§7.2), both samplers equilibrate and agree, three independent starting ensembles
converge on one answer, and the finite-sample ceiling is measured rather than
assumed (§7.3). What remains is a property of the parameters:
at $\beta = 1$ this model generates sequences about 88 in energy above its own
training set, reproducing its composition but under-reproducing its connected
correlations by roughly 46 % -- both figures confirmed at ten times the batch size
in §7.5. For the design work of §6 the consequence is
specific -- energies and conditionals computed from these parameters are being
read off a model whose correlation structure is softer than the family's, which is
the direction that makes designs look more plausible than they are.

The figure is `docs/generative_check.png`:

```bash
# adabmDCA's own sampler, from each starting point
python scripts/run_generative_check.py ... --sampler gibbs --output ../results_gibbs/
python scripts/run_generative_check.py ... --sampler gibbs --start natural \
    --output ../results_gibbs_natural/
python scripts/run_generative_check.py ... --sampler gibbs --start random \
    --output ../results_gibbs_random/

python scripts/plot_generative_check.py \
    --run-dir ../results_gibbs ../results_gibbs_natural ../results_gibbs_random \
              ../results_generative_gaps_long \
    --log-x --no-nogap --figsize 15 8.6 \
    --labels "Gibbs, started at the wild type" \
             "Gibbs, started at natural sequences" \
             "Gibbs, started at random sequences" \
             "Metropolis, started at the wild type" \
    --save docs/generative_check.png
```

---

### 7.5 Ten thousand chains: which numbers are properties of the model

Every Pearson coefficient above is read against a ceiling measured at $M = 1000$.
Raising $M$ raises the ceiling, and the question is which of the measured numbers
follow it. Two Gibbs runs were repeated at $M = 10\,000$, identical in every other
respect, from the two extreme starts.

| | $M = 1000$ | $M = 10\,000$ | ceiling at 1000 | ceiling at 10 000 |
|---|---|---|---|---|
| $f_i$ | 0.990 | 0.991 | 0.9988 | 0.9999 |
| $f_{ij}$ | 0.977 | 0.981 | 0.9960 | 0.9997 |
| **Pearson $C_{ij}$** | **0.548** | **0.716** | **0.848** | **0.986** |
| **slope of $C_{ij}$** | **0.539** | **0.538** | **1.000** ± 0.029 | **0.995** ± 0.006 |
| $\langle E\rangle$ | 449.3 | 450.5 | — | *361.8* |
| $\langle d_H^{\mathrm{WT}}\rangle$ | 168.9 | 169.1 | — | *160.4* |

(wild-type start; the random start at $M = 10\,000$ gives $C_{ij} = 0.711$,
slope 0.536, $\langle E\rangle = 451.1$ -- the two starts now agree to 0.6 in
energy and 0.005 in $C_{ij}$, tighter than at $M = 1000$ as the smaller batch
noise predicts.)

**The ceiling moves as a $1/M$ noise model says it should.** Treating the Pearson
coefficient as attenuated by estimator noise, $\rho^2 \approx S/(S + N/M)$,
the $M = 1000$ ceiling of 0.848 implies $N/S = 0.391$ and so a ceiling of 0.981 at
$M = 10\,000$. The measured value is 0.986. The ceiling is doing exactly the job
§7.3 claims for it.

**The energy gap is a property of the model.** 449.3 against 450.5, with the
weighted natural mean at 361.8 either way. Ten times the sampling does not close
88 in energy, and was never going to: $\langle E\rangle$ is an average over
chains, not a correlation between noisy estimates, so batch size affects its error
bar and not its value.

**The Pearson coefficient of $C_{ij}$ is not.** It rises from 0.548 to 0.716 --
a third of the apparent shortfall at $M = 1000$ was noise in the chains' own
$C_{ij}$ estimate. Worse for the earlier framing, the ratio to the ceiling is not
stable either: $0.548/0.848 = 0.65$ against $0.716/0.986 = 0.73$. Dividing by the
ceiling is *not* a valid disattenuation, and §7.4's original phrasing -- "0.55 of
a possible 0.848" -- should not be read as an $M$-independent statement. It has
been rewritten.

**The slope is.** 0.539 at $M = 1000$, 0.538 at $M = 10\,000$: unchanged in the
third decimal while its own ceiling sits at 1.00 in both. This is structural rather
than lucky. `adabmDCA.stats._get_slope(x, y)` computes
$\mathrm{cov}(x,y)/\mathrm{var}(x)$ with $x$ the **natural** $C_{ij}$ and $y$ the
chains'. Classical errors-in-variables attenuation applies to noise in the
*predictor*; here the predictor is the full reweighted alignment, whose noise is
negligible beside a 1000- or 10 000-sample batch, and noise in the response $y$
inflates the residual variance without biasing the coefficient. Hence a ceiling of
1.000 at both $M$, and hence a measured 0.54 that means what it says.

**The conclusion of §7.4 stands, on a firmer number.** At $\beta = 1$ this model
equilibrates about 88 in energy above its training set and generates connected
correlations about 46 % too weak, and neither figure is an artefact of batch size.
What changes is which statistic carries the claim: quote the slope, 0.538, and
report the Pearson coefficient only beside the ceiling for its own $M$.

The figure is `docs/generative_check_10k.png`:

```bash
for START in wt random; do
  python scripts/run_generative_check.py ... --sampler gibbs --n-chains 10000 \
      --start $START --output ../results_gibbs10k_$START/
done

python scripts/plot_generative_check.py \
    --run-dir ../results_gibbs10k_wt ../results_gibbs10k_random \
    --log-x --no-nogap --figsize 15 8.6 \
    --labels "Gibbs, 10000 chains, started at the wild type" \
             "Gibbs, 10000 chains, started at random sequences" \
    --save docs/generative_check_10k.png
```

---

## 8. Reproducibility

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

---

## 9. The same analysis with the `model_pcd_serine_protease` parameters

A second model of the same family (adabmDCA `bmDCA`, Metropolis PCD, 10 000 chains,
stopped at its own `target_pearson = 0.95`). Energies are gauge-dependent, so none
of the numbers in §6 and §7 transfer; everything below is recomputed from scratch.
The calibration gives $\mu^* = 316.53$ (§3 method unchanged, 20 000 sequences,
unweighted): it removes 94.7 % of the variance of the energy, with
$\mathrm{corr}(E, \mathrm{CDE}) = 0.973$. The figure is
`docs/mu_calibration_pcd.png`.

### 9.1 Generative check

Two runs of 25 000 Gibbs chains, 4000 sweeps, every column free and the gap an
ordinary state, started from uniformly random sequences and from natural sequences.

| start | $f_i$ | $f_{ij}$ | $C_{ij}$ | slope | $\langle E\rangle$ |
|---|---|---|---|---|---|
| random (from $E = 1199$) | 0.9926 | 0.9846 | 0.7620 | 0.595 | 373.5 |
| natural (from $E = 289$) | 0.9928 | 0.9850 | 0.7612 | 0.593 | 373.6 |
| *ceiling at $M = 25\,000$* | *1.0000* | *0.9999* | *0.9964* | *0.998* | *289.0* |

The two starts agree to 0.1 in energy and 0.0008 in $C_{ij}$ having approached from
910 apart, so this is equilibrium, not a plateau. At $M = 25\,000$ the finite-sample
ceiling is 0.996 and no longer limits the reading (§7.5). The conclusion matches
§7.4 on a different parameter set: composition reproduced, connected correlations
about 40 % too weak, equilibrium some 85 in energy above the training set.

**The model's own equilibrium sample.** `chains.fasta` holds the 10 000 persistent
chains at the end of training. Scored through this repository's `NaturalStatistics`
against the same alignment and `weights.dat`, with the finite-sample ceiling
measured at the same $M = 10\,000$:

| | $f_i$ | $f_{ij}$ | $C_{ij}$ | slope | $\langle E\rangle$ |
|---|---|---|---|---|---|
| training chains ($M = 10\,000$) | 0.9933 | 0.9859 | **0.7550** | 0.576 | 376.4 ± 53.6 |
| *ceiling at $M = 10\,000$* | *0.9999* | *0.9996* | *0.9797* | *1.004* | *289.0* |
| 25 000 Gibbs chains from random | 0.9926 | 0.9846 | 0.7620 | 0.595 | 373.5 ± 56.0 |

Excluding the gap state throughout gives 0.7184 with slope 0.504. The training
chains and an independent run of this repository's sampler therefore describe the
same distribution — 0.755 against 0.762 in $C_{ij}$, 376.4 against 373.5 in energy,
from different code and opposite starting points.

**A discrepancy worth recording.** The training log reports `Pearson = 0.950027`
for those very chains. Rebuilding its own `fij_target` with `DatasetDCA` and
comparing with `get_correlation_two_points` reproduces the log's
`M_eff = 25888.93359` and `effective_pseudocount = 3.862654274e-05` exactly, so it
is the same target, and still gives 0.755; the result is unchanged in float64, and
`train_graph` saves the very chains the metric was computed on. So `params.dat`,
`chains.fasta` and `weights.dat` agree with each other and with this repository's
sampler; the logged 0.95 is the outlier. The log records `package_version: 1.0.0`
against 2.0.0 installed, which is where to look. Since `target_pearson` is the
stopping rule, training may have stopped early — which is consistent with the
measured 0.755 sitting well below the 0.980 that 10 000 samples could reach.

The figure is `docs/generative_check_pcd.png`.

### 9.2 Designs

The same six lambdas as §6, 1000 chains each, 60 000 proposed moves,
`--allowed-cols 180-220` (36 mutable columns after dropping the wild type's gaps),
S written into the critic site afterwards as in §6.6.

| $\lambda$ | $\langle E\rangle$ | $\langle d_H^{\mathrm{WT}}\rangle$ | $\langle P_S\rangle$ | $P_S > 0.8$ | $\langle Q_\mu\rangle$ | median distance to nearest natural S-carrier |
|---|---|---|---|---|---|---|
| 0 | 155.0 | 6.4 | 0.005 | 0 % | −94.3 | 25.0 % |
| 10 | 179.8 | 10.2 | 0.138 | 0 % | −84.0 | 33.3 % |
| 50 | 203.6 | 15.4 | 0.480 | 0 % | −75.9 | 41.7 % |
| 100 | 216.8 | 18.1 | 0.611 | 0 % | −68.6 | 44.4 % |
| 300 | 242.1 | 23.0 | 0.748 | 0 % | −53.0 | 50.0 % |
| 1000 | 266.4 | 27.3 | 0.796 | 23.6 % | −34.9 | 55.6 % |
| *natural* | *289.6* | *161.0* | *0.110* | — | *−79.1 ± 16.5* | *38.9 %* |

The same trade-off as §6, on different parameters, and monotone in $\lambda$ down
every column: $P_S$ is bought with energy and with distance from the family. At
$\lambda = 1000$ the designs are closer to the natural median in energy than the
$\lambda = 0$ chains are, but their quality $Q_\mu$ has left the natural band
entirely (−34.9 against −79.1 ± 16.5), and their nearest natural S-carrier is
55.6 % of the mutable sites away, against a natural median of 38.9 %. $\lambda =
1000$ is also the only setting that puts any chains above $P_S = 0.8$ (23.6 %);
at $\lambda = 300$ the mean $P_S$ is already 0.748 but no single chain clears the
threshold, so the ladder buys its last tenth of specificity at a steep price.

Writing S into the critic site, over the chains not already carrying it, costs a
median of +5.3 in energy at $\lambda = 0$ and becomes progressively favourable as
the bias rebuilds the surrounding pocket for it: +1.3 at $\lambda = 10$ (837 chains),
−1.5 at 50 (506), −2.3 at 100 (370), −3.3 at 300 (262) and −3.6 at 1000 (189).

The figures are `docs/ensemble_pairs_pcd.png` (5×5),
`docs/ensemble_trace_pcd.png` and `docs/critic_substitution_deltas_pcd.png`, the
last being the per-chain cost of that substitution (§6.6) on this model: at
$\lambda = 0$ the distribution sits wholly above zero, at $\lambda = 10$ it straddles
it, and from $\lambda = 50$ up it is wholly below.

```bash
python scripts/calibrate_mu.py --natural ../iter_aln_dedup.faa \
    --model ../model_pcd_serine_protease/params.dat --output ../results_pcd/calibration/ \
    --max-sequences 20000 --wt data/wt_trypsin.fasta --critic-col 197 --device cuda --cache-model

python scripts/run_ensemble.py --wt data/wt_trypsin.fasta \
    --model ../model_pcd_serine_protease/params.dat --output ../results_pcd/ensembles/ \
    --critic-col 197 --allowed-cols 180-220 --lambdas 0 10 50 100 300 1000 \
    --n-chains 1000 --n-steps 60000 --trace-every 1000 --trace-schedule log \
    --trace-lambdas 0 100 1000 --device cuda --cache-model

python scripts/substitute_critic.py --ensemble-dir ../results_pcd/ensembles \
    --output ../results_pcd/ensembles_serine --wt data/wt_trypsin.fasta \
    --model ../model_pcd_serine_protease/params.dat --critic-col 197 --device cuda --cache-model

python scripts/add_natural_neighbour.py --ensemble-dir ../results_pcd/ensembles_serine \
    --natural ../iter_aln_dedup.faa --critic-col 197 \
    --natural-observables ../results_pcd/calibration/natural_observables.csv \
    --allowed-cols 180-220 --wt data/wt_trypsin.fasta --as-percentage --device cuda

python scripts/plot_ensemble.py --ensemble-dir ../results_pcd/ensembles_serine \
    --mu ../results_pcd/calibration/mu.json \
    --natural-observables ../results_pcd/calibration/natural_observables.csv \
    --quantities E_DCA hamming_to_wt P_S Q min_hamming_natural_S_pct --max-points 1000 \
    --title "Final sequences of 1000 chains per lambda, with S written into the critic site" \
    --save docs/ensemble_pairs_pcd.png

python scripts/plot_ensemble_trace.py --ensemble-dir ../results_pcd/ensembles \
    --mu ../results_pcd/calibration/mu.json \
    --natural-observables ../results_pcd/calibration/natural_observables.csv \
    --log-x --save docs/ensemble_trace_pcd.png

for START in random natural; do
  python scripts/run_generative_check.py --wt data/wt_trypsin.fasta \
      --model ../model_pcd_serine_protease/params.dat --natural ../iter_aln_dedup.faa \
      --weights ../model_pcd_serine_protease/weights.dat --critic-col 197 --allowed-cols 0-259 \
      --sampler gibbs --n-chains 25000 --n-steps 1040000 --trace-every 8000 \
      --trace-schedule log --device cuda --cache-model --start $START \
      --output ../results_pcd25k_$START/
done
```


## 10. Cooling the Potts term: what $\beta$ buys

A suggestion from the group: put a $\beta$ in front of $E_{\mathrm{DCA}}$ and tune
it so the designs keep an energy, or a quality $Q_\mu$, comparable to the wild
type's, in the spirit of Calvanese *et al.* on RNA. The Hamiltonian already had the
$\beta$; what it did not have was the right scope for it.

### 10.1 The critic is always at $\beta = 1$

Both samplers used to evaluate the critic's conditional at the sampling
temperature, $\log P_\beta(S \mid i_{\mathrm{critic}}, \sigma)$. That makes $\beta$
sharpen the bias as well as cool the energy, and the two effects are impossible to
separate: a first scan appeared to improve the energy *and* the specificity at
once, which was an artefact of the softmax getting sharper. The critic is now
pinned at inverse temperature 1 (`CRITIC_BETA` in both `src/mcmc.py` and
`src/ensemble.py`), so

$$H(\sigma) = \beta\, E_{\mathrm{DCA}}(\sigma) - \lambda \log P_{1}(S \mid i_{\mathrm{critic}}, \sigma).$$

At $\beta = 1$ the two conventions coincide, so §§6–9 are unaffected.

With the critic fixed, the sampled measure factorises exactly:

$$\exp(-\beta E + \lambda \log P_1) = \exp\!\big(-\beta\,[\,E - \tilde\lambda \log P_1\,]\big), \qquad \tilde\lambda = \lambda/\beta .$$

So the ensemble is Boltzmann at temperature $1/\beta$ for the modified energy
$E - \tilde\lambda \log P_1$: **$\tilde\lambda$ says what is being optimised, $\beta$
says only how cold**. The two are separate knobs, and a scan at fixed $\lambda$
moves both at once.

### 10.2 The grid

Nineteen ensembles, 1000 chains each, `--allowed-cols 180-220`, every one rescored
at $\beta = 1$ because $\mu$ was calibrated there and a $\beta \neq 1$ conditional is
not the same observable. Diversity is the mean pairwise Hamming distance *inside*
each library over the 36 mutable columns; the natural alignment scores 21.6 there.

| $\beta$ | $\lambda$ | $\tilde\lambda$ | $\langle E\rangle$ | $\langle Q_\mu\rangle$ | $\langle P_S\rangle$ | reach S alone | nearest natural S-carrier | unique | pairwise $d_H$ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 155.0 | −94.3 | 0.005 | 0.7 % | 25.0 % | 965 | 6.5 |
| 1 | 10 | 10 | 179.8 | −84.0 | 0.138 | 16.3 % | 33.3 % | 1000 | 11.0 |
| 1 | 50 | 50 | 203.6 | −75.9 | 0.480 | 49.4 % | 41.7 % | 1000 | 11.4 |
| 1 | 100 | 100 | 216.8 | −68.6 | 0.611 | 63.0 % | 44.4 % | 1000 | 10.8 |
| 1 | 300 | 300 | 242.1 | −53.0 | 0.748 | 73.8 % | 50.0 % | 981 | 7.4 |
| 1 | 1000 | 1000 | 266.4 | −34.9 | 0.796 | 81.1 % | 55.6 % | 466 | 3.7 |
| 1.25 | 1000 | 800 | 259.9 | −40.5 | 0.791 | 91.3 % | 52.8 % | 332 | 3.2 |
| 1.5 | 1000 | 667 | 255.0 | −44.3 | 0.787 | 95.9 % | 52.8 % | 223 | 3.0 |
| 2 | 1000 | 500 | 250.4 | −47.6 | 0.781 | 98.9 % | 52.8 % | 143 | 2.3 |
| 3 | 1000 | 333 | 244.7 | −50.6 | 0.770 | 100 % | 52.8 % | 172 | 2.5 |
| 5 | 1000 | 200 | 226.9 | −62.7 | 0.719 | 100 % | 47.2 % | 79 | 1.9 |
| 1.5 | 1500 | 1000 | 265.4 | −35.8 | 0.797 | 96.3 % | 55.6 % | 211 | 2.8 |
| 2 | 2000 | 1000 | 264.8 | −36.5 | 0.797 | 99.2 % | 55.6 % | 114 | 2.2 |
| 3 | 3000 | 1000 | 264.3 | −37.2 | 0.797 | 100 % | 55.6 % | 53 | 1.5 |
| 5 | 500 | 100 | 208.6 | −74.1 | 0.631 | 100 % | 41.7 % | 144 | 2.4 |
| 5 | 1500 | 300 | 241.6 | −53.3 | 0.763 | 100 % | 50.0 % | 74 | 2.0 |
| 10 | 1000 | 100 | 208.9 | −74.5 | 0.635 | 100 % | 41.7 % | 28 | 1.2 |
| 10 | 2000 | 200 | 225.9 | −62.8 | 0.717 | 100 % | 47.2 % | 13 | 1.0 |
| 10 | 3000 | 300 | 241.5 | −54.1 | 0.763 | 100 % | 50.0 % | 22 | 1.7 |
| *natural* | | | *289.6* | *−79.1 ± 16.5* | *0.110* | *10.6 %* | *38.9 %* | | *21.6* |
| *wild type* | | | *143.7* | *−100.0* | *0.008* | | | | |

Three figures:

* `docs/beta_scan_pcd.png` -- the grid in six panels. The top row is read against
  $\tilde\lambda$: energy, quality and specificity, with the natural mean, the wild
  type and the $\lambda = 0$ control ruled on each. The bottom row is read against
  the **energy**, which is the quantity the designs are meant to keep comparable to
  the wild type's: at a given energy, what specificity, what distance from the
  natural S-carriers, and what is left of the library. Circles joined by a solid
  line are $\beta = 1$, the thermal reference; diamonds are the cooled runs, joined
  by a dashed line from $\beta = 5$ up where $\beta = 5$ and $\beta = 10$ land on top
  of each other;
* `docs/beta_matched_pairs_pcd.png` -- the 5×5 at matched $\tilde\lambda = 100$ for
  $\beta = 1, 5, 10$. The cold libraries are a tight knot *inside* the warm one,
  which is the collapse seen per chain rather than as a summary statistic;
* `docs/beta_matched_trace_pcd.png` -- the same three along the run, against
  accepted moves per chain. All three overshoot to $E \approx 232$ before relaxing;
  the cold ones relax further, to 208 against 218, and their inter-quartile bands
  are visibly narrower. Note the axis: in 60 000 proposals $\beta = 1$ accepts some
  2500 moves per chain and $\beta = 10$ about 700.

### 10.3 What the grid says

**$\beta$ saturates, and it saturates early.** Hold $\tilde\lambda$ and compare
$\beta = 5$ with $\beta = 10$: at $\tilde\lambda = 100$, $E = 208.6$ against 208.9 and
$P_S = 0.631$ against 0.635; at 200, 226.9 against 225.9; at 300, 241.6 against
241.5. Identical to the third digit. By $\beta \approx 5$ the sampler has reached
its zero-temperature limit and the ensemble is a function of $\tilde\lambda$ alone.

**Cooling at a fixed bias changes essentially nothing.** Down the
$\tilde\lambda = 1000$ line, $\beta = 1 \to 3$ moves the energy from 266.4 to 264.3,
leaves $P_S$ at 0.797 to three digits and the distance to the nearest natural
S-carrier at 55.6 %. Meanwhile the unique sequences fall from 466 to 53 and the
pairwise diversity from 3.7 to 1.5. Two units of energy for a ninth of the library.

**At matched energy the gain is largest exactly where it is wanted.** The
professor's framing is to hold $E_{\mathrm{DCA}}$ near the wild type's, so the
comparison that answers it is: at the energy a cooled run reaches, what would
$\beta = 1$ have given? Interpolating along the thermal curve,

| cooled run | $\langle E\rangle$ | $P_S$ cooled | $P_S$ at $\beta = 1$ | gain | diversity cooled | at $\beta = 1$ |
|---|---|---|---|---|---|---|
| $\beta = 5$, $\tilde\lambda = 100$ | 208.6 | 0.631 | 0.530 | **+0.101** | 2.4 | 11.2 |
| $\beta = 5$, $\tilde\lambda = 200$ | 226.9 | 0.719 | 0.666 | +0.054 | 1.9 | 9.5 |
| $\beta = 5$, $\tilde\lambda = 300$ | 241.6 | 0.763 | 0.746 | +0.018 | 2.0 | 7.5 |

The gain grows as the energy falls toward the wild type: +0.02 at $E = 242$, +0.10
at $E = 209$, a fifth more specificity at the same energy. That is a point in the
suggestion's favour, and it is the low-energy end that the suggestion is about.
What it costs is the library: 2.4 against 11.2 in pairwise diversity at
$E = 209$. The distance to the nearest natural S-carrier is a wash (41.7 % against
an interpolated 42.7 %).

**At matched bias the gain is smaller.** Holding $\tilde\lambda$ instead, so that
cooling is the only difference:

| | $\langle E\rangle$ | $\langle Q_\mu\rangle$ | $\langle P_S\rangle$ | reach S alone | unique | pairwise $d_H$ |
|---|---|---|---|---|---|---|
| $\tilde\lambda = 100$, $\beta = 1$ | 216.8 | −68.6 | 0.611 | 63.0 % | 1000 | 10.8 |
| $\tilde\lambda = 100$, $\beta = 5$ | 208.6 | −74.1 | 0.631 | 100 % | 144 | 2.4 |

Cooling removes the thermal excess — 8 in energy, 5.5 in $Q_\mu$, both toward the
wild type — lifts $P_S$ by 0.02, brings the designs 2.7 points nearer the natural
S-carriers and makes every chain reach S on its own. It also leaves 144 distinct
sequences where there were 1000, at a pairwise diversity of 2.4 against the
natural 21.6. The $\beta = 1$ libraries at $\tilde\lambda = 10–100$ are already only
half as diverse as the natural alignment; the cold ones are one sequence with
jitter.

**Conclusion.** The suggestion is sound in its aim, the machinery now supports it,
and at the low-energy end it does buy a fifth more specificity at equal energy.
But the quantity that *places* the designs relative to the wild type is
$\tilde\lambda$, not $\beta$: $\beta$ moves the energy by at most a few tens while
$\tilde\lambda$ moves it by a hundred, and every unit of $\beta$ is paid for in
library diversity. To keep $Q_\mu$ inside
the natural band, lower $\tilde\lambda$ at $\beta = 1$: $\tilde\lambda = 50$ gives
$Q_\mu = -75.9$ against a natural $-79.1 \pm 16.5$, with $P_S = 0.48$, 1000 distinct
sequences and the diversity of the $\lambda = 0$ control. A mild $\beta = 1.5–2$ is
defensible if every chain must carry S without the substitution step; beyond
$\beta \approx 3$ the ensemble stops being a library.

One caveat. These are all wild-type starts, and the cold runs accept 0.2–0.6 % of
proposals, so they are quenches rather than equilibrium samples — which is exactly
what the $\beta = 5$/$\beta = 10$ agreement shows them to be. Their traces are flat
over the last factor 2.5 in steps, but a quench depends on where it started, and a
colder run explores less of the basin, not more of it. That is a further reason to
read the diversity column as the real cost.

```bash
python scripts/run_ensemble.py --wt data/wt_trypsin.fasta \
    --model ../model_pcd_serine_protease/params.dat --output ../results_pcd/beta_scan/b3/ \
    --critic-col 197 --allowed-cols 180-220 --lambdas 1000 --beta 3 \
    --n-chains 1000 --n-steps 60000 --trace-every 1000 --trace-schedule log \
    --device cuda --cache-model
# then substitute_critic.py --beta 1.0 and add_natural_neighbour.py as in 9.2,
# and finally
python scripts/plot_beta_scan.py --summary ../results_pcd/beta_scan_summary.csv \
    --save docs/beta_scan_pcd.png
```
