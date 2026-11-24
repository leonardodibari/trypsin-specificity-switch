# t-steering — the restraint applied across three mutation sets

**What a design costs at a chosen mutation budget, for each of the three
candidate sets, under a full factorial in (β, λ, t).**

---

## Why this subproject exists

`hamming-restraint/` asked whether the `t·d_H` term is worth having. The answer
it reached is in that README: `t` sets `⟨d⟩` monotonically, the
fluctuation–response identity `−∂⟨d⟩/∂t = Var(d)` holds to within 5 % on the
79-column block, and `t` moves the budget at ~4× less quality cost than cooling
and without collapsing the library.

This subproject asks the applied question that follows:

> Across the three candidate mutation sets, which `(β, λ, t)` actually delivers
> a design worth building — few mutations, wild-type-like quality, a diverse
> batch, and high `P(S)`?

## The grid

A full factorial, **3 sets × 2 β × 2 λ × 4 t = 48 runs**, 1000 chains and
60,000 steps each, gap moves on.

| set | columns | what it is |
|---|---|---|
| `segment` | 79 | the grafting window, 165–243 (3TGI 160–228) |
| `hedstrom14` | 14 | what Hedstrom, Szilagyi & Rutter built and assayed |
| `coupling20` | 20 | the 20 window columns most strongly coupled to the critic |

The sets are **not redefined here**. `analysis/config.py` reads them through
`hamming-restraint/analysis/config.py`, which reads them from the ladder's own
configuration, so "the 1992 construct" means here exactly what it means
everywhere else. `coupling-20` is **model-derived** and a copied constant would
be silently wrong rather than loudly wrong.

| knob | values | why these |
|---|---|---|
| `β` | 1, 1.5 | **not 2 or above.** CLAMP_CONTROL caveat 1: at `β ≥ 2` a chain is frozen rather than settled — the library falls to 89 % unique at `β = 2` and 7.9 % at `β = 10`, and `RESULTS.md` cannot tell a converged cold run from a stuck one. `β = 1.5` is the coldest rung that is still warm. |
| `λ` | 100, 300 | `P(S) ≈ 0.93` and `≈ 0.97` on the 79-column block. `λ = 10` reaches only 0.74, which fails the "high `P(S)`" requirement before the restraint is even applied. |
| `t` | 0, 0.5, 1, 2 | phase B's own grid. All 20 phase-A runs on the 79-column block saturated on it, and it spans `⟨d⟩ ≈ 17 → 8` there — the range a construct is commissioned in. |

**60,000 steps** for all three sets: the figure all 20 phase-A runs on the
79-column window saturated at, and the 14- and 20-column sets are a smaller
state space. No set here is the 265-column one, so the 400,000-step figure of
`run_phase_b.sh` does not apply — and neither does the equilibration problem
that figure exists for (`results/EQUILIBRATION_NOTE.md` is about all-free runs).

## The gate comes first

**Nothing is read off a run that has not saturated.** `analysis/s01_gate.py`
applies the drift test of `hamming-restraint/analysis/saturation.py` to
**`E_DCA`, `Q`, `hamming_to_wt` and `P_S` separately** — flattening in one says
nothing about the others, since `Q` is dominated by the energy and keeps falling
long after `P(S)` has settled. `data.load_all()` drops what fails and names it;
`s01_gate.py` **exits non-zero** unless all 48 pass, so a driver that chains the
figures cannot build a later one on an ungated run.

The second failure mode, which a drift test cannot see: a run that is flat
because it is **frozen** passes trivially — zero motion is zero drift. So the
accepted moves per mutable site is reported beside every verdict and anything
under 20 is marked stuck. That is also why `β ≥ 2` is absent from the grid, and
panel (b) of S1 is where that choice is checked rather than assumed.

## Choosing the six

Four requirements that pull against each other. `analysis/selection.py` applies
them as **admissibility thresholds, never as a score** — a weighted sum would
bury the trade-off under weights nobody chose and then report its own weighting
as a result. `figures_ladder/fig05_table.py` resolves the same problem the same
way.

| requirement | threshold | why |
|---|---|---|
| high `P(S)` | `≥ 0.90` | the grid's `λ` already clears this; the threshold checks the restraint has not eaten the specificity it was applied on top of |
| `Q` like the wild type | `≤ Q_WT` | read as *at least as plausible as the wild type under the model*, the form the ladder's dashed WT line uses. This is the one that bites. |
| diversity | `≥ 100 distinct` **or** `divergence ratio > 0.3` | either suffices. A unique-*percentage* floor is the wrong instrument: unique share is combinatorially capped by set size and budget, so at 16 columns and `⟨d⟩≈4` a thousand chains must collide — a 90 % floor admitted nothing from the red sector and would have selected `segment` four times over. The ratio is mean pairwise Hamming *within* the batch over mean distance to WT: scale-free, so it compares 16-, 20- and 79-column sets fairly. Healthy batches measure 0.73–1.02 with 500–970 distinct; the collapsed `β=1.5, t=2` red-sector run measures 0.272 with 50. |
| short budget | per-set ceiling | 10 / 14 / 20 for the 14-, 20- and 79-column sets: 20 mutations is most of a 14-column construct and a quarter of the window |

Among the admissible runs the six are spread across the grid — at most two per
set, differing in `t` within a set — so the figures compare settings rather than
showing six versions of one.

## The figures

| script | what | ladder analogue |
|---|---|---|
| `s01_gate.py` | the drift test per observable per run; acceptance vs the stuck floor | `h01_saturation`, `si06_equilibration` |
| `s02_means.py` | every quantity against `t`, one line per (set, β, λ) cell, with interquartile bands — the marginal effect of each knob | `fig03_cooling` |
| `s03_corner.py` | the selected runs as a **5×5 corner plot**, points larger than the ladder's | `fig02_corner_grafting`, `fig06_corner_selected` |
| `s04_table.py` | the selected runs side by side, best value in each column bold | `fig05_table` |
| `s05_ladder.py` | the **`t` ladder**: one knob moved, five ensembles drawn as the clouds they are, plus the numbers as a table | `fig03_cooling`, but along `t` |
| `s06_ladder_corner.py` | the same five rungs as a **5×5 corner**, on `d`, `E_DCA`, CDE, `Q`, `P(S)` — every axis defined for the wild type too | `fig02_corner_grafting` |
| `s07_sectors.py` | **red-sector enrichment** along each ladder, plus the cumulative curve against the sector's own ranking | `fig08_sector_enrichment` |
| `s08_indels.py` | **indel composition and gap abundance**: how much of the budget is not a substitution, and whether the gap pattern moves | `si03_gaps` |

`s05` and `s06` take the set as an argument (`python analysis/s05_ladder.py
redsector`) and `LADDER_BETA` / `LADDER_LAM` as environment overrides, so the
same figure is made for each set and each `β` without a second copy of the
script. Outputs are suffixed by set and `β`.

### The fifth axis, and why it is here

`fig06` **drops** the natural-neighbour distance when comparing across site sets,
because `add_natural_neighbour.py` scores it over each block's own allowed
columns, and "% of 14 columns" and "% of 79 columns" are not the same quantity
(CLAMP_CONTROL caveat 3).

That caveat is about the *scoring*, not the observable, and it has a fix: all
three sets here are subsets of the grafting window, so scoring every selected
run over that **one common 79-column window** makes the axis mean the same thing
for all of them. `s03_prepare.sh` does that, with the window hard-coded and a
comment saying not to localise it. `s03_corner.py` falls back to a 4×4 if the
column is missing rather than drawing an incomparable axis.

## Running it

```bash
bash t_steering/run_steering.sh              # 48 runs, then rescore, then check
cd t_steering && ../.venv/bin/python analysis/s01_gate.py   # THE GATE
../.venv/bin/python analysis/s02_means.py
bash ../t_steering/s03_prepare.sh            # adds the common-window 5th axis
../.venv/bin/python analysis/s03_corner.py
../.venv/bin/python analysis/s04_table.py
```

Stages can be run singly: `bash t_steering/run_steering.sh coupling20`.
`DEVICE`, `NCHAINS`, `NSTEPS`, `BETAS`, `LAMBDAS` and `T_VALUES` are environment
overrides. Every figure writes a PNG and a vector PDF to `figures/` and its
numbers to `results/`.

**Pin the machine awake, or an overnight run gets ~5 % of the night.** This is
not a small effect and it looks exactly like a hang:

```bash
caffeinate -s -i bash t_steering/run_steering.sh      # or, for a running job:
caffeinate -s -i -w $(pgrep -f run_steering.sh) &
```

Measured on the first pass of this grid. The driver was launched at 23:02 and
macOS entered `'Idle Sleep'` at 23:14:50, twelve minutes later. From then until
morning it cycled `DarkWake -> Maintenance Sleep` roughly every 16 minutes —
**46 sleep events** in the window — granting the sampler about 45–60 s of
compute per cycle. A 60,000-step run needs ~100 s, so it spanned two cycles:

| machine state | measured rate |
|---|---|
| awake | **1.7 min / run** |
| cycling through idle sleep | **~33 min / run** |

37 of 48 runs had completed by 10:52 the next morning instead of all 48 by
00:30. Nothing had crashed and no run was corrupted — each one simply advanced
only during the wake windows. Check with
`pmset -g log | grep -E "Entering Sleep|DarkWake"` before blaming the sampler.

**One run at a time on this machine.** Two MPS processes sampling at once thrash:
with the `hamming-restraint` bracket running alongside, a 60,000-step run went
from 1.5 min to ~35 min, and both jobs were slowed. Suspend one (`kill -STOP`,
which loses nothing and resumes exactly) rather than racing them.

## Everything here is in silico

Sequence arrays, CSVs of observables, and figures. The three sets are mutation
*column* sets on an MSA, the "1992 construct" is the published Hedstrom
comparison read off a paper, and no construct is synthesised, expressed or
assayed anywhere in this subproject.

---

## Results

All **48 / 48** runs pass the gate on `E_DCA`, `Q`, `hamming_to_wt` and `P_S`.
**29 / 48** are admissible on all four requirements.

### What each knob does (`s02_means`)

Marginal effect on the mean, stepping one knob across its range with the other
two held (48 runs, serine-rescored at `beta = 1`):

| knob | `⟨d⟩` | `Q` | `P(S)` | LOR | distinct | div/`⟨d⟩` | acc/site |
|---|---|---|---|---|---|---|---|
| `t`: 0 → 2 | **−3.22** | −1.54 | −0.014 | −0.59 | **−230** | −0.16 | −33.2 |
| `β`: 1 → 1.5 | −1.57 | **−4.30** | −0.019 | −0.71 | −224 | −0.25 | −37.1 |
| `λ`: 100 → 300 | +1.43 | **+5.18** | **+0.048** | **+1.82** | +65 | +0.05 | +1.4 |

The division of labour holds, and the restraint's price has moved:

* **`t` buys budget, and pays in library.** It is the only knob that moves `⟨d⟩`
  as its main effect (−3.2 against `β`'s −1.6), and it barely touches `Q`
  (−1.5, a third of `β`'s −4.3) or `P(S)` (−0.014). But it costs **230 of 1000
  distinct designs** across its range. On the 79-column block of
  `hamming-restraint` the restraint was nearly free; on these small sets it is
  not, and that is the honest headline here — with 16 or 20 mutable columns
  there are simply fewer distinct ways to spend a smaller budget.
* **`λ` is still the specificity knob** (`+0.048` in `P(S)`, `+1.82` in LOR,
  both an order of magnitude above the others) and the only one that *raises*
  `⟨d⟩`.
* **`β` is the quality knob here** (`−4.3` in `Q`, the largest), bought with
  acceptance — `−37` accepted moves per site.

### The six selected designs (`s03_corner`, `s04_table`)

| set | `β` | `λ` | `t` | `⟨d⟩` | sd | `Q` | `P(S)` | LOR | distinct | div/`⟨d⟩` | acc/site |
|---|---|---|---|---|---|---|---|---|---|---|---|
| coupling20 | 1 | 100 | 0.5 | 5.92 | 1.16 | −97.2 | 0.907 | 4.79 | 767 | 0.79 | 111 |
| coupling20 | 1 | 100 | 1.0 | 5.30 | 1.07 | −97.5 | 0.904 | 4.69 | 631 | 0.74 | 91 |
| redsector | 1 | 100 | 0.5 | **4.37** | 0.93 | −94.3 | 0.911 | **5.94** | 313 | 0.65 | 65 |
| redsector | 1 | 100 | 1.0 | **3.99** | 0.85 | −95.3 | 0.904 | 5.54 | 239 | 0.59 | 51 |
| segment | 1.5 | 100 | 0 | 12.30 | 1.79 | **−102.8** | **0.912** | 5.47 | **996** | 0.76 | 11.7 |
| segment | 1.5 | 100 | 0.5 | 10.82 | 1.60 | **−102.8** | 0.908 | 5.41 | 992 | 0.73 | 8.9 |

No row wins every column, which is the point of showing the table rather than a
ranking. The trade is legible:

* **`redsector` is the cheapest switch.** `P(S) = 0.91` at `⟨d⟩ = 4.0` — four
  mutations — and the **highest log-odds of the six** at the shortest budget.
  It pays in library (239–313 distinct) and in `Q` (`−94`, the worst of the six,
  though still 0.4 natural sd from the wild type). For an experimentalist with a
  four-mutation budget this is the row to read.
* **`segment` at `β = 1.5` is the quality option.** `Q = −102.8`, within
  **0.09 natural sd** of the wild type, and a near-complete library (996/1000
  distinct) — but at three times the budget, and with acceptance at 8.9–11.7
  moves per site, below the stuck floor (see the caveat below).
* **`coupling20` sits between the two** on every axis, which is what a set of 20
  columns chosen by coupling to the critic site ought to do.

**The novelty guard.** Every selected design sits **25–37 %** of the common
window away from the nearest natural S-carrying sequence, so none of them is
retrieval of a natural chymotrypsin.

### The `t` ladder (`s05_ladder`)

`segment`, `β = 1`, `λ = 300`, five rungs, everything else identical — same
model, same start, same 79 columns, same 60,000 steps:

| `t` | `⟨d⟩` | sd(`d`) | `E_DCA` | CDE | `Q` | `P(S)` | LOR | distinct | div/`⟨d⟩` | acc/site |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 19.17 | 2.67 | 131.7 | 0.528 | −91.6 | 0.971 | 8.31 | 1000 | 1.04 | 35.2 |
| 0.5 | 15.94 | 2.34 | 123.0 | 0.510 | −92.7 | 0.967 | 7.89 | 1000 | 1.02 | 26.7 |
| 1 | 13.54 | 2.11 | 117.7 | 0.499 | −93.1 | 0.964 | 7.61 | 1000 | 1.02 | 20.6 |
| 2 | 10.07 | 1.71 | 111.3 | 0.486 | −94.3 | 0.956 | 7.34 | 996 | 0.95 | 14.1 |
| 4 | 5.94 | 1.17 | 107.4 | 0.477 | −94.3 | 0.941 | 6.84 | 640 | 0.78 | 7.0 |

**This is the clearest statement in the subproject that `t` does its own job.**
`⟨d⟩` runs 19.2 → 5.9, a factor of 3.2, monotonically, while `P(S)` slips only
0.971 → 0.941 and `Q` *improves* slightly (−91.6 → −94.3). The library is intact
for the first four rungs (1000, 1000, 1000, 996 distinct of 1000) and only
starts to cost at `t = 4` (640).

**The energy is why `Q` must not be read alone.** Down the ladder the energy
improves by **−24.34** while `Q` moves only **−2.78**, because

```
ΔQ = ΔE − μ·ΔCDE = −24.34 − (−21.56) = −2.78        (μ = 422.6, ΔCDE = −0.0510)
```

**89 % of the energy gain is cancelled by the loss of context-dependent
entropy.** Reading `Q` alone gives "the restraint barely affects quality", which
is true of `Q` and hides that both of its components move by an order of
magnitude more and nearly cancel. Panels (b) and (f) of `s05_ladder` are that
decomposition, and `E_DCA` now appears in every table and in `s02_means` for all
48 runs for the same reason. The marginal over the whole factorial shows the
same pattern: `t: 0 → 2` moves `E_DCA` by −6.75 and `Q` by only −1.54.

`s06_ladder_corner` shows the same five rungs pairwise, and the
`E_DCA`–CDE panel is where the cancellation stops being two numbers and becomes
a correlation: the five clouds lie along one line, so the restraint is not
trading energy against entropy, it is moving both down together along a single
direction that `Q` is nearly blind to. Its axes are `d`, `E_DCA`, CDE, `Q` and
`P(S)` rather than the ladder figure's set, because all five are defined for the
wild type — the reference appears in all twenty panels instead of fifteen — and
because the ladder's block was never scored for the natural-neighbour column
(`s03_prepare.sh` only runs on blocks holding a selected run, and the selected
`segment` runs are at `β = 1.5`).

Two further details worth noticing:

* **`sd(d)` falls 2.67 → 1.17 down the ladder.** That is
  `Var(d) = −∂⟨d⟩/∂t` made visible: a stronger conjugate field necessarily
  narrows the distribution it acts on. The figure shows the identity rather than
  asserting it.
* **On 79 columns the restraint is nearly free**, unlike the 16- and 20-column
  sets where it costs 230 distinct designs across its range. A bigger state
  space has more distinct ways to spend a smaller budget.

**Why there is no negative rung.** A `t = −0.5` rung was sampled first and
**dropped by the gate**: `t < 0` pays chains to leave the wild type, and at
60,000 steps it missed on `E_DCA` and `hamming_to_wt` (drift/spread 0.065 and
0.068 against the 0.05 rule, the indel part worst at 0.118) with acceptance at
47.7 per site — still moving, not frozen. Extending it to 240,000 steps would
have fixed it but would have made one rung cost 4× the others, so `t = 4` was
used instead and every rung costs the same. The dropped run is under
`results/_superseded/`.

A guard came out of this: `data.load_all(grid_only=True)` restricts loading to
the `t` values the factorial declares. Without it the ladder's extra rung would
have become a 49th cell of a 48-cell design, and every marginal in `s02` would
silently have been a mean over an unbalanced grid. `s05` is the only caller that
passes `grid_only=False`.

### A tighter budget is spent on the red sector (`s07_sectors`)

Enrichment is `(share of mutations in red) / (share of MUTABLE columns in red)`.
The denominator is the number of **mutable** columns, never the number of
columns — otherwise a site restriction manufactures its own enrichment
(CLAUDE.md). At `β = 1`, `λ = 300`:

| set | red / mutable | enrichment at `t=0` | at `t=4` |
|---|---|---|---|
| `segment` | 16 / 79 (20.3 %) | 1.44 | **3.28** |
| `coupling20` | 7 / 20 (35.0 %) | 1.45 | **2.29** |
| `redsector` | 16 / 16 (100 %) | 1.00 | 1.00 — **degenerate** |

**The restraint does not just buy fewer mutations, it makes the remaining ones
land where they matter.** On `segment`, `t: 0 → 4` takes the red-sector share of
mutations from 29.1 % to **66.4 %** against a 20.3 % baseline: two-thirds of a
scarce budget is spent on the 20 % of columns the sector decomposition marks as
specificity-determining. The bias alone already concentrates somewhat (1.44 at
`t = 0`); `t` amplifies that by a further 2.3×.

For the red-sector set the enrichment is identically 1 **by construction** —
every column it can touch is red — so that line is drawn dotted and labelled
degenerate rather than presented as a flat result.

Panel (b) goes inside the sector, using its own 1-to-16 importance ranking
(`red_rank` in `data/sectors_iter_aln_v2.csv`; not model-derived, so it carries
between models). Every curve bows above the diagonal, and the curves bow
*further* as `t` rises — within the red sector too, a tighter budget follows the
ranking.

### The budget is almost all substitutions, and the gaps barely move (`s08_indels`)

| set | indel share of `d_H`, `t=0 → 4` | gaps per design (wild type: 42) |
|---|---|---|
| `segment` | 5.1 % → **1.9 %** | 41.72 → 42.10 |
| `coupling20` | 5.9 % → 7.5 % | 42.42 → 42.26 |
| `redsector` | 5.6 % → 5.8 % | 42.34 → 42.24 |

**Indels are 2–7 % of the budget and the gap pattern is essentially preserved** —
every design carries the wild type's 42 gaps to within half a gap on average.
A commissioned "ten mutations" on these sets really is ten substitutions.

This also settles how much the single-`t` conflation cost. Every run here was
sampled with one `t` (`t_sub` and `t_indel` both falling back to it), which
prices an indel at the substitution price. Since indels are a few per cent of
`d_H`, that conflation was numerically almost free on these sets — the split is
theoretically necessary and was worth implementing, but it does not change these
numbers. One nuance it does expose: on `segment` the indel share *falls* with
`t` (5.1 % → 1.9 %), so a single `t` suppresses indels faster than
substitutions. With `t_indel = 0` that gap pattern would have been left free.

### What cooling buys, and what it costs (`β = 3` ladders)

The same five-rung ladder was sampled at `β = 3` for all three sets, everything
else identical. `β = 3` is **deliberately inside the frozen regime** — the
48-run factorial stops at 1.5 for the reason below — and it was run to measure
the trade rather than assume it. On `segment` at `t = 0`:

| | `β = 1` | `β = 3` | |
|---|---|---|---|
| `Q` | −91.6 | **−104.4** | +12.9 **bought** (wild type: −105.4) |
| `E_DCA` | 131.7 | **97.1** | +34.7 bought |
| `⟨d⟩` | 19.17 | 11.54 | a shorter budget, thrown in |
| `P(S)` | 0.971 | 0.92 | paid |
| distinct designs | 1000 | 594 | paid |
| divergence / `⟨d⟩` | 1.04 | 0.43 | **halved** |
| accepted / site | 35.2 | **5.1** | **7× collapse** |

So cooling to `β = 3` buys **almost exactly the wild type's own quality** — `Q`
within 1.0 of it, against 13.8 at `β = 1`. That is the real attraction and it
should be said plainly.

**What it costs is the restraint's control.** The span of `⟨d⟩` the `t` ladder
can reach collapses:

| set | `t`-span at `β = 1` | at `β = 3` |
|---|---|---|
| `segment` | 19.2 → 5.9  (**3.2×**) | 11.5 → 6.9  (1.7×) |
| `coupling20` | 8.0 → 4.5  (1.8×) | 5.2 → 4.2  (1.3×) |
| `redsector` | 6.0 → 4.2  (1.4×) | 4.0 → 3.3  (1.2×) |

A frozen chain cannot travel to the equilibrium the restraint is asking for, so
`t` loses most of its leverage. On `segment` the direction even reverses: `Q`
gets *worse* down the `β = 3` ladder (−104.4 → −102.1) where it improved at
`β = 1`.

**And the gate cannot see any of this.** All **15 / 15** `β = 3` runs pass the
drift test on `E_DCA`, `Q`, `hamming_to_wt` and `P_S` — and all **15 / 15** sit
below the 20-accepted-moves-per-site floor, at 2.8–5.1 per site on `segment`.
That is the "flat because frozen, not because settled" failure in its pure form:
zero motion is zero drift.

The decisive evidence is an identity rather than a threshold. `coupling20` at
`β = 3` has **non-monotone `⟨d⟩`**: 5.20, 5.04, **5.16**, 4.77, 4.17 across
`t = 0, 0.5, 1, 2, 4`. Since `−∂⟨d⟩/∂t = Var(d) ≥ 0` holds for *any* equilibrium
distribution of this family, a rise is not a measurement — it is proof that the
rung is not an equilibrium sample. The drift test passed it anyway.

**Read `β = 3` as an upper bound on quality, not as a design protocol.** It
reaches wild-type `Q` and it does so by refusing moves; the ensembles are
bounds, not converged values, and the restraint barely works inside them.

### Two caveats that travel with these numbers

**No design reaches wild-type `Q`.** All 48 are worse than the wild type, by
`+2.5` to `+19.1`. Read against the family rather than against the wild type
alone that is not alarming — the natural alignment's own `sd(Q)` is 28.0, so the
whole grid lies within **0.09–0.68 natural standard deviations** of the wild
type — but "`Q` similar to the wild type" had to be defined as a tolerance band
on that scale, not as `Q ≤ Q_WT`, which admitted nothing. The wild type is an
unusually good sequence under this model and `β ≤ 1.5` does not reach it.

**17 of 48 runs sit below 20 accepted moves per mutable site**, including both
selected `segment` rows (8.9 and 11.7). They are not collapsed — 992–996 of 1000
designs are distinct — and `β = 1.5` is the rung the ladder itself treats as the
warm competitor, where its own runs measure 10.1–11.3 accepted/site. But a drift
test cannot tell a settled chain from a frozen one, so the figure reports the
number beside every verdict rather than burying it, and `β ≥ 2` is excluded from
the grid for exactly this reason.
