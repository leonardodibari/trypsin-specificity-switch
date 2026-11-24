# The Hamming restraint

**Making the locality of the designs an explicit, tunable, equilibrium-respecting
term instead of an accident of the sampler.**

---

## Why this subproject exists

The main study's scope is already "chymotrypsins close to wild-type trypsin", but
**by accident rather than by design**. Three things enforced it implicitly, and
none of them is a term in the Hamiltonian:

1. **the starting point** — chains start at the wild type, which sits ~143 below
   the family mean in energy, and `docs/RESULTS.md` §2 records that a WT start
   does not equilibrate in 4,000 sweeps;
2. **the allowed-column set** — the 79-column grafting window forbids most of the
   moves the switch needs. `docs/CLAMP_CONTROL.md` §2.2: 50 of the 71 columns
   that move in `clamp_free` lie *outside* the graft segment;
3. **cooling** — at β ≥ 2 acceptance collapses to 0.008 and the chains are
   "stuck near the wild type, not converged".

Net: designs sit at Hamming 6–28 from WT where natural chymotrypsins are 81–196,
and close ≤5 % of the gap. **The locality is produced by a frozen state space and
a frozen sampler.** That is the argument for this change: make the locality an
explicit term, and let the state space go free.

## The term

The sampled Hamiltonian gains one piece:

```
H(σ) = β·E_DCA(σ) − λ·log P(S | σ_\i*) + t·d_H(σ, WT)
```

with the critic conditional always at `CRITIC_BETA = 1`, so cooling the energy
does not also sharpen the bias.

Two properties make this cheap and exact, and the whole subproject rests on them:

* **It is exactly a field.** `d_H(σ,WT) = Σ_i 1[σ_i ≠ wt_i]` is a sum of
  single-site indicators, so `+ t·d_H` is *identical* to shifting the field by
  `+t` on every non-wild-type state. The sampled distribution is still exactly a
  Potts model, `ΔH` is O(1) per move, equilibrium is well defined, and every
  existing Q/CDE machinery stays valid. **There is no approximation anywhere.**
* **t is the chemical potential conjugate to d.** Because the term is linear in
  `d`, two identities follow and are used as measurements, not assumptions:

  ```
  −∂⟨d⟩/∂t = Var(d)                         (susceptibility)
  P_t(d) ∝ exp(−F(d)) · exp(−t·d)           (reweighting → F(d) by WHAM)
  ```

**Sign convention.** `t > 0` restrains toward WT (minimal-mutation designs, the
experimentally buildable regime); `t < 0` pays the chains to leave. t is scanned
through zero, so the sign is a scan parameter rather than a choice.

**Two gotchas, both handled — the first now in the Hamiltonian, not just in the
reporting.** With `--allow-gap-moves` indels count in `d_H`, so a single `t`
prices a gained or lost residue at the substitution price without saying so. The
restraint is therefore split:

```
H(σ) = β·E_DCA(σ) − λ·log P(S | σ_\i*) + t_sub·d_sub(σ,WT) + t_indel·d_indel(σ,WT)
```

`d_sub` counts the positions where both residues are present and differ,
`d_indel` the positions where exactly one of the two is a gap; they are disjoint
and sum to `d_H`. Each is still a single-site indicator, so the sum is still
exactly a field — at a site whose reference residue is `r` the shift is 0 on `r`,
`t_indel` on the gap-vs-residue states and `t_sub` on the rest — and every
property above survives unchanged. `t_indel` unset falls back to `t`, which is
what `t` alone has always meant, so **no run sampled before the split is
reinterpreted and none has to be resampled**. Both contributions are traced
separately (`sub_to_wt`, `indel_to_wt`) rather than one being inferred from the
other.

Second: `d_H` is defined over the **allowed columns**, not over all L. The two
definitions differ only where the reference differs from the start *outside* the
allowed set — frozen columns — so the difference is a constant in `H`, cancels in
every `ΔH`, and changes no sampled distribution; it changes only whether a
reported `d` carries a frozen offset. In the runs as scripted the two coincide
outright: the graft runs (`--allowed-cols 165-243`) start at the wild type, which
*is* the reference, so the outside contributes nothing, and the bracket runs,
the only ones whose start is not the reference (`--start-fasta`, a natural
chymotrypsin), are run over `--allowed-cols 0-264` = all 265 columns.

### Correctness

`t = 0` reproduces the pre-restraint results bit-for-bit under the same seed, and
the restrained sampler is validated against an **exactly enumerated** small
model, not only against itself. `tests/test_ensemble.py`, `tests/test_mcmc.py`; 64 tests pass. The split is
checked three ways: against an exact enumeration of the reachable space with the
gap included and the two prices given opposite signs (which a single summed
counter cannot reproduce at all), against a closed form on a zero-coupling model
where the sites factorise and `<d_sub>`, `<d_indel>` are sums of single-site
softmaxes, and by fault injection — dropping one incremental indel update makes
the end-of-run guard fire, as it must, since the two counters are compared
separately and a leak between them would otherwise keep their sum right.

---

## The discipline: nothing is read off a run that has not saturated

This is the rule the subproject is built around, and it is enforced in code
rather than by memory.

`analysis/saturation.py` checks **every** traced observable separately —
`E_DCA`, `Q`, `hamming_to_wt`, `indel_to_wt`, `P_S`, `LOR`, `CDE` — and a run
passes only if *all* of them pass. `analysis/data.py:designs()` then silently
drops the runs that failed, so a figure cannot come to rest on a chain that was
still moving.

The criterion, per observable, over the second half of the run:

```
drift  = |mean over last quarter − mean over third quarter|
spread = the ensemble's own IQR, averaged over the second half
range  = the full excursion of the mean over the whole run

saturated  ⟺  (drift < NEGLIGIBLE[obs]  OR  drift < 0.05·spread)
              AND drift < 0.10·range
```

**Stuck is not settled.** A chain with acceptance 0.0015 produces a flat trace
because it is frozen, not because it has equilibrated. A run with fewer than 5
accepted moves per mutable site is flagged *stuck* however flat its trace is, and
a stuck run is reported as NOT saturated.

### What that discipline caught

**The all-columns-free runs needed 3.3× more steps than they were given.** The
first pass sampled them at 120,000 steps, the figure inherited from the 79-column
window. Fitting `a + b·exp(−n/τ)` to the traces puts their relaxation at
**τ ≈ 19,000–31,000 steps**, so 120k was only 4–6 τ, and **7 of the 9 rungs failed
the drift test**. They were *not* stuck — 12 to 75 accepted moves per site, far
above the floor — they were simply still moving.

| block | columns | steps | verdict |
|---|---|---|---|
| `a1_graft` | 79 | 60,000 | **20 / 20 saturated** |
| `a2_free_lam100` (first pass) | 265 | 120,000 | 2 / 9 saturated — **discarded** |
| free runs (re-sampled) | 265 | 400,000 ≈ 16 τ | see `results/saturation.csv` |

The 265-column runs are therefore sampled at **400,000 steps**, a number that was
*measured* rather than chosen. A 79-column window needs 60,000; the same figure
applied to a 3.4× larger state space is not conservative, it is wrong.

**The gate was being applied to one side of the comparison only.** `data.py`
asserted `saturated = True` for every run of the ladder the restraint is measured
against, with a comment deferring to the ladder's own SI. The ladder's traces are
in `results/ladder/*/ensembles/lam*/trace.csv` and had never been put through
this subproject's drift test — and two of the 21 fail it: `β=1 λ=0`, whose `P(S)`
is flat at 0.0014, and `β=1 λ=1000`, still drifting in `E_DCA` and
`hamming_to_wt`. `load_ladder` now **measures** it, reconstructing
`Q = E_DCA − μ·CDE` from the traced `E_DCA` and `CDE` so the quality the designs
are ranked by is gated too, and `cooling_table` drops what fails. Both failures
were levers on the `λ` slope of test 2, and removing them *strengthened* it
(`dP_S/dd` +0.032 → +0.057). `h02`, `h04` and `h05` were regenerated; `h04`'s
verdict is unchanged, because neither dropped run appears in a matched pair.

A caveat that travels with this: the drift test cannot see a run that is flat
because it is **frozen**, and at `β ≥ 2` every cooled rung has drift exactly 0 on
every observable. Those runs *pass* the gate and are still not equilibrium
samples — which is why `stuck` exists beside `saturated`, and why every slope
quoted for `β` in test 2 is taken from the warm rungs (`β = 1, 1.5`) alone.

---

## The tests, and which figure answers each

| # | question | figure |
|---|---|---|
| 1 | **Is the trypsin island a basin or a ramp?** Scan `t`, watch `Var(d) = −∂⟨d⟩/∂t`. A peak with a bimodal `P(d)` is a genuine barrier and locates it; a smooth ramp means the §6 caveat is a cost gradient, not a basin. Plus `F(d)` by WHAM, with its own `dF/dd = −t` consistency check. **Gated by `h07`: 1 of 16 rungs closes, so this is currently unanswered — see Results.** | `h03_susceptibility`, gated by `h07_bracket` |
| 2 | **Are the three knobs independent?** Each knob is measured by the path it drags the ensemble along, not the point it lands on: the slope of every observable against `⟨d⟩` along that knob's own axis, the other two held. `λ` should trade `P(S)` against `Q`, `β` should trade `Q` against diversity, `t` should set `⟨d⟩` and touch the other two only through it. | `h08_knobs`, with the per-column spectra in `h04_degeneracy` |
| 3 | **Does `t` beat `β` on the Pareto front?** The test that justifies the term. At matched `P(S)` and matched `⟨d⟩`, which protocol gives better `Q` and a more diverse library? Prediction stated in advance: β restrains by *freezing* (non-equilibrium), `t` by *tilting the equilibrium*, so at matched Hamming `t` should give the same Q with far more unique designs. | `h02_pareto`, `h04_degeneracy` |
| 4 | **The mutation-budget curve, benchmarked against Hedstrom.** Best achievable **log-odds** (never `P(S)`, which saturates at 0.99) against budget, with the 1992 constructs overlaid at their known `d`. | `h05_budget` |
| 5 | **Can you leave the island?** Negative `t`, all columns free. Does the ensemble move along the trypsin→chymotrypsin axis or orthogonally into junk; do the four refusals of `CLAMP_CONTROL` §2.3 (226, 216, 172, loop 185–188) finally happen; does the nearest natural neighbour switch family. | `h06_escape` |
| — | **saturation of every observable**, for every run in the study | `h01_saturation` |
| — | **the two-start bracket**: whether an all-columns-free rung has an equilibrium at all. The gate for every free-column number, and strictly stronger than the per-run drift test — it catches rungs that pass the drift test on `Q` while the two arms disagree on `Q` by 78 sem. | `h07_bracket` |

**The degeneracy check (part of 3).** Match `β_eff` to each `t` by `⟨d⟩` and
compare the per-column mutation spectra. If they are identical, `t` is a
reparametrised temperature and the project says so plainly.

**The matched control (part of 5), non-negotiable.** `t < 0` alone drags chains
toward the family average, and the family carries S at column 201 in 10.4 % of
its members — so negative `t` raises `P(S)` with no bias at all. Every t<0 run is
paired with the identical run at `λ = 0`. Same design as clamp-vs-relax. **If
both arms land in the same place, the study has been diluting toward the family
mean, not designing.**

**The novelty guard (part of 5).** At each `t`, the minimum Hamming distance to
the 89,439-sequence alignment. A design at `d = 100` that is a copy of some
natural chymotrypsin is *retrieval, not design* — and this becomes essential the
moment `t < 0` lets chains travel 100+ substitutions.

---

## Layout

```
hamming-restraint/
  run_phase_a.sh      A1 grafting λ×t · A2 free t-scan at λ=100 · A3 the λ=0 control
  run_phase_a2.sh     what phase A did not finish, and the re-sampling at 400k steps
  run_phase_b.sh      the four site sets, under whatever phase A decided about t
  postprocess.sh      rescore every block at β=1 with S stamped at the critic column
  analysis/           config · data · saturation · h0*.py (one module per figure)
  results/            one directory per run; saturation.csv is the gate
  figures/            committed PNG + PDF
```

### The four site sets of phase B

Read from the ladder's own `config.site_sets()` — **never redefined here**, so
there is exactly one definition of each in the repository. `coupling-20` in
particular is model-*derived* and would be wrong if copied from an older model.

| set | columns | what it is |
|---|---|---|
| `grafting` | 79 | the window the main study used, cols 165–243 |
| `all free` | 265 | everything |
| `1992 construct` | 14 | what Hedstrom, Szilagyi & Rutter actually built and assayed |
| `coupling-20` | 20 | the 20 window columns most strongly coupled to the critic site |

Phase B's `t` grid depends on phase A's verdict, and is passed in rather than
edited:

```bash
# t is useful -> scan it in every set
bash hamming-restraint/run_phase_b.sh
# t is not useful -> the four sets as four unrestrained conditions
T_VALUES="0" bash hamming-restraint/run_phase_b.sh
```

## Running it

```bash
bash hamming-restraint/run_phase_a2.sh     # re-sample the free runs at 400k steps
bash hamming-restraint/run_phase_b.sh      # the four site sets
bash hamming-restraint/run_bracket.sh      # the two-start bracket (c1/c2 arms)
cd hamming-restraint/analysis && ../../.venv/bin/python h01_saturation.py   # the gate
../../.venv/bin/python h08_knobs.py        # or any other hNN; each is standalone
```

Each `hNN_*.py` runs on its own and writes its figure to `figures/` and its
numbers to `results/`. There is no `make_all.py` here: the figures do not share
intermediate state, and `h08_knobs` imports `h02_pareto`'s two tables directly
rather than rebuilding them.

Every driver takes stage names, so a single stage can be re-run without redoing
the rest. `DEVICE`, `NCHAINS`, `NSTEPS`, `NSTEPS_FREE`, `LAMBDAS` and `T_VALUES`
are environment overrides. The heavy data (the 24 MB alignment, the 137 MB
parameter file) is not in git — see `docs/DATA.md`.

---

## Results

*Filled in as each block clears the saturation gate. A number does not enter this
section until the run behind it has passed `analysis/saturation.py` on every
observable.*

### Test 1 — not answerable at the current sampling (`h07_bracket`)

**The two-start bracket closes on 1 of 16 rungs.** Every rung of the `λ=100`
scan was sampled twice over all 265 columns, 1000 chains each, annealed in `t`:
the near arm from the wild type (`d=0`, approaching each equilibrium from below)
and the far arm from bovine chymotrypsinogen B (`d=145`, from above). A rung
closes only when the arms agree on `⟨d⟩` within the sampling error, on the
per-column marginals within the finite-sample floor, **and** on `Q`, `E` and
`P(S)` within 2 sem.

| `t` | `⟨d⟩` near | `⟨d⟩` far | gap | `Q` gap | verdict |
|---|---|---|---|---|---|
| −1.00 | 232.6 | 230.4 | 2.2 (7.3 sem) | 1.2 (1.5 sem) | open — marginals at the floor |
| −0.25 | 113.1 | 197.9 | **84.8** | **65.4 (66 sem)** | open |
| 0.00 | 98.4 | 182.3 | **83.9** | **57.2 (77 sem)** | open |
| +0.50 | 52.2 | 120.1 | 67.9 | 23.5 (31 sem) | open |
| +1.00 | 35.8 | 40.1 | 4.3 (6.9 sem) | 0.8 (1.9 sem) | open — `d`, marginals |
| **+2.00** | **19.10** | **19.03** | **0.07 (0.4 sem)** | **0.05 (0.2 sem)** | **closed** |

Across `t ∈ [−0.4, +0.5]` the two arms are **55 to 91 substitutions apart in
`⟨d⟩`**, with per-column total variation of 0.30–0.57 against a floor of
0.04–0.07. The near arm's "equilibrium" at `d ≈ 100` and the far arm's at
`d ≈ 185` differ by nearly a factor of two. **Nothing may be read off those
rungs** — not `⟨d⟩`, not `Var(d)`, not `F(d)`.

**So the basin-vs-ramp question is not answered.** `t = 2` is the only
equilibrium point in the scan, and one point gives no `Var(d)` curve to find a
peak in. Any apparent susceptibility peak in the interior of this scan sits
exactly where the bracket is widest, which is where a peak is least
interpretable: an ensemble still expanding outward from `d = 0` is a mixture of
chains at different stages of that journey, so its spread is inflated by the
travel rather than by a barrier.

**The methodological finding, and it is the sharp one.** The per-run drift test
is **necessary but not sufficient**, and this is where that stops being a
caution and becomes a measurement: there are rungs whose drift test *passes* on
`Q` while the two arms disagree on `Q` by **up to 68 units — 78 sem** — and on
`E_DCA` by up to 177 units (71 sem). Only 3 of 16 rungs agree between arms on
`Q` within 2 sem, 2 of 16 on `E`, 4 of 16 on `P(S)`. A single chain that has
stopped moving has not thereby reached equilibrium; it has stopped where it
happened to be. Two chains started at opposite ends of the order parameter and
stopping in the same place is the test with content, and it is strictly stronger.

One rung is worth a second look: at `t = −1` the arms agree on every energetic
observable (`Q` 1.5 sem, `E` 0.17 sem, `P(S)` 0.97 sem) and still differ on `d`
(7.3 sem) and, marginally, on composition (TV 0.082 against a floor of 0.081).
Two ensembles at `d ≈ 231` that are energetically indistinguishable and
compositionally distinguishable is what a large degenerate manifold looks like,
and it is a question the bracket raises rather than settles.

**What is still running.** The `c2` arms — the same bracket at `λ = 0` — are
sampling. That control is not optional: `t < 0` alone drags chains toward the
family mean, and the family carries S at column 201 in 10.4 % of its members, so
a negative `t` raises `P(S)` with no design pressure whatever. Until the control
lands, the `λ = 100` numbers above cannot be attributed to the bias.

### Test 2 — the knobs are distinguishable, not orthogonal (`h08_knobs`)

79-column grafting block, 60,000 steps on both axes. All 20 runs of the `λ×t`
grid saturate; of the 21 ladder runs, `β=1 λ=0` (P_S flat at 0.0014) and
`β=1 λ=1000` (still drifting in `E_DCA` and `hamming_to_wt`) do not and are
dropped, so the `λ` path is `λ = 10…300`. Cost per unit of mutation budget,
each knob along its own axis:

| knob | `dQ/dd` | `dP_S/dd` | unique designs across its range | accepted/site |
|---|---|---|---|---|
| `t` (restraint) | **+0.33 … +0.40** | +0.0016 … +0.0071 | 100 % → 96.7 % | 34.5 → 9.9 |
| `β` (cooling), warm rungs only | +1.43 … +1.73 | −0.0042 … +0.0021 | 100 % → **7.9 %** | 29.1 → **1.1** |
| `λ` (bias), `λ = 10…300` | +2.23 | **+0.0569** | 100 % → 100 % | 35.2 → 29.1 |

Read as a division of labour, it holds in the direction claimed but **not as
orthogonality**:

* **`λ` is the specificity knob.** `dP_S/dd = +0.057`, **16×** `t`'s `+0.0036`.
  Both runs that the gate removed were levers on this number and both were
  removed before it was quoted: the `λ = 0` control sits at `P(S) = 0.0014`
  against ≥ 0.74 everywhere else, and including it would have reported `+0.094`
  (26×) off a run that does not pass the drift test.
* **`t` is the budget knob, and it is the cheap one.** It buys a substitution of
  budget at `dQ/dd ≈ +0.37` against `≈ +1.57` for warm cooling — **4.2× less
  quality given up per substitution** — and it does it without touching the
  library (96.7–100 % unique at every budget) or the acceptance.
* **`β` restrains by freezing, as predicted.** Across its range the library
  collapses to 7.9 % and acceptance to 1.1 accepted moves per mutable site,
  26× below its own warm start. Both blocks ran 60,000 steps over the same 79
  columns, so those acceptance numbers are directly comparable.

**`t` is not a reparametrised temperature**, and this is the part summary
statistics cannot carry: at matched `⟨d⟩`, `h04` finds **48–53 of 79 columns**
whose per-column spectra differ beyond the resampling null, against **2–6 of 79**
for the `t = 0, β = 1` self-comparison that sets the noise floor. The two
protocols select genuinely different sequences at the same cost.

**What the term does not buy.** At matched budget cooling still reaches the
better `Q` — `ΔQ = −4.8 … −6.5` over `d ≈ 10.4 … 12.0` (`h08_matched_budget.csv`)
— so the restraint is not dominant on quality. It is dominant on everything that
makes that quality readable: every one of those cooled rungs is at `β ≥ 2`, where
CLAMP_CONTROL caveat 1 applies and `Q` is not an equilibrium number, while the
`t` rungs at the same budget are warm and 100 % unique.

The honest summary is that the three knobs are **distinguishable but coupled**:
`t` moves `Q` and `P(S)` too, just along a different and much cheaper path than
`β` or `λ` trace. The byproduct problem is solved — the budget is now set
directly — without the knobs having become independent.
