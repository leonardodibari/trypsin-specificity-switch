# The free-column runs do not equilibrate in `d` from a wild-type start

**Status: blocking finding for every all-265-columns-free result. Recorded
2026-11-23.** It does **not** affect the 79-column grafting block, which is
fully saturated.

## What was measured

A pilot at the susceptibility peak (`λ=100`, `t=−0.25`, all 265 columns free,
gap moves on, 1000 chains), sampled at **400,000 steps** — 3.3× the 120,000 of
the first pass.

Verdict: **still NOT saturated.** `acc/site = 172.9`, so the chains are nowhere
near stuck; they are simply still moving.

| observable | drift | drift/spread | verdict |
|---|---|---|---|
| `E_DCA` | 3.58 | 0.039 | ok |
| `Q` | 0.75 | 0.022 | ok |
| `P_S` | 0.0007 | 0.048 | ok |
| `CDE` | 0.0067 | 0.026 | ok |
| **`hamming_to_wt`** | **3.66** | **0.165** | **FAIL** |
| **`indel_to_wt`** | **1.66** | **0.150** | **FAIL** |
| `LOR` | 0.33 | 0.065 | FAIL (marginal) |

## The trace

```
   step    E_DCA   hamming   indel    P_S     CDE
      1     82.9      0.0     0.0   0.0005   0.445
  40000    298.3    127.9    14.3   0.9788   0.948
  80000    322.8    150.1    20.1   0.9799   1.029
 120000    333.2    159.9    23.8   0.9807   1.066
 160000    338.6    166.8    26.8   0.9811   1.085
 200000    340.9    171.0    28.9   0.9815   1.096
 240000    342.5    174.4    30.5   0.9821   1.102
 280000    339.9    175.7    31.3   0.9820   1.099
 320000    339.2    177.4    32.0   0.9813   1.096
 360000    337.0    178.4    32.3   0.9824   1.092
 400000    335.4    179.0    32.8   0.9832   1.089
```

## What this means

**The energetic observables have settled; the geometric one has not.** `E_DCA`
peaks near 240k and comes back down; `P_S` is flat from 40k; `CDE` is flat from
~280k. But `hamming_to_wt` climbs monotonically for the whole run and is *still
climbing* at 400k.

The decisive detail is the **rate**: the drift fell only from 7.86 (at 120k) to
3.66 (at 400k). A 3.3× longer run bought a 2.1× reduction. Single-exponential
relaxation with the τ ≈ 25,000 fitted to the early trace would have predicted a
reduction of many orders of magnitude. So **`d` is not relaxing exponentially —
it is diffusing**, and the increments decay roughly logarithmically.

Extrapolating that behaviour, reaching `drift/spread < 0.05` by brute force
would take **several million steps per rung**, times ~12 rungs times 2 arms.
That is not the right answer.

### Why `d` is the slow coordinate

The chains start at the wild type, `d = 0`, and the restrained equilibrium at
`t = −0.25` sits somewhere above `d ≈ 180`. Every chain has to *travel* that
distance by single-site moves, and `d` is a slow collective coordinate: the
energy equilibrates locally long before the ensemble has finished expanding
outward. This is the same pathology `docs/RESULTS.md` §2 records for a WT start,
now visible quantitatively because the restraint gives us an order parameter and
a conjugate field to watch it with.

Note the sign: the ensemble is approaching its plateau **from below**, so every
number read off these runs is a *lower bound* on `⟨d⟩`.

## The fix (not more steps)

Three things, in order of importance:

1. **Stop starting every chain at the wild type.** `run_ensemble` already
   accepts one starting sequence per chain (`(n, L)`), and
   `run_generative_check.py` already uses that to start chains spread over the
   family — precisely to tell a model that cannot reproduce its statistics from
   chains that have not mixed. The free runs should start from a spread of
   natural sequences, or by **annealing in `t`**: sample the rungs in sequence
   and start each one from the final chains of its neighbour, so each rung pays
   only the small displacement between adjacent `t` values rather than the whole
   journey from `d = 0`.

2. **Run the two-start bracket, and treat it as the verdict.** Start one set of
   chains at the wild type (`d = 0`) and another at a natural chymotrypsin
   (`d ≈ 150+`), under identical `(β, λ, t)`. The two starts bracket the `d`
   axis from opposite sides, so **agreement between them is a far sharper
   equilibrium test than any single-run drift criterion**, and disagreement
   bounds the residual error honestly. This is the cheapest rigorous answer
   available and it should gate every free-column number.

3. Only then, extend the step count — and size it from the two-start gap, not
   from a fitted τ that the data has already shown to be the wrong model.

## What is and is not affected

| block | columns, steps | status |
|---|---|---|
| `a1_graft` | 79 cols, 60k steps | **20 / 20 saturated — unaffected** |
| `a2_free_lam100` (first pass) | 265 cols, 120k steps | 2 / 9 saturated — discarded |
| free pilot (λ=100, t=−0.25) | 265 cols, 400k steps | still fails on `d` |

**The two tests that justify the term are unaffected.** The Pareto comparison
against cooling (test 3) and the mutation-budget curve against the 1992
constructs (test 4) both live entirely on the 79-column grafting block, which is
fully saturated. Only the two all-free questions — is the island a basin
(test 1), and can the chains leave it (test 5) — are blocked by this.
---

## The bracket has now been run — and it is worse than this note assumed

**Recorded 2026-11-23, after `h07_bracket` on the `λ=100` arms.** The fix this
note prescribed (item 2: "run the two-start bracket, and treat it as the
verdict") has been carried out. Both arms, 16 rungs each, all 265 columns free,
1000 chains per arm, 150,000 steps per rung, annealed in `t`.

**1 of 16 rungs closes** (`t = 2`, `⟨d⟩ = 19.1`, arms agreeing to 0.4 sem on
`d`, 0.2 sem on `Q`, and below the finite-sample floor on composition). Across
`t ∈ [−0.4, +0.5]` the arms sit **55–91 substitutions apart** in `⟨d⟩`.

### What this note got wrong

It said: *"the energetic observables have settled; the geometric one has not."*
That was measured **within one arm**, and it does not survive the bracket.
Between arms the energetics do not agree either:

| quantity | rungs agreeing between arms (2 sem) | worst disagreement |
|---|---|---|
| `Q` | 3 / 16 | 68 units, **78 sem** |
| `E_DCA` | 2 / 16 | 177 units, 71 sem |
| `P(S)` | 4 / 16 | — |

So the picture of "energetics fast, `d` slow" was an artefact of asking only
whether a single chain had stopped moving. A chain that starts at `d = 0` and
one that starts at `d = 145` both stop moving in `E_DCA` long before either has
found the equilibrium, and they stop at energies 177 units apart. **The per-run
drift test is necessary and not sufficient**, and `analysis/saturation.py`
cannot be the last word on a free-column run; `h07` is.

### Consequence for the step count

This note argued for 400,000 steps on the strength of a fitted `τ`. The bracket
makes that number unsupportable as a *sufficiency* claim: the arms are
85 substitutions apart after 150,000 annealed steps per rung, having each paid
only the displacement to their neighbour rather than the full journey, and
annealing was supposed to be the cheaper route. Sizing the run from the
two-start gap — as item 3 said — is now the only defensible procedure, and the
gap says the interior of the `t` scan is far from settled at any step count
these runs have been given.

### What is unaffected

Still only the all-265-free questions. The 79-column grafting block remains
20/20 saturated, and tests 2, 3 and 4 live entirely on it.
