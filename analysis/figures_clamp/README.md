# Clamp-and-relax

A separate pipeline from `analysis/figures_ladder/`, answering a different
question with no bias term at all.

```
          analysis/figures_ladder/    this folder
steering  H = βE − λ log P(S|·)       H = βE,  σ_i* ≡ S  (frozen)
λ         a ladder, 0 … 1000          0, always
P(S)      the objective               a diagnostic
results   results/ladder              results/clamp
```

## Why it exists

The project's central number is `P(S | σ\i*)`, and the project's central worry is
that it can be high for the wrong reason. Two things sharpened that worry.

**The log-odds pilot.** `log P(S|·)/P(D|·)` is a better *classifier* than
`log P(S|·)` — it separates natural trypsins from chymotrypsins at AUC 0.983
against 0.965 for a no-epistasis profile. So it looked like a better steering
function too. It is not. Steering on it for 40 000 steps at λ = 3 gave

```
          P(S)    log-odds   residue at i*
λ = 3     0.007   +24.3      G = 0.992,  S = 0.007
λ = 10    0.046   +32.1      G = 0.954,  S = 0.046
λ = 30    0.566   +38.6      S = 0.566,  G = 0.433
```

The objective only asks that S beat D. The cheapest way to do that is to build a
pocket where D is impossible, and what fits that pocket is **glycine**. Compare
the same model steered on `log P(S)`: λ = 1000 reaches P(S) = 0.989 with 20
substitutions and E = 145, where the log-odds run needed 51 substitutions and
E = 301 to reach P(S) = 0.57. The log-odds steering was dropped.

`log P(S|·)` survived that test for a structural reason worth stating: it is
*already* a log-odds, against all 20 alternatives at once. Its denominator
contains every way of cheating. The single-competitor version's does not.

**What that implies.** If a gameable objective produces a glycine pocket, the
question for the objective we kept is what IT is quietly producing. The answer
cannot come from another objective. It has to come from a protocol with nothing
to optimise.

## The protocol

Put a serine at the critic column, freeze it, and sample the rest from the plain
model. Zero bias parameters. `P(S | σ\i*)` does not depend on the residue at i*,
so it stays perfectly well defined — and since nothing is pushing on it, what it
reads is the model's own opinion of how well the surrounding context supports
the serine that was imposed on it.

Four conditions, and the two controls are not optional:

| condition | what it is | columns free | beta |
|---|---|---|---|
| `clamp_graft` | S imposed | 165-243 | 1, 1.5, 2, 3, 5 |
| `clamp_free` | S imposed | all 265 | 1, 2, 5 |
| `relax_graft` | control, wild-type D kept | 165-243 | 1, 2, 5 |
| `relax_free` | control, wild-type D kept | all 265 | 1, 2, 5 |

Cold sampling moves every observable on its own, so each clamped condition is
read only against the control sampled the same way without the serine.
`relax_free` was added after `clamp_free` produced a number that looked like a
result and turned out to be drift.

## What came out

**C1 — the context does not reorganise to support the serine.** At beta >= 1.5
the clamped runs read 0.001-0.004 for P(S), the same as their controls. Only at
beta = 1 is there anything (0.040 against 0.001). Cooling pulls the context back
to the trypsin consensus with the serine still fixed in place. So the lambda
bias is doing work that energy minimisation does not do on its own — which is
the thing a reader is entitled to suspect, and now does not have to.

**C2 — the family gap is untouched, and the large number belonged to the control.**

```
                           beta=5   gap closed
clamp, grafting window                 4.7%
no clamp (control)                     4.1%
clamp, all sites free                 23.3%
no clamp, all sites free              24.6%   <- the control closes more
```

Relaxing the whole chain closes about a fifth of the 82-residue gap whether or
not a serine is in it. That is the model drifting toward its own global
consensus. The serine contributes nothing, at any beta, in any window.

**C3 — the pocket does respond, at one position, and it was a real prediction.**
MSA column 241 (3TGI residue 226) is read by nothing in this project: no site
set, no bias, no selection rule. The two natural families differ there sharply,
and the clamped runs move toward the chymotrypsin profile while their controls
stay put.

```
                             G      S      A      T
natural trypsins           0.91   0.03   0.04    -
natural chymotrypsins      0.10   0.28   0.22   0.21
clamp, grafting window     0.67   0.11   0.08    -
no clamp (control)         0.96   0.01   0.01    -
clamp, all sites free      0.50   0.30   0.09    -
no clamp, all sites free   0.95   0.03   0.02    -
```

The free clamp puts S at 0.30 where natural chymotrypsins sit at 0.28, against
0.03 in its own control. Column 229 (residue 216) moves the same way but weakly:
0.89-0.91 against 0.98 in the controls, 0.68 in the chymotrypsins. Column 177
(residue 172) does not move at all.

So the model has learned the 189-226 coupling and uses it with nothing asking it
to. It has not learned how to cross the distance between the two families, and
it will not rebuild the context around a serine on its own.

## Running it

From the repository root:

```bash
bash analysis/run_clamp.sh                  # STAGE 1, all four conditions (~2 h, GPU)
bash analysis/run_clamp.sh clamp_graft      # one condition
.venv/bin/python analysis/figures_clamp/make_all.py   # STAGE 2, the figures (seconds)
CAVEATS=1 .venv/bin/python analysis/figures_clamp/make_all.py   # with the text under each figure
```

The figures are written to `figures/clamp/`.

The figures carry a title and nothing else. The caveats are still in the scripts
and are printed to stdout by `make_all.py`, so what has to be said out loud next
to each figure is recorded rather than deleted.

`config.py` keeps the fingerprint guard of the lambda pipeline: an ensemble
sampled from a different parameter file is a fatal error, not a warning. `mu` and
the natural-sequence observables are read from `results/ladder/calibration/`
rather than recalibrated: they are properties of the model, not of a protocol,
and recomputing them would make the two pipelines needlessly incomparable.
