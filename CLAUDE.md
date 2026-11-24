# CLAUDE.md

Orientation for Claude Code working in this repository. The prose documentation is
already extensive — this file says *where to look* and *what not to get wrong*, it
does not restate `README.md`.

## What this is

Biased MCMC on a Potts/bmDCA model of the serine-protease family, sampling sequences
that start from wild-type rat trypsin (P00763) and drift toward chymotrypsin-like S1
specificity. Sampled Hamiltonian:

```
H(σ) = β·E_DCA(σ) − λ·log P(S | i_critic, σ)
       + t_sub·d_sub(σ, WT) + t_indel·d_indel(σ, WT)
```

The critic conditional is always evaluated at `CRITIC_BETA = 1`, so cooling the
energy does not also sharpen the bias. The last two terms are the Hamming
restraint (new; see *Work in progress*), split because with `--allow-gap-moves` a
single `t` prices an indel at the substitution price without saying so. `t` is
still accepted and means both: `t_sub`/`t_indel` unset fall back to it.

Model fitted with adabmDCA 2.1; everything in `src/` is a thin
layer over the adabmDCA API (`load_model`, `compute_energy`, `get_freq_*`,
`encode_sequence`, …) — prefer extending that delegation over reimplementing.

## Read before editing

| file | why |
|---|---|
| `docs/NUMBERING.md`, `docs/SETUP.md` | the column mapping. **Read before indexing anything.** |
| `docs/DATA.md` | the two files not in git, and the 265-vs-260 trap |
| `docs/RESULTS.md` | the current results, re-derived from committed CSVs |
| `docs/METHODS.md` | the method in paper form, with the command behind each number |
| `analysis/figures_ladder/README.md` | the figure pipeline and what is model-derived |
| `hamming-restraint/README.md` | the active subproject |

## Layout

```
$TRYP_DATA_ROOT/                     (= repo parent by default)
├── iter_aln_v2.faa                  24 MB alignment, L=265, M=89,439   NOT in git
├── DCAmodel-serine_protease/        params.dat.gz (137 MB)             NOT in git
└── trypsin-specificity-switch/      this git repository
    ├── src/        potts, mcmc, ensemble, observables, sectors, sites, encoding,
    │               natural_stats — the library; flat modules, `package-dir = src`
    ├── scripts/    CLI drivers (run_*, plot_*, calibrate_mu, annotate_sectors);
    │               `_common.py` holds the shared argparse plumbing
    ├── analysis/   bash drivers (stage 1, sampling) + figures_ladder/,
    │               figures_clamp/ (stage 2, plotting; `make_all.py`)
    ├── hamming-restraint/   self-contained subproject, same shape
    ├── results/    per-chain observables + summaries, tracked; FASTA is not
    ├── figures/    committed PNG/PDF — results are readable without rerunning
    └── tests/      pytest, builds a small random Potts model on the fly
```

## Commands

```bash
./.venv/bin/python -m pytest -q        # ~53 tests, ~2 s, no model file needed
bash analysis/run_ladder.sh [stage]    # stage 1, sampling, ~1.5 h on mps
cd analysis/figures_ladder && ../../.venv/bin/python make_all.py [figNN]
```

- Python **≥3.10** (macOS `/usr/bin/python3` is 3.9 and will refuse). The venv is
  expected at `.venv/`; `analysis/_paths.sh` picks it up, override with `PY=`.
- Pass `--cache-model` on anything that loads `params.dat.gz` — 30 s → a few s.
- `--list-sites` loads the model, prints the resolved critic/allowed columns and
  exits. Use it before any long run that changes a site spec.
- `--device mps` on this machine (all committed results used it); `cuda` elsewhere.
- Heavy sampling and cheap plotting are **separate stages on purpose**. Do not
  fold a figure change into a resample.
- `analysis/run_ladder.sh` stages are **not** independent: `derive` reads consensus
  columns out of the `graft` output, so `consensus` cannot run before both.

## Conventions and traps

- **L = 265, critic column = 201, label `189`, grafting window `165–243`
  (residues 160–228).** Column 197 and `--allowed-cols 180-220` appear in older
  text and in the superseded `Results from a real run` section of the README —
  those belong to a deleted 260-column model. Never copy a reference row between
  alignments; regenerate with `scripts/extract_aligned_row.py --expect-length 265`.
- All MSA columns are **0-based**; `--mapping-base 1` only for Julia-written CSVs.
  Residue-label lookups are **case-sensitive** (`188b` and `188A` are distinct
  columns, 198 and 200).
- The gap is alphabet index 0 (adabmDCA `protein` tokens, gap first). Design runs
  now take `--allow-gap-moves`; a gap-frozen run freezes 12 of the 79 grafting
  columns and is not comparable in energy or acceptance with a gap-moving one.
- `P(S | critic)` does not depend on the residue at the critic site. A chain can end
  on D with high `P(S)`; `substitute_critic.py` writes the S in and rescores.
- **Nothing model-derived is portable between models**: μ, the λ/β ladders, the
  coupling ranking, the consensus column sets. The red sector is *not* model-derived
  and does carry over. Recompute per model, keep separate result trees.
- Sector enrichment denominators use the number of **mutable** columns, never the
  number of columns — otherwise a site restriction manufactures its own enrichment.
- Adding a module to `src/` means adding it to `py-modules` in `pyproject.toml`;
  a missing entry imports fine from a checkout and fails after `pip install`.
- `.gitignore` blocks `*.faa`, `*.dat.gz`, `*.cache.pt` and `results/**/
  final_sequences.fasta`. Do not force-add any of them.

## Work in progress

`hamming-restraint/` is untracked and `src/{ensemble,mcmc,observables}.py`,
`scripts/{run_ensemble,run_mcmc,substitute_critic}.py`,
`tests/{test_ensemble,test_mcmc}.py` carry uncommitted changes — all of it the
Hamming restraint. Properties the subproject relies on: each term is exactly a
field shift (at a site with reference residue `r`, 0 on `r`, `t_indel` on the
gap-vs-residue states, `t_sub` on the rest), so the sampled distribution is still
exactly a Potts model, ΔH stays O(1), and `−∂⟨d⟩/∂t = Var(d)` is used as a
measurement. `t = 0` must reproduce pre-restraint results bit-for-bit at the same
seed, and `t_sub = t_indel = t` must reproduce the single-knob runs bit-for-bit —
treat both as regression tests.

- `d_H` is over the **allowed columns**, not all L. The two differ only by a
  constant (frozen columns where the start differs from the reference), so no
  sampled distribution changes; as scripted the two coincide outright in every
  run — see `hamming-restraint/README.md`.
- Directory names: `run_dirname(lam, t, t_indel)` appends `_ti…` **only** when
  the indel price differs from the substitution price, so every already-sampled
  tree keeps its names. Read a tree through `parse_run_dirname_full`;
  `parse_run_dirname` returns `(lam, t_sub)` and drops the indel price.
- Both contributions are traced separately as `sub_to_wt` and `indel_to_wt`.
  The Gibbs sampler refuses any non-zero price rather than sample the wrong
  distribution, since it draws from the model's own fields.

## Writing here

- Technical register, no basics re-explained. Documentation in this repo states the
  *why* and the caveat beside every number; match that, and say what is superseded
  rather than deleting the record of it.
- Figures: 300 dpi PNG + vector PDF, via the shared `style.py` of each figure
  directory. λ is an ordered magnitude (shades of one hue), not a category.
- Don't commit or push unless asked.
