# Where the data lives

Written for: whoever picks this project up next.

Two files are too large for git and therefore are **not** in this repository: the
alignment (24 MB) and the fitted Potts model (137 MB). Everything else — the
reference sequences, the per-chain observables, the run summaries and every
rendered figure — is tracked and arrives with `git clone`.

---

## The contract

Code never hard-codes a path. It resolves the heavy data through one variable:

```
TRYP_DATA_ROOT    default: the repository's parent directory
```

That directory must contain exactly two things:

```
$TRYP_DATA_ROOT/
├── iter_aln_v2.faa                         24 MB   the alignment
└── DCAmodel-serine_protease/
    └── params.dat.gz                      137 MB   the fitted Potts model
```

Nothing else in that directory is read by the pipeline, though in practice the
model folder also carries its training by-products (below).

If either file is missing, every driver in `analysis/` stops **before** sampling
and prints both paths it looked for. That check is in `analysis/_paths.sh`, and
it exists because the alternative — discovering the problem forty minutes into a
GPU run, or worse, not discovering it at all — happened often enough to be worth
a preflight.

To put the data somewhere else:

```bash
export TRYP_DATA_ROOT=/mnt/scratch/tryp
bash analysis/run_clamp.sh
```

The figure pipelines read the same variable, so set it once in your shell.

---

## `iter_aln_v2.faa` — the alignment

89,439 sequences of the serine-protease family, aligned to **L = 265 columns**,
21-letter alphabet (`-ACDEFGHIKLMNPQRSTVWY`). Headers are bare UniProt
accessions. It was supplied by S. Yamamoto together with its column labelling
and its sector map, both of which **are** in this repo:

| file | what |
|---|---|
| `data/iter_aln_v2_numbering.txt` | one label per line, in column order; 0-based MSA column = line number − 1 |
| `data/iter_aln_v2_redsector.txt` | red-sector columns, most to least important (floats; cast to int) |
| `data/sectors_iter_aln_v2.csv` | the full sector annotation |
| `data/mapping_iter_aln_v2.csv` | MSA column ↔ literature residue number |

Read [`NUMBERING.md`](NUMBERING.md) before you index anything. The labelling
convention is load-bearing and the off-by-one is real: the critic site is MSA
column **201**, which is label `189` in chymotrypsinogen numbering.

### The trap this alignment sets

An earlier alignment had **260** columns. A reference sequence taken from it
loads cleanly, looks like a protein, and is silently wrong at every column after
the first indel. The repository shipped exactly such a stale
`data/wt_trypsin.fasta` until November 2026; it was caught only because
`run_ensemble.py` refuses a wild type whose length does not match the model's L.

So every single-sequence file under `data/` is **generated**, not copied:

```bash
python scripts/extract_aligned_row.py \
    --alignment $TRYP_DATA_ROOT/iter_aln_v2.faa \
    --id P00763 --expect-length 265 \
    --output data/wt_trypsin.fasta \
    --description "TRY2_RAT P00763 anionic trypsin, aligned row of iter_aln_v2 (L=265)"
```

`--expect-length 265` is the point of the exercise. The four references in the
repo, and the sanity check that each one carries the residue it should at the
critic column:

| file | accession | col 201 | |
|---|---|---|---|
| `data/wt_trypsin.fasta` | P00763 | `D` | rat anionic trypsin — the MCMC wild type |
| `data/chymotrypsin_rat.fasta` | P07338 | `S` | rat chymotrypsin B |
| `data/chymotrypsin_bovine.fasta` | P00767 | `S` | bovine chymotrypsinogen B |
| `data/elastase_pig.fasta` | P00772 | `S` | pig elastase 1 |

Bovine chymotrypsinogen **A** (P00766), the textbook reference, is not in this
alignment; P00767 (chymotrypsinogen B) is used instead.

---

## `DCAmodel-serine_protease/` — the model

A bmDCA Potts model fitted with [adabmDCA 2.1](https://github.com/spqb/adabmDCApy).

* **Trained on a 71,551-sequence train split**, leaving 17,888 held out. Both
  splits ship inside the model directory as `iter_aln_v2.{train,test}.fasta`;
  together they are exactly the 89,439 rows of the alignment.
* That held-out set is what lets figure F1 and SI S1 be **predictions** rather
  than consistency checks, and it is the whole reason this model replaced its
  predecessor. Never regenerate the split by guessing — the figures fall back to
  the whole alignment and say so, because a fabricated split would make the one
  figure meant to answer the circularity objection quietly dishonest.
* μ = 422.62, wild-type E = 82.9, wild-type CDE = 0.445.

Only `params.dat.gz` is required. The rest of the directory is training
provenance — `history.csv`, `adabmDCA.log`, `events.jsonl`, `training.json`,
`chains.fasta`, `weights.dat`, plus `training_plots/` and `sampling/` — and it is
worth keeping, but no code reads it.

`params.dat.gz.cache.pt` (124 MB) is a torch cache the scripts write on first
load with `--cache-model`. It is derived, and safe to delete; it will be rebuilt.

### Fingerprinting

Both figure pipelines stamp each results directory with a fingerprint of the
parameter file (size + sha256 of the first and last 1 MB) and **refuse to build**
if it no longer matches. Deliberately not mtime: copying a file changes mtime
without changing the model, and a check that cries wolf gets switched off.

The failure this prevents: point the config at a new model, rebuild only the
figures, and you get a new model's scores drawn over an old model's designed
sequences. Nothing looks wrong. Every number is plausible. Every number is
meaningless.

---

## What is tracked, and what is not

| | size | in git? |
|---|---|---|
| source, scripts, docs | small | yes |
| `data/` reference sequences, sector maps, weights | 4 MB | yes |
| `figures/ladder/`, `figures/clamp/` rendered PNG + PDF | 28 MB | **yes** |
| `results/**/observables.csv`, `summary.json`, `trace.csv` | 13 MB | **yes** |
| `results/**/final_sequences.fasta` designed ensembles | 36 MB | no |
| `iter_aln_v2.faa`, `params.dat.gz` | 161 MB | no |

The rendered figures are committed on purpose: a successor can see every result
without a GPU, without the alignment and without re-running anything.

The designed ensembles are the one derived product left out. That is a
deliberate trade, and it has a cost worth knowing: four figure scripts read
them (`figures_ladder/data.py`, `figures_ladder/fig05_table.py`,
`figures_clamp/data.py`, `figures_clamp/c03_pocket.py`), so **from a bare clone
the figures cannot be re-rendered until the ensembles are regenerated**. The
committed PNGs are there to look at; regenerating them means re-running Stage 1:

```bash
bash analysis/run_clamp.sh     # ~18 min on an M-series GPU, all four conditions
bash analysis/run_ladder.sh    # ~1.5 h
```

If you would rather trade 36 MB for a repo that rebuilds its own figures from a
clean clone, delete this line from `.gitignore` and commit them:

```
results/**/final_sequences.fasta
```
