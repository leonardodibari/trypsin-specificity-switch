# The `iter_aln_v2` setup: mapping, sectors, gaps

Written for: whoever reruns or reviews this analysis. It records what the new
alignment is, how every site index was verified, and the two changes of method
that come with it. Results are not here — they are in [`RESULTS.md`](RESULTS.md).

---

## 1. Inputs

| file | what it is |
|---|---|
| `$TRYP_DATA_ROOT/iter_aln_v2.faa` | the alignment, **L = 265**, **M = 89 439** sequences |
| `data/iter_aln_v2_numbering.txt` | one residue label per column, 265 lines, chymotrypsinogen (3TGI) numbering with insertion codes |
| `data/iter_aln_v2_redsector.txt` | the red-sector columns, **0-based indices**, ordered from most to least important |
| `data/weights_iter_aln_v2.dat` | sequence-identity reweighting, **M_eff = 40 704.43** |

The alignment itself is too large for git and lives outside the repository, under
`TRYP_DATA_ROOT` — which defaults to the repository's parent directory. See
`docs/DATA.md`.

Derived here, all of them regenerable from the three files above:

| file | what it is |
|---|---|
| `data/wt_trypsin.fasta` | `P00763` — TRY2_RAT, the wild type, as its aligned row |
| `data/chymotrypsin_bovine.fasta` | `G1K1L9_P00766` — CTRA_BOVIN, chimera donor |
| `data/chymotrypsin_rat.fasta` | `P07338` — CTRB_RAT, chimera donor |
| `data/elastase_pig.fasta` | `P00772` — CELA1_PIG, the third sequence of the alignment figure |
| `data/mapping_iter_aln_v2.csv` | residue label → 0-based MSA column, all 265 |
| `data/sectors_iter_aln_v2.csv` | column → sector (`red` / `grafting` / `other`), with the red-sector rank |

All three reference sequences occur **verbatim** in `iter_aln_v2.faa` — their
ungapped forms match a row of the alignment exactly — so none of them had to be
realigned by hand.

---

## 2. The mapping, and how it was checked

The numbering file is monotone non-decreasing, runs from `16` to `245d`, has
exactly 265 entries, and carries 35 insertion-coded positions. Translating the
labels the study uses:

| label | MSA column (0-based) |
|---|---|
| **189** (the critic site) | **201** |
| 18 | 2 |
| 160 (start of the grafting segment) | 165 |
| 228 (end of the grafting segment) | 243 |

The mapping was not taken on trust. Serine proteases have landmarks whose
identity is fixed by catalysis, not by this alignment, and every one of them
lands where it should:

| label | col | TRY2_RAT | CTRA_BOVIN | CELA1_PIG | role |
|---|---|---|---|---|---|
| His57 | 45 | H | H | H | catalytic triad |
| Asp102 | 98 | D | D | D | catalytic triad |
| Ser195 | 207 | S | S | S | catalytic triad |
| Gly193 | 205 | G | G | G | oxyanion hole |
| Asp194 | 206 | D | D | D | buried salt bridge |
| **Asp189** | **201** | **D** | **S** | **S** | **the specificity site** |
| Ser214 | 227 | S | S | S | S1 pocket |
| Gly216 | 229 | G | G | V | S1 pocket |
| Gly226 | 241 | G | G | T | S1 pocket |

Three independent catalytic-triad residues at the right columns in three
different sequences is a strong check: a frame error of even one column would
break all of them at once. Column 201 carries `D` in trypsin and `S` in both
chymotrypsin and elastase — exactly the contrast the design exploits — and sits
in the `KDSCQGDSGGP` motif, the same landmark used in the previous alignment.

**One cosmetic oddity.** The numbering file contains both `188b` and `188A`
(lowercase, then uppercase) as separate labels. They are distinct columns (198
and 200), so nothing is ambiguous, but label lookups must be case-sensitive.

---

## 3. Sectors

Three disjoint classes, `data/sectors_iter_aln_v2.csv`:

| sector | columns | definition |
|---|---|---|
| `red` | **17** | the red-sector positions |
| `grafting` | **63** | the rest of the window 165–243 (residues 160–228) |
| `other` | **185** | the remaining alignment |

The red sector, most to least important, with its wild-type residue:

| rank | col | residue | WT | rank | col | residue | WT |
|---|---|---|---|---|---|---|---|
| 1 | 243 | 228 | Y | 10 | 165 | 160 | A |
| 2 | **201** | **189** | **D** | 11 | **2** | **18** | **G** |
| 3 | 241 | 226 | G | 12 | 242 | 227 | V |
| 4 | 228 | 215 | W | 13 | 233 | 219 | G |
| 5 | 229 | 216 | G | 14 | 235 | 221 | A |
| 6 | 204 | 192 | Q | 15 | 198 | 188a | G |
| 7 | 192 | 183 | V | 16 | 197 | 188 | G |
| 8 | 189 | 180 | M | 17 | 177 | 172 | Y |
| 9 | 193 | 184 | G | | | | |

Sixteen of the seventeen lie inside the grafting segment. **Column 2
(residue 18) is the exception** — it is the one red-sector position a graft of
160–228 would not transplant, and a run restricted to that window cannot
mutate it. The enrichment statistics account for this by using the number of
*mutable* columns in a sector as the denominator, never the number of columns
in it; otherwise a site restriction would manufacture an enrichment of its own.

---

## 4. Two changes of method

### 4.1 Gaps are now part of the mutational process

Previously, in a design run, the gap was inert: a gap column of the wild type
was dropped from the mutable set, and the gap was never proposed as a
replacement, so every sampled sequence carried the wild type's gap pattern
exactly. Only the generative check let gaps move.

Design runs now take `--allow-gap-moves`, which makes the gap an ordinary
state — a gap column may mutate, and any column may become a gap. Insertions
and deletions are sampled rather than assumed away.

This matters here more than it would have before: **12 of the 79 columns of the
grafting segment are a gap in the wild type**, so the old convention silently
froze a seventh of the window. The wild type has 42 gaps in all, out of 265
columns; none of them is in the red sector.

Consequences to keep in mind when reading the numbers:

* the state space is larger, so energies and acceptance rates are not
  comparable with a gap-frozen run of the same λ;
* `hamming_to_wt` now counts indels as well as substitutions;
* two new per-sequence columns, `gaps_opened` and `gaps_closed`, separate the
  two halves of a changed gap pattern, which a bare gap count cannot.

The scatter matrix gained `n_gaps`, the number of gaps in each sequence, so the
drift in gap content can be crossed with energy, `P(S)` and quality like any
other quantity. Under the old convention that column would have been a
constant.

### 4.2 Every mutation is labelled with its sector

`scripts/annotate_sectors.py` writes, for

* **endpoints** — into each `observables.csv`: `n_mut_red`, `n_mut_grafting`,
  `n_mut_other`, `n_mut_total`, `gaps_opened`, `gaps_closed`;
* **trajectories** — beside each `trajectory.csv`, a `trajectory_sectors.csv`
  whose every accepted move carries the `sector`, the `pdb_residue` label and
  the `red_rank` of the column it changed;
* **summaries** — `sector_enrichment.csv` (mutations per sector against that
  sector's share of the mutable columns) and `column_frequency.csv` (the
  per-column mutation rate, which says whether a sector's signal is really a
  handful of columns).

`scripts/plot_sector_mutations.py` draws the three panels from those.

---

## 5. The two design runs

Both use the same model, the same λ ladder (0, 10, 50, 100, 300, 1000), 1000
chains, 60 000 proposed moves, and both move gaps. They differ only in which
columns a move may touch:

| run | `--allowed-cols` | mutable columns | mutable red-sector columns |
|---|---|---|---|
| **A, grafting** | `165-243` | 79 | 16 of 17 (not column 2) |
| **B, free** | `0-264` | 265 | 17 of 17 |

Run A asks what the grafting segment can do. Run B removes the restriction and
asks the inverse question: given a free hand, **does the sampler go to the red
sector on its own?** That is what the enrichment panel is for.

**A caveat on reading run B.** With every column free and no bias, a chain does
not stay near the wild type — it relaxes to the model's own equilibrium, which
sits on the order of 160 substitutions away. So in run B the absolute mutation
counts are dominated by that drift, not by the specificity bias, and the
λ = 0 ensemble is the baseline the biased ones must be read against. Enrichment
against a uniform sampler is reported, but the sharper quantity is the *excess*
over λ = 0 at the same column.

---

## 6. Reproducing

```bash
bash analysis/run_ladder.sh          # every stage
bash analysis/run_ladder.sh graft    # one stage
```

Stages: `graft`, `derive`, `consensus`, `derive_large`, `consensus_large` — the
grafting-segment runs first, then this model's own consensus column sets read
back out of them, then the consensus runs. Run B, with every column free, is
`analysis/run_ladder_sectors.sh` (stages `free`, `annotate`), which also writes
the sector annotation. Results land under `results/ladder/`. The model is
trained separately (see `docs/TRAINING.md`).
