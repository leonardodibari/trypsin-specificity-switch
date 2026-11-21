# Residue numbering in `iter_aln_v2` — the definitive mapping

Settled 17 Nov 2026 by S. Yamamoto, who supplied the alignment and the sector map.

## The convention

> Upper case = positions that exist in the **historical numbering, in the literature**.
> Lower case = the columns that are **extra in this alignment**.

So a label is read as follows:

| label form | example | meaning |
|---|---|---|
| bare number | `189` | a position in the standard (chymotrypsinogen) numbering |
| number + UPPER case letter | `188A` | an insertion code that is part of the standard numbering |
| number + lower case letter | `188a` | a column this alignment adds; **no counterpart in the literature** |

`iter_aln_v2_numbering.txt` is one label per line, in column order, 1-indexed by
line. **0-based MSA column = line number − 1.** That off-by-one is the whole
source of the confusion below, so it is worth stating twice.

## How the convention partitions the alignment

Of 265 columns, 35 carry a letter suffix:

- **34 are lower case** — alignment-specific insertions (`36a–36d`, `61a–61e`,
  `74a`, `96a/b`, `122a`, `127a/b`, `133a`, `144a`, `146a/b`, `149a/b`,
  `173a–173d`, `188a`, `188b`, `204a`, `218a`, `223a`, `245a–245d`).
- **exactly 1 is upper case** — `188A`, MSA column 200.

That lone upper-case code is a good check on the convention: `188A` is a real
insertion code in chymotrypsin numbering, and it is the only letter-suffixed
label here that the literature would recognise.

## The region that caused the question

| MSA col | label | WT (TRY2_RAT) | gaps in MSA | kind |
|---|---|---|---|---|
| 195 | `186` | L | 0.7 % | literature |
| 196 | `187` | E | 0.7 % | literature |
| 197 | `188` | G | 3.1 % | literature |
| **198** | **`188a`** | **G** | **34.5 %** | **alignment-specific** |
| 199 | `188b` | – | 90.5 % | alignment-specific |
| 200 | `188A` | K | 13.6 % | literature insertion code |
| 201 | `189` | D | 1.3 % | literature |

## What this means for the analysis

**The red sector position is `188a` = MSA column 198.** That is what
`iter_aln_v2_redsector.txt` selects and what the whole analysis used, so nothing
needs re-running. Column 200 (`188A`) is *not* in the red sector.

**The design target is unaffected.** The site is MSA column 201 = label `189` —
a plain literature position, Asp189 in rat trypsin, Ser in chymotrypsin.

**Only one of the 17 red-sector columns is alignment-specific.** The other 16 are
plain literature positions:

| col | label | WT | col | label | WT |
|---|---|---|---|---|---|
| 2 | 18 | G | 204 | 192 | Q |
| 165 | 160 | A | 228 | 215 | W |
| 177 | 172 | Y | 229 | 216 | G |
| 189 | 180 | M | 233 | 219 | G |
| 192 | 183 | V | 235 | 221 | A |
| 193 | 184 | G | 241 | 226 | G |
| 197 | 188 | G | 242 | 227 | V |
| **198** | **188a** | **G** | 243 | 228 | Y |
| 201 | 189 | D | | | |

**No conclusion depended on the label anyway.** All four columns of the 188
region are rewritten by the design, so whichever one is "the" sector position,
the loop is remodelled either way (fraction of the 1000 chains that changed the
column, grafting run):

| λ | `188` (197) | `188a` (198) | `188b` (199) | `188A` (200) |
|---|---|---|---|---|
| 0 | 0.12 | 0.02 | 0.01 | 0.08 |
| 100 | 0.75 | 0.31 | 0.52 | 0.34 |
| 300 | 0.82 | 0.48 | 0.77 | 0.53 |
| 1000 | 0.88 | 0.63 | 0.96 | 0.77 |

## One caveat when reporting

`188a` has no literature counterpart and is a gap in 34.5 % of the family. When
writing this up for anyone working from standard trypsin numbering, it should be
described as "an inserted position in the 186–190 loop, between 188 and 188A",
not as a residue number they can look up. The same applies to any other
lower-case label that turns up in a result.
