# Finish the trypsin → chymotrypsin deck

The deck is at 17 main slides + 10 appendix. Slides 10 and 11 are placeholders
("to come"). Fill them, add one appendix slide, and make three edits. Keep the
house style exactly as it is — nothing else changes.

`deck_final_figures.zip` has `figures/` (three new figures, named by the slide
they belong to) and two CSVs with the full numbers behind them. Place the
figures as they are; do not redraw or recolour.

All numbers below are at **λ = 300, β = 1**, 1000 chains × 60 000 steps.

---

## SLIDE 10 — replace the placeholder

**Title:** Designing to a mutation budget
**Figure:** `figures/10_site_budget.png` (wide, under the bullets)

**Bullets:**
- The mutation count is set by how many columns may move — not by λ
- 19 columns are rewritten in ≥ 95 % of the chains; allow only those
- Below ~19 columns the library collapses
- The literature positions get most of the way, but stop short

**Table (small, right or under the figure):**

```
  site set            cols   muts   P(S)   ≥.80       Q   unique
  red sector            16   12.5   0.70     0 %   −89.3      410
  literature (loops)    17   15.1   0.78     0 %   −95.9      228
  consensus-19          19   18.6   0.853  100 %  −106.1      984
  coupling-20           20   17.5   0.85   100 %   −97.8      998
  consensus-23          23   22.1   0.88   100 %  −105.1     1000
  full window           79   34.4   0.93   100 %   −89.3     1000
```

**Small note under the table:** the natural family sits at Q = −85.6 and the wild
type at −103.0, so consensus-19 is deeper than either.

**Banner:** 19 columns, 19 mutations — **every chain switches**

---

## SLIDE 11 — replace the placeholder

**Title:** The actual mutations
**Figure:** `figures/11_design_mutations.png` (wide and centred, no bullets)

**Caption:** consensus-19 at λ = 300 · bold = the five load-bearing positions

**Three notes in small text:**
- D189S is the target and appears in 100 % of designs; Y228F is also universal.
- Not a pure point-mutant library: one insertion after residue 195 and the
  deletion of residue 201. Net length unchanged, 223 residues.
- A no-indel variant exists — 18 positions, P(S) = 0.82, 93.5 % of chains over
  the threshold, Q = −97.6. The two indels are worth 8.5 units of Q.

**Banner:** 17 substitutions, **one insertion, one deletion**

Reporting caveat, ties to appendix A3: `223a` has no literature counterpart —
call it the deleted position in the 220–224 loop, never a residue number.

---

## NEW SLIDE 12 — insert after slide 11, renumbering what follows

This is the parallel of slide 7 ("The λ = 300 library") for the library actually
being recommended, and it is the deck's real climax: the 19-mutation designs
against the natural family on the same six quantities.

**Title:** The 19-mutation library
**Figure:** `figures/12_consensus19_pairs.png` (wide, under the bullets)

**Bullets:**
- One dot = one design · grey = the natural family
- 984 distinct designs, 19 substitutions each
- P(S) = 0.853, every chain over the threshold
- Q = −106, deeper than the wild type's −103

**Caption:** consensus-19 at λ = 300 · 1000 chains · μ = 331

**Small note:** the distance to the nearest natural S-carrier falls in discrete
bands because it is measured over only 18 compared columns — each step is one
residue, about 5.6 %.

**Banner:** 19 mutations, **below the natural family in both E and Q**

Point out when presenting, but do not put in a bullet: the gap panel shows the
designs sitting at the wild type's 42 while the natural family spreads around
30. These are substitutions inside trypsin's own gap pattern, apart from the one
insertion and one deletion on the previous slide.

---

## NEW APPENDIX SLIDE A11 — after A10

**Title:** Appendix · which positions carry the switch
**Figure:** `figures/A11_ablation.png`

**Bullets:**
- Leave-one-out over the 19 positions, one rerun each
- 226 is the keystone — removing it alone takes the library from 100 % over the
  threshold to 0 %, and from 989 unique designs to 564
- Five positions carry the specificity: 226, 221, 228, 222, 218
- The other fourteen cost ≤ 0.015 each — but the drops compound

**Small text:** removing the seven cheapest at once gives a 13-column set with
237 unique designs, and a 10-column set gives 16. Every chain then mutates every
available position: there is no library left, only a single design with jitter.
That is why 19 is a floor rather than an arbitrary choice.

**Banner:** Five positions carry the switch — **the other fourteen carry the library**

---

## OPTIONAL APPENDIX SLIDE A12

`figures/A12_consensus19_pairs_all.png` — the same corner plot for all three λ
on the 19 columns, showing the ladder inside the restricted set. Only include it
if there is room; the single-λ version in the main line carries the argument.

---

## EDIT 1 — slide 12, now slide 13 ("Where the mutations land")

Add the decisive line to the existing enrichment slide:

> Allowing **only** the red sector cannot switch the enzyme: P(S) = 0.70, and
> 0 % of chains reach the threshold at any λ. Only 5 of the 19 consensus columns
> are red-sector, and 10 of the 17 red columns are never touched at all.

This is the hard version of the caveat already on the slide — the sector is
enriched per opportunity *and* insufficient on its own.

---

## EDIT 2 — the Results slide (now slide 18)

Replace the fourth paragraph ("At 34 mutations the designs are about double…
Designing to a budget is the next step.") with the answer:

> **Designing to a budget works.** Restricting to the 19 columns the design
> itself uses gives 19 substitutions, P(S) = 0.853 with every chain over the
> threshold, Q = −106.1 — deeper than the wild type — and 984 distinct designs.
> Below 19 columns the library collapses to a single sequence.

And add a fifth, short paragraph:

> The model recovers the published positions as most of the answer: Hedstrom's
> loops take P(S) from 0.005 to 0.78 on their own, and stop just under the 0.80
> threshold. The remaining ~19 positions finish the job.

---

## EDIT 3 — one factual correction wherever it appears

If any slide says the converted enzyme used **Y172W**, drop the "W". Residue 172
is a specificity determinant (Hedstrom & Perona, a separate paper), but the
specific tryptophan substitution is not confirmed. The 1992 *Science* paper
(Hedstrom, Szilagyi & Rutter, 255:1249–1253) establishes D189S plus the two
surface loops **185–188** and **221–225**; its converted enzyme matched
chymotrypsin's catalytic rate but had impaired substrate binding.

---

## OPTIONAL EDIT — only if the deck is running long

With the new slide 12 in place, slide 7 ("The λ = 300 library") is a corner plot
of a library that is no longer the recommendation. It still earns its place by
motivating the budget — it is where "34 mutations, no chain below 26" is stated
— so if it moves to the appendix, that sentence has to move onto slide 10.

---

## Do not add

The compression experiment and the full ablation table beyond slide A11. One
line on slide 10 ("below ~19 columns the library collapses") carries it.
