# Build a Keynote deck: trypsin → chymotrypsin specificity switch

**Read this whole file before starting.** It contains (1) what you are being asked to
do, (2) the house style the deck must follow, (3) the science, so you can write captions
and takeaways that are actually true, and (4) a slide-by-slide specification.

You are also given **`trypsin_switch_figures.zip`**. Unzip it; `figures/` holds the
eleven figures the deck uses, named by the slide they belong to, and `extras/` holds
figures you may pull in if you decide a slide needs them. Every figure is a finished
matplotlib render — **do not redraw, recolour or crop them.** Place them as they are.

---

## 1. The task

Produce a **Keynote presentation, 16:9 (1920 × 1080 px)**, 13 slides plus a 3-slide
appendix, in the house style described in §2. The audience is a group meeting of
physicists and computational biologists who know what a Potts model is but have not seen
this project. It is a results talk, roughly 15 minutes.

Write it as a `.key` file if you can; otherwise produce the deck in whatever slide format
you can actually deliver, keeping the layout spec below exactly.

Two rules that matter more than anything else:

- **One idea per slide.** The bottom banner carries that idea in one line. If you cannot
  write the banner, the slide is doing too much.
- **Short bullets.** Three or four, each a fragment, not a sentence. The figure does the
  talking. Never put a number in a bullet that is not also visible in the figure or the
  banner.

---

## 2. House style — follow the previous deck exactly

The earlier deck in this project established the layout below. Reproduce it.

### Canvas and grid

| element | spec |
|---|---|
| slide | 1920 × 1080, white background |
| title | top-left, x ≈ 90, y ≈ 58, bold, ≈ 44 pt, black |
| bullet column | x ≈ 95, top y ≈ 300, width ≈ 470, ≈ 25 pt |
| figure (with bullets) | x ≈ 640, width ≈ 1190 |
| figure (no bullets) | x ≈ 170, width ≈ 1580 — centred, wide |
| caption | directly under the figure, centred on it, ≈ 17 pt, grey |
| takeaway banner | a thin-bordered rectangle, 1700 × 74, centred, y ≈ 940 |
| banner text | centred inside it, ≈ 24 pt |
| slide number | small, bottom-left, grey |

### Palette

Only three colours. Nothing else.

- ink / text **`#1a1a1a`**
- accent blue **`#2b57c3`** — used for the *one or two emphasised words* in the banner,
  and nowhere else in the text
- grey **`#7a7a7a`** — captions and the slide number

The figures carry their own blue sequential scale (light = λ small, dark = λ large) and a
grey for the natural alignment. That scale is why the accent is blue; keep it.

### Voice

- Titles are sentence case. A middle dot separates a title from its qualifier:
  *"The goal · switch the enzyme's specificity"*, *"Candidate sequences · all λ"*.
- Captions state the run conditions, flatly: *"1000 chains per λ · μ = 331"*.
- Banners are short declaratives with one or two words in blue:
  *"Temperature is not the useful knob — the **bias** is"*.
- Use λ, μ, β, σ as symbols, not spelled out. Use "·" as a separator, "⇒" for implication.
- Equations are typeset as real equations (Keynote's equation editor takes LaTeX), not
  as images and not as plain text.

---

## 3. What was done — the science

### 3.1 The problem

Rat trypsin (UniProt P00763, TRY2_RAT) cuts after basic residues because **Asp189** sits
at the bottom of its specificity pocket. Chymotrypsin has **Ser** there and cuts after
bulky hydrophobics. The goal is to redesign trypsin so the model *prefers* serine at that
site — and to do it with a library of sequences, not one point mutant, because a single
D189S swap is known not to transfer the specificity on its own.

In the wild-type sequence the model gives that site **P(S) ≈ 0.005**. The target is
**P(S) ≥ 0.80**.

### 3.2 The model

A Potts model (bmDCA / adabmDCA, PCD strategy) trained on an iteratively built alignment
of the trypsin family:

| item | value |
|---|---|
| alignment | `iter_aln_v2.faa`, L = 265 columns, M = 89 439 sequences, 12.4 % gaps |
| reweighted depth | M_eff = 40 704 |
| model quality | Pearson on connected correlations C_ij = **0.826**, regression slope 0.950 |
| critic site | MSA column **201** = residue **189** |
| wild type | TRY2_RAT, 42 gaps in its aligned row |

Energy, with the coupling sum over ordered pairs:

> E(σ) = − Σ_i h_i(σ_i) − Σ_{i<j} J_ij(σ_i, σ_j)

### 3.3 The sampler

Metropolis MCMC over sequences, with a bias that pushes serine into the critic site:

> π(σ) ∝ exp{ −β [ E(σ) − λ · log P(S | i*, σ) ] }

`P(S | i*, σ)` is the model's conditional probability of serine at the critic site given
the rest of the sequence. λ = 0 is plain sampling from the model; larger λ pushes harder.
β = 1 throughout the runs shown here.

Two details that matter:

- **Gaps are in the move set.** Insertions and deletions are sampled, not frozen. The
  conditional correctly treats a gap at the critic column as not-an-S.
- **Only chosen columns may mutate.** This is the key design knob, see §3.5.

### 3.4 Sample quality, Q

A low DCA energy can be misleading: a sequence scores low partly by sitting in a rigid,
low-entropy context that the training family happens to cover densely. Correct for it by
subtracting the **mean conditional entropy** CDE — how constrained each site is when the
rest of the sequence is held fixed:

> Q_μ = E_DCA − μ · CDE   (lower is better, like a free energy)

μ is chosen by minimising Var(Q) over the natural alignment, which is the same as the
regression slope of E on CDE:

> μ* = Cov(E, CDE) / Var(CDE) = **331**

At μ = 331 this removes **94 %** of the energy variance across the natural family, so Q
compares sequences fairly where the raw energy does not. The natural alignment sits at
**⟨Q⟩ = −85.6**; that is the bar a design has to beat.

### 3.5 The two runs

Same model, same bias, 1000 chains × 60 000 steps per λ, λ ∈ {0, 10, 50, 100, 300, 1000}.
They differ only in which columns are allowed to mutate:

- **A · grafting segment** — MSA columns 165–243, i.e. residues 160–228, **79 sites**.
  This is the segment around the pocket; it contains 16 of the 17 "red sector" columns.
- **B · all sites free** — all 265 columns.

After the run, S is written into column 201 in any chain that does not already carry it,
and each design is matched to its nearest natural serine-carrier.

### 3.6 Endpoints — grafting segment (the real designs)

| λ | ⟨E_DCA⟩ | ⟨Hamming to WT⟩ | ⟨P(S)⟩ | chains ≥ 0.80 | ⟨Q⟩ | ⟨gaps⟩ | → nearest natural S-carrier |
|---|---|---|---|---|---|---|---|
| 0 | 175.2 | 19.3 | 0.011 | 0 % | −96.2 | 41.8 | 33.5 % |
| 10 | 201.3 | 25.6 | 0.492 | 0 % | −94.6 | 41.0 | 38.8 % |
| 50 | 214.2 | 28.6 | 0.766 | 25 % | −93.0 | 40.8 | 42.4 % |
| 100 | 222.1 | 30.6 | 0.848 | 96 % | −91.8 | 40.4 | 44.2 % |
| **300** | **236.1** | **34.4** | **0.929** | **100 %** | **−89.3** | **39.4** | **46.1 %** |
| 1000 | 255.9 | 40.0 | 0.971 | 100 % | −85.7 | 36.9 | 47.1 % |
| *natural* | *321.1* | *163.3* | *0.104* | *4 %* | *−85.6* | *32.9* | — |

All 1000 designs are distinct at every λ.

**λ = 300 is the operating point**: every chain switches, the designs are 34 mutations
from trypsin, and ⟨Q⟩ = −89.3 is still better than the natural family's −85.6.
λ = 10 is the failure case — it does not reliably switch (only 2 of 8 single chains
reached threshold in 200 000 steps).

### 3.7 Endpoints — all sites free (the control, not designs)

| λ | ⟨E_DCA⟩ | ⟨Hamming⟩ | ⟨P(S)⟩ | ⟨Q⟩ |
|---|---|---|---|---|
| 0 | 320.9 | 108.4 | 0.019 | −58.8 |
| 100 | 387.5 | 148.5 | 0.957 | −69.0 |
| 300 | 379.2 | 160.1 | 0.984 | −70.9 |
| 1000 | 355.5 | 166.9 | 0.994 | −69.6 |

Freeing all the sites **destroys the design**. Even unbiased, the chains wander 108
mutations away; by λ = 1000 they sit 167 away — further from trypsin than an average
natural homologue is (163). Their energies and distances look natural, but **Q is 15–27
units worse than natural**: they have equilibrated into a high-entropy region the model
scores poorly. They are generic serine proteases, not redesigned trypsin.

**The switch itself is easy either way. Restricting the mutable sites is what keeps the
result a design.**

### 3.8 Where the mutations land

Enrichment = (share of mutations in a sector) / (share of the *mutable* columns it owns).
1.0 = indifferent.

| λ | red sector (grafting run) | red sector (free run) | grafting window outside red (free) | rest of alignment (free) |
|---|---|---|---|---|
| 0 | **0.66** | **0.70** | 0.88 | 1.07 |
| 100 | **1.32** | 1.25 | 0.92 | 1.00 |
| 300 | 1.32 | 1.19 | 0.95 | 1.00 |
| 1000 | 1.24 | 1.16 | 0.98 | 0.99 |

Three things follow, and the slide should say all three:

1. **Unbiased, the red sector is avoided** (0.66). It is the most constrained part of the
   molecule, so unguided moves there are costly.
2. **The bias specifically recruits it**, peaking near λ = 100–300. The effect is real but
   modest — a 25–32 % excess, not a landslide.
3. **The grafting window itself is not special.** In the free run the rest of the alignment
   sits at exactly 1.00 and the window outside the red sector is slightly *depleted*. Only
   the red sector is enriched.

Per column at λ = 1000: residues 228, 189, 226, 221 are rewritten in 100 % of chains, 160
in 98.5 %, 172 in 96.1 %, 188 in 87.6 %; four red positions barely move (216, 183, 184, 215,
all under 7 %).

### 3.9 Gaps

Gaps really are part of the mutational process. In the grafting run the net change at
λ = 1000 is **−5.94 gaps per chain** (5.94 wild-type gaps closed, 0.85 opened), against
−0.25 at λ = 0. The ensemble moves from the wild type's 42 gaps towards the natural mean
of 32.9. Chains gapped the critic column itself in at most 0.8 % of cases, and never at
λ ≥ 300.

### 3.10 What writing the serine in costs

At λ = 0, forcing S into column 201 is uphill: ΔE = +4.71, ΔQ = +1.67. From λ = 10 it is
already downhill (ΔE = −0.94). By λ = 1000, 97 % of chains have arrived at S on their own,
so the substitution is almost a no-op. Where the bias has done its work, the serine is
essentially free.

### 3.11 Two caveats worth a slide in the appendix

- **The model stops at Pearson 0.826, and that is not the new alignment's fault.**
  Retraining the *old* alignment from scratch under the identical protocol reaches
  0.821 / slope 0.946 — the same place by the same route, even though it carries 37 % less
  effective data (M_eff 25 488 vs 40 704). The 0.95 attributed to the old model therefore
  came from a different protocol or a different statistic, not from better data. The
  regression *slope* does reach 0.95 on both.
- **Residue numbering.** The alignment mixes literature numbering with alignment-specific
  insertions: upper case suffixes (`188A`) are real insertion codes from the literature,
  lower case (`188a`) are columns this alignment adds and have no counterpart a reader can
  look up. The red-sector position in the 186–190 loop is **`188a` = MSA column 198** and
  should be described as *"an inserted position in the 186–190 loop"*, never as a residue
  number. The design target is unambiguous: column 201 = residue 189. Confirmed by
  S. Yamamoto, 17 Nov 2026.

---

## 4. Slide-by-slide specification

Bullets are given as the actual text to use. Captions and banners are given verbatim —
use them, or improve them without changing what they claim. **Bold in a banner marks the
words to set in accent blue.**

---

### Slide 1 — The goal · switch the enzyme's specificity

- **Figure:** `figures/01_goal.png`, wide and centred (no bullets)
- **Caption:** residue 189, the bottom of the specificity pocket
- **Banner:** Make the sequence prefer **serine** instead of aspartate

---

### Slide 2 — The method · bias the sampling toward serine

No figure. Two equations, centred, the first large:

```
π(σ) ∝ exp{ −β [ E(σ) − λ · log P(S | i*, σ) ] }
E(σ) = − Σ_i h_i(σ_i) − Σ_{i<j} J_ij(σ_i, σ_j)
```

- **Bullets:** Score with the Potts energy · Push serine with strength λ · Mutate only the
  chosen sites · λ = 0 is plain sampling
- **Small note under the equations:** i* is the site — residue 189, MSA column 201.
  Gaps are in the move set, so insertions and deletions are sampled too.
- **Banner:** λ = 0 is plain sampling · **larger λ** pushes in serine

---

### Slide 3 — The model and the two runs

No figure. Bullets on the left, the two runs laid out as a labelled A/B block on the right.

- **Bullets:** Potts model retrained on iter_aln_v2 · L = 265 · M = 89 439 · M_eff = 40 704 ·
  Pearson on C_ij = 0.826, slope 0.95
- **A · grafting segment** — MSA columns 165–243 — 79 sites, residues 160–228
- **B · all sites free** — all 265 columns
- **Footline:** 1000 chains × 60 000 steps per λ, λ ∈ {0, 10, 50, 100, 300, 1000}
- **Banner:** Same model, same bias — only the **mutable sites** differ

---

### Slide 4 — Simulation run

- **Figure:** `figures/04_ensemble_trace.png` (four panels: E_DCA, Hamming to WT, P(S),
  Q, each against accepted moves per chain, for λ = 0, 100, 1000)
- **Bullets:** MCMC with the bias, on the grafting segment · Stronger bias → more serine ·
  Energy overshoots, then settles
- **Caption:** 1000 chains per λ · μ = 331
- **Banner:** λ ≥ 300 takes **every chain** past P(S) = 0.8

Point out when you present it, but do not put it in a bullet: the energy and Q both
overshoot around 50 accepted moves and relax back. The chains find the serine fast and
then repair the context around it.

---

### Slide 5 — Inverse sample quality

- **Figure:** `figures/05_mu_calibration.png` (energy histogram of the natural alignment;
  E vs CDE with the fitted slope; the Var(Q) scan that picks μ; E and Q overlaid after
  centring and scaling)
- **Bullets:** A low energy can just mean a rigid, low-entropy context · Subtract the
  context entropy · Lower Q is better
- **Equations, small, bottom left:** `Q_μ = E − μ·CDE` and
  `μ* = Cov(E, CDE) / Var(CDE) = 331`
- **Caption:** μ = 331 flattens Q across the natural family · 94 % of the energy variance removed
- **Banner:** Q compares sequences **fairly** — the energy alone does not

---

### Slide 6 — Candidate sequences · all λ

- **Figure:** `figures/06_candidates_all_lambdas.png` (5 × 5 corner plot: E_DCA, Hamming to
  WT, P(S), Q, distance to nearest natural S-carrier; blue = designs, darker = larger λ,
  grey = natural alignment)
- **Bullets:** One dot = one design · Darker = stronger bias · Grey = the natural family ·
  S written in where missing
- **Caption:** grafting segment · 1000 chains per λ
- **Banner:** More serine ⇒ **farther from the wild type**, farther from natural S-carriers

---

### Slide 7 — Choosing λ

No figure. The table below, set in a monospaced face, left-aligned, on a single slide.

```
  λ        P(S)   ≥.80       Q    Ham.WT   unique   → natural
  0        0.01     0 %   −96.2      19     1000       33 %
  10       0.49     0 %   −94.6      26     1000       39 %
  50       0.77    25 %   −93.0      29     1000       42 %
  100      0.85    96 %   −91.8      31     1000       44 %
  300      0.93   100 %   −89.3      34     1000       46 %
  1000     0.97   100 %   −85.7      40     1000       47 %
  natural  0.10     4 %   −85.6     163        —        —
```

- **Two lines under the table:** At λ = 300 every chain switches, Q still beats the natural
  family, and all 1000 designs are unique, 34 mutations from trypsin.
- **Banner:** λ = 300 · high P(S), low Q, **diverse**, close to trypsin

---

### Slide 8 — The chosen library · λ = 300

- **Figure:** `figures/08_candidates_lam300_grafting.png` (same corner plot, λ = 300 only,
  against the natural family)
- **Bullets:** 1000 designs, all unique · P(S) = 0.93, every chain over the threshold ·
  34 mutations from trypsin · Q = −89, better than natural
- **Caption:** a tight island to the left of the natural cloud
- **Banner:** Low energy, high specificity, **still recognisably trypsin**

---

### Slide 9 — Writing the serine in

- **Figure:** `figures/09_critic_substitution_deltas.png`, wide (ΔE and ΔQ of inserting S,
  over only the chains that did not already carry it, per λ)
- **Line above the figure:** What does it cost to insert S into the chains that do not have it?
- **Caption:** ΔE and ΔQ over the chains that did not already carry S · μ = 331
- **Banner:** Where the bias has done its work, the serine is **essentially free**

Note for the caption or the talk: the λ = 0 distribution sits to the *right* of zero
(uphill, ΔE ≈ +4.7); every biased λ sits to the left. At λ = 1000 only 3 % of chains
needed the substitution at all.

---

### Slide 10 — Restricting the sites is what keeps the design

Two figures side by side, each ≈ 620 wide, with a label above and a line below.

- **Left:** `figures/08_candidates_lam300_grafting.png` — label *grafting segment — 79 sites*,
  under it *34 from WT · Q = −89, beats natural*
- **Right:** `figures/10_candidates_lam300_free.png` — label *all sites free — 265 sites*,
  under it *160 from WT · Q = −71, worse than natural*
- **Banner:** Freed, the chains drift **further from trypsin** than a natural homologue

---

### Slide 11 — Only the red sector is recruited

- **Figure:** `figures/11_sector_mutations.png` (mutations per sector vs λ; enrichment vs λ
  on a log axis with the indifferent line at 1.0; per-column mutation frequency at λ = 1000)
- **Bullets:** Unbiased, the red sector is avoided — it is the most constrained region ·
  The bias reverses that · The rest of the alignment stays at exactly 1.00
- **Equation, small:** `r_s = (m_s / m) / (n_s / n)`
- **Caption:** mutations per sector, divided by that sector's share of the mutable columns
- **Banner:** Red sector **0.66 unbiased → 1.32** at λ = 300

---

### Slide 12 — Gaps are part of the design

- **Figure:** `figures/12_gap_abundance.png` (gaps per sequence by λ as violins, against the
  wild type's 42 and the natural mean of 32.9; gaps opened vs closed; per-column gap
  frequency at λ = 1000)
- **Bullets:** Insertions and deletions are sampled, not frozen · Stronger bias closes
  wild-type gaps · −5.9 gaps per chain at λ = 1000
- **Caption:** grafting segment · outlined bars are gaps in the wild type
- **Banner:** The designs drift from trypsin's 42 gaps towards the **natural** 33

---

### Slide 13 — Results

No figure. Three short paragraphs, generously spaced, plus one small line at the foot.

1. λ = 300 on the grafting segment gives 1000 unique designs that all carry S, 34 mutations
   from trypsin, with sample quality better than the natural family.
2. Restricting the mutable sites is what preserves the design. Freed, the chains
   equilibrate into generic serine proteases 160 mutations away.
3. The bias recruits the red sector and nothing else — the rest of the alignment sits at
   exactly 1.00. Unbiased, the red sector is avoided.

- **Footline, small:** Numbering confirmed by S. Yamamoto: the sector position is 188a = column 198.
- **Banner:** A chymotrypsin-like pocket, **still close to trypsin**

---

## 5. Appendix slides

Same style. These answer questions the audience will ask.

### A1 — Why does the new model stop at 0.83?

- **Figure:** `figures/A1_training_curves.png` (Pearson on C_ij against gradient step for
  both alignments, with the 0.95 that was reported for the old model marked; and a zoom
  showing the two differ by 0.005)
- **Bullets:** The new alignment trains to 0.826 · The old model was reported at 0.95 ·
  Test: retrain the old alignment, same protocol
- **Banner:** Both alignments stop in the **same place** — 0.826 and 0.821

### A2 — It is not the data

- **Figure:** `figures/A2_slope_and_data.png` (regression slope of C_ij reaching 0.95 on
  both; and M / M_eff for the two alignments)
- **Bullets:** The slope does reach 0.95 on both · The old alignment carries 37 % less
  effective data · and still lands in the same place
- **Banner:** The 0.95 was a **different statistic**, not better data

### A3 — Reporting the numbering

No figure. Short table:

| MSA col | label | WT | kind |
|---|---|---|---|
| 197 | `188` | G | literature |
| **198** | **`188a`** | **G** | **alignment-specific, 34.5 % gapped** |
| 199 | `188b` | – | alignment-specific |
| 200 | `188A` | K | literature insertion code |
| 201 | `189` | D | literature — **the design target** |

- **Banner:** Call it **an inserted position** in the 186–190 loop, not a residue number

---

## 6. Figures in the zip

### `figures/` — used by the deck

| file | slide | what it shows |
|---|---|---|
| `01_goal.png` | 1 | the D → S switch, as two boxes and an arrow |
| `04_ensemble_trace.png` | 4 | four ensemble observables along the run, λ = 0 / 100 / 1000 |
| `05_mu_calibration.png` | 5 | how μ = 331 is calibrated on the natural alignment |
| `06_candidates_all_lambdas.png` | 6 | 5 × 5 corner plot of the final designs, every λ, vs natural |
| `08_candidates_lam300_grafting.png` | 8, 10 | the same corner plot, λ = 300, grafting run |
| `09_critic_substitution_deltas.png` | 9 | ΔE and ΔQ of writing S in |
| `10_candidates_lam300_free.png` | 10 | λ = 300 with all sites free — the control |
| `11_sector_mutations.png` | 11 | where the mutations land, and sector enrichment |
| `12_gap_abundance.png` | 12 | gaps per sequence, opened vs closed, per column |
| `A1_training_curves.png` | A1 | training curves for both alignments |
| `A2_slope_and_data.png` | A2 | slope reaching 0.95, and M / M_eff |

### `extras/` — available if you want them

`grafting_trajectories.png` (single chains towards the threshold),
`grafting_sector_trajectories.png` (which sector each accepted move hit),
`grafting_ensemble_pairs.png` (the 6 × 6 version of the corner plot, including gap count),
and the free-run counterparts `free_ensemble_trace.png`,
`free_candidates_all_lambdas.png`, `free_sector_mutations.png`, `free_gap_abundance.png`.

---

## 7. Things to get right

- **μ = 331**, not 317. An earlier version of this deck used 317; it was recalibrated on
  the retrained model. Every figure in the zip says 331.
- **P(S) in the wild type is 0.005.** The natural family averages 0.104.
- The operating point is **λ = 300**, not λ = 1000. λ = 1000 reaches higher P(S) but costs
  40 mutations and gives up the Q advantage over the natural family.
- The free run is a **control, not a set of designs**. Never present its sequences as
  candidates.
- β appears in the sampler but every run shown here is at **β = 1**. Do not make a
  temperature slide; cooling was explored earlier and is not part of this story.
- `188a` is an inserted column with no literature counterpart — describe it, do not cite it
  as a residue number.
