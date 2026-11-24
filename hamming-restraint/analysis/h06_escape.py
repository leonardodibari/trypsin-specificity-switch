"""H6 -- test 5: if the chains are PAID to leave, where do they go?

THE QUESTION. The project's headline negative result is that the designs "stay
inside the trypsins": they sit at d_H = 6..28 from the wild type and close at
most ~5 % of the gap to chymotrypsin, while natural chymotrypsins are 82..211
substitutions away. Every explanation offered so far is about cost -- the bias
cannot pay for the residues it would take to get out.

A negative t attacks that explanation directly, because

    H = beta*E_DCA - lambda*log P(S | sigma_\\i*) + t*d_H(sigma, WT)

makes t the chemical potential conjugate to the Hamming distance, and t < 0 PAYS
the chains to travel. So the chains CAN be made to leave; the only question left
is what they are when they get there. This figure asks it in three parts, all of
them at mean distances comparable to the naturals:

  (a) DIRECTION   do they move along the trypsin -> chymotrypsin axis in
                  sequence-PCA space, or orthogonally into nowhere?
  (b) THE FOUR REFUSALS  CLAMP_CONTROL section 2.3 names four things the model
                  will not do -- columns 241 (3TGI 226), 229 (216), 177 (172)
                  and the loop 194..200 (185-188). Do they finally happen?
  (c) IDENTITY    does the nearest natural neighbour switch family, and does the
                  held-out critic-site discriminator call these chymotrypsins?

THE CONTROL, which is the point of the whole figure. a3_free_lam0 is the
IDENTICAL t scan at lambda = 0. The restraint on its own drags chains toward the
family average, and the family carries S at column 201 in 10.4 % of its members
(9,259 of 89,439), so a negative t raises P(S) with no design pressure at all.
If lambda = 100 reaches chymotrypsin markers and lambda = 0 only reaches the
family average at the same <d>, the effect is real. If both land in the same
place, the bias has done nothing and the project has merely been diluting toward
the family mean. Every panel therefore carries both arms, and an arm that has
not been sampled yet is reported as missing rather than quietly omitted.

THE NOVELTY GUARD, which is non-negotiable the moment t < 0 lets chains travel
100+ substitutions. A design 100 mutations from the wild type that happens to be
a copy of some natural chymotrypsin is RETRIEVAL, not design. Panel (i) is the
minimum Hamming distance from each ensemble to the 89,439-sequence alignment.
Both arms here are all-columns-free, so unlike the per-block numbers that
scripts/add_natural_neighbour.py writes (scored over each block's own
--allowed-cols, and therefore not comparable between a 79-column block and this
one) these two ARE comparable with each other, and are scored over all 265
columns.

THE PANELS
  (a)   the natural families in the PCA of the alignment, with both arms' designs
        projected into that basis and shaded by t.
  (b)   the same thing as two numbers: the fraction of the WT -> chymotrypsin gap
        closed ALONG the axis, and the displacement ORTHOGONAL to it, against the
        mean distance travelled. Travel without direction shows up here as the
        dashed line rising while the solid one stays flat.
  (c-f) the four refusals, column by column, against t.
  (g)   the family of the nearest natural neighbour: trypsin or chymotrypsin.
  (h)   the share of designs the held-out discriminator calls chymotrypsin.
  (i)   the novelty guard.

WHAT IS READ. The UNSUBSTITUTED block, like h03: substitute_critic.py stamps S
at column 201, which shifts d_H by one wherever a chain did not already carry S,
and d_H is the coordinate the restraint acted on and the one this figure plots
against. It would also turn panel (h) into a tautology at the critic site.

WHAT IS CACHED. The PCA projection of 89,439 sequences and the nearest-neighbour
sweep of every chain against the whole alignment are both written to
``results/h06_cache/`` keyed by the model fingerprint, so the figure can be
redrawn without the GPU and without the alignment.
"""

from __future__ import annotations

import functools
import importlib
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import config
import data
import style

#: The biased arm and its matched control -- the same two blocks, the same
#: colours and the same markers as h03. A reader moving between the two figures
#: must not have to re-learn which line is which.
BIASED = "a2_free_lam100"
CONTROL = "a3_free_lam0"

ARMS = {
    BIASED: (r"$\lambda=100$  (biased)", style.BLUE, "o"),
    CONTROL: (r"$\lambda=0$  (matched control)", style.ORANGE, "s"),
}

#: The four things CLAMP_CONTROL section 2.3 says the model will not do, as
#: (name, 3TGI label, MSA columns). The first three are single columns; the loop
#: is seven, because the 185-188 swap changes length and the insertion-code
#: columns go with it. Taken from the ladder's config so there is one definition.
#:
#: These columns were never used to build anything -- no site set, no bias, no
#: selection rule reads them -- which is what makes them a prediction rather
#: than a readout.
def _refusal_sites() -> List[Tuple[str, str, List[int]]]:
    ladder_config = _ladder()["config"]
    return [
        ("226", "the S1 pocket floor", [ladder_config.POS_226]),
        ("216", "the pocket wall", [ladder_config.POS_216]),
        ("172", "the pocket rim", [ladder_config.POS_172]),
        ("185-188", "the surface loop", list(ladder_config.LOOP_1)),
    ]


#: Where the heavy intermediates live. Under RESULTS_ROOT, not under the ladder's
#: .cache, because these are this study's numbers.
CACHE_DIR = config.RESULTS_ROOT / "h06_cache"

#: Below this many substitutions from its nearest natural, a design is not a
#: design: it is a sequence the alignment already contains, up to a handful of
#: positions. The band is drawn, not enforced -- the point is that a reader sees
#: it without having to do arithmetic.
RETRIEVAL_FLOOR = 10


# --------------------------------------------------------------------------
# the ladder's modules, loaded without letting them capture the name "config"
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _ladder() -> Dict[str, object]:
    """The ladder's ``config``, ``data``, ``metrics`` and ``fig07_pca``.

    Loaded BY HAND rather than with a plain import, for the reason config.py
    already documents for ``site_sets``: the ladder's modules are *also* called
    ``config`` and ``data``, and this package's own ``config``/``data`` are
    already in ``sys.modules``. A bare ``import data`` from here returns the
    wrong one; worse, leaving the ladder's ``config`` installed under that name
    would make ``style.save`` -- which resolves OUT_DIR through whatever
    ``config`` it finds -- redirect this figure into ``figures/ladder/``.

    So the ladder's names are imported with its directory first on sys.path and
    ours evicted, and then ours are put straight back. The ladder modules keep
    working afterwards because each one bound its ``config`` as a module OBJECT
    at import time; only the sys.modules entry is taken away.
    """
    names = ("config", "data", "metrics", "selection", "fig07_pca")
    saved = {name: sys.modules.pop(name, None) for name in names}
    sys.path.insert(0, str(config.LADDER_ANALYSIS))
    try:
        modules = {name: importlib.import_module(name) for name in names}
    finally:
        sys.path.remove(str(config.LADDER_ANALYSIS))
        for name in names:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]
    return modules


def _cache_path(stem: str) -> Path:
    """A cache file keyed by the model fingerprint, so a new model misses cleanly.

    The same contract as ``figures_ladder/config.cache_path``: a cached PCA basis
    or neighbour sweep belongs to the parameter file it was computed from, and
    silently reusing one across models is the failure mode that produces
    plausible, wrong figures.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint = _ladder()["config"].model_fingerprint()
    return CACHE_DIR / f"{stem}.{fingerprint}.npz"


# --------------------------------------------------------------------------
# the reference frame: what the answer would look like
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def reference() -> Dict[str, np.ndarray]:
    """Everything the designs are measured against, computed once and cached.

    * ``natural_pc`` / ``natural_class``  the alignment in the PCA plane, with
      0 = other, 1 = trypsin, 2 = chymotrypsin at the critic site.
    * ``wt_pc``, ``chymo_pc``, ``gap``    the WT -> chymotrypsin-centroid vector
      that defines "the right direction" in panel (b).
    * ``target_state`` / ``chymo_freq`` / ``tryp_freq``  per refusal column, the
      residue to look for and the two natural families' frequencies of it.
    * ``threshold`` / ``auc``             the held-out discriminator of panel (h).
    """
    cache = _cache_path("reference")
    if cache.is_file():
        return {k: v for k, v in np.load(cache, allow_pickle=True).items()}

    ladder = _ladder()
    ldata, lconfig, lmetrics, lfig07 = (ladder["data"], ladder["config"],
                                        ladder["metrics"], ladder["fig07_pca"])
    _, encoded = ldata.alignment()
    classes = ldata.natural_classes()
    mean, components, explained, natural_pc, total, q = lfig07._pca_basis()

    wt = ldata.wild_type()
    wt_pc = lfig07._project(wt, mean, components, q)[0]
    chymo_rows = encoded[classes["chymotrypsin"]]
    tryp_rows = encoded[classes["trypsin"]]
    chymo_pc = natural_pc[classes["chymotrypsin"]].mean(axis=0)

    # The axis of panel (b). Measured from the WILD TYPE, not from the trypsin
    # centroid: the chains start at the wild type, so "fraction of the gap
    # closed" is only a fraction of anything if the gap is the one they face.
    gap = chymo_pc - wt_pc

    # Per refusal column, the residue to count. NOT the modal chymotrypsin
    # residue -- at 172 and 216 that is the same residue the trypsins carry
    # (Y at 94 %, G at 97 %), so counting it would score a refusal as a success.
    # The chymotrypsin residue that MATTERS is the one chymotrypsins have and
    # trypsins do not, i.e. the maximum of f_chym(r) - f_tryp(r). That choice
    # recovers exactly the residues CLAMP_CONTROL names: S at 226 (28 % against
    # 2.8 %) and W at 172 (16 % against 1.5 %).
    columns = [c for _, _, cols in _refusal_sites() for c in cols]
    target_state, chymo_freq, tryp_freq = [], [], []
    for column in columns:
        f_chymo = np.bincount(chymo_rows[:, column], minlength=q) / len(chymo_rows)
        f_tryp = np.bincount(tryp_rows[:, column], minlength=q) / len(tryp_rows)
        state = int(np.argmax(f_chymo - f_tryp))
        target_state.append(state)
        chymo_freq.append(float(f_chymo[state]))
        tryp_freq.append(float(f_tryp[state]))

    # The distance band the naturals actually occupy, measured rather than
    # quoted, so the figure cannot drift away from the alignment it is drawn on.
    distance = (encoded != wt[None, :]).sum(axis=1)
    chymo_distance = distance[classes["chymotrypsin"]]

    # The held-out discriminator of F1, with an operating point. metrics.roc
    # returns (fpr, tpr, auc) and no thresholds, so the sweep is redone here to
    # recover the score at the Youden point; the AUC itself still comes from
    # metrics.auc, which is the rank form and handles the enormous tie pile at
    # P(S) ~ 0 exactly.
    P, _ = ldata.natural_conditional()
    log_odds = ldata.log_odds(P)
    held_out = ldata.split_mask()
    evaluated = held_out if held_out is not None else np.ones(len(P), dtype=bool)
    positive = classes["chymotrypsin"] & evaluated
    negative = classes["trypsin"] & evaluated
    values = np.concatenate([log_odds[positive], log_odds[negative]])
    labels = np.concatenate([np.ones(int(positive.sum())), np.zeros(int(negative.sum()))])
    order = np.argsort(-values, kind="mergesort")
    values, labels = values[order], labels[order]
    tpr = np.cumsum(labels) / labels.sum()
    fpr = np.cumsum(1 - labels) / (len(labels) - labels.sum())
    best = int(np.argmax(tpr - fpr))

    blob = dict(
        natural_pc=natural_pc.astype(np.float32),
        natural_class=np.where(classes["chymotrypsin"], 2,
                               np.where(classes["trypsin"], 1, 0)).astype(np.int8),
        explained=np.asarray(explained, dtype=float),
        total_variance=np.asarray(total, dtype=float),
        wt_pc=wt_pc.astype(np.float32),
        chymo_pc=chymo_pc.astype(np.float32),
        gap=gap.astype(np.float32),
        refusal_columns=np.asarray(columns, dtype=int),
        target_state=np.asarray(target_state, dtype=int),
        target_token=np.asarray([ldata.alphabet().tokens[s] for s in target_state],
                                dtype=object),
        chymo_freq=np.asarray(chymo_freq, dtype=float),
        tryp_freq=np.asarray(tryp_freq, dtype=float),
        chymo_distance_range=np.asarray(
            [chymo_distance.min(), chymo_distance.max()], dtype=float),
        threshold=np.asarray(values[best], dtype=float),
        auc=np.asarray(lmetrics.auc(log_odds, positive, negative), dtype=float),
        tpr_at_threshold=np.asarray(tpr[best], dtype=float),
        fpr_at_threshold=np.asarray(fpr[best], dtype=float),
        held_out=np.asarray(int(evaluated.sum()), dtype=int),
        n_chymo=np.asarray(int(classes["chymotrypsin"].sum()), dtype=int),
        n_natural=np.asarray(len(encoded), dtype=int),
    )
    np.savez_compressed(cache, **blob)
    print(f"  wrote {cache}")
    return blob


# --------------------------------------------------------------------------
# per-run heavy work
# --------------------------------------------------------------------------

def run_profile(block: str, run: str) -> Dict[str, np.ndarray]:
    """PCA coordinates, nearest-natural distance and refusal counts for one run.

    This is the expensive half of the figure -- one 1000 x 89,439 Hamming sweep
    per run, on ``--device mps`` through the ladder's config -- so it is cached
    per run under RESULTS_ROOT and never recomputed for a redraw.

    The nearest neighbour is scored over ALL 265 columns. Both arms are
    all-columns-free, so there is nothing to restrict and the two are directly
    comparable; that is the one case the caveat in add_natural_neighbour.py
    allows.
    """
    cache = _cache_path(f"run.{block}.{run}")
    if cache.is_file():
        return {k: v for k, v in np.load(cache, allow_pickle=True).items()}

    ladder = _ladder()
    ldata, lfig07 = ladder["data"], ladder["fig07_pca"]
    directory = config.RESULTS_ROOT / block / run
    sequences = ldata.run_sequences(str(directory))

    mean, components, _, _, _, q = lfig07._pca_basis()
    points = lfig07._project(sequences, mean, components, q)

    # (distance, index) to the closest member of the whole alignment. The index
    # is what turns the novelty guard into the identity panel for free: the
    # family of the nearest natural is read straight off it.
    distance, index = ldata.nearest_natural(sequences)
    classes = ldata.natural_classes()
    neighbour_class = np.where(classes["chymotrypsin"][index], 2,
                               np.where(classes["trypsin"][index], 1, 0))

    ref = reference()
    counts = [
        (sequences[:, column] == state).astype(np.float32)
        for column, state in zip(ref["refusal_columns"], ref["target_state"])
    ]

    blob = dict(
        pc=points.astype(np.float32),
        nn_distance=np.asarray(distance, dtype=np.int32),
        nn_class=neighbour_class.astype(np.int8),
        refusal_hits=np.stack(counts).astype(np.float32),   # (columns, chains)
        n_chains=np.asarray(len(sequences), dtype=int),
    )
    np.savez_compressed(cache, **blob)
    return blob


# --------------------------------------------------------------------------
# loading and summarising an arm
# --------------------------------------------------------------------------

def arm(block: str) -> Optional[pd.DataFrame]:
    """Per-chain observables of one arm, saturated runs only, or None.

    Read through the UNSUBSTITUTED block, for the reason h03 gives: the critic
    substitution changes the residue at column 201 and therefore shifts d_H by
    one for most chains, and d_H is both the coordinate the restraint acted on
    and the x axis of panel (b).

    ``data.designs`` drops every run that failed the saturation drift test, so a
    rung that has not relaxed cannot reach this figure at all. Returning None
    rather than raising is what lets the control arm be missing: it is being
    re-sampled, and the figure must still render without it.
    """
    try:
        frame = data.designs(block, serine=False)
    except FileNotFoundError:
        return None
    if frame.empty:
        return None
    frame = frame.copy()
    frame["block"] = block
    return frame


def summarise_arm(block: str, frame: pd.DataFrame) -> pd.DataFrame:
    """One row per t: everything the figure draws, and everything the CSV holds."""
    ref = reference()
    gap = ref["gap"].astype(float)
    gap_length = float(np.hypot(*gap))
    unit = gap / gap_length
    sites = _refusal_sites()
    column_index = {int(c): i for i, c in enumerate(ref["refusal_columns"])}

    rows = []
    for (t, run), group in frame.groupby(["t", "run"], sort=True):
        profile = run_profile(block, run)
        # The per-chain rows of observables.csv and the rows of
        # final_sequences.fasta are the same chains in the same order -- the
        # sampler writes them together -- but an inconsistency here would
        # silently pair one chain's distance with another's sequence.
        if len(group) != int(profile["n_chains"]):
            raise SystemExit(
                f"{block}/{run}: {len(group)} rows in observables.csv against "
                f"{int(profile['n_chains'])} sequences in final_sequences.fasta")

        displacement = profile["pc"].astype(float) - ref["wt_pc"].astype(float)[None, :]
        along = displacement @ unit / gap_length
        orthogonal = np.linalg.norm(
            displacement - np.outer(displacement @ unit, unit), axis=1) / gap_length

        distance = profile["nn_distance"].astype(float)
        neighbour = profile["nn_class"]
        row = dict(
            block=block,
            run=run,
            t=float(t),
            n_chains=len(group),
            # --- how far, and how much of it is indel rather than substitution
            mean_d=float(group["hamming_to_wt"].mean()),
            median_d=float(group["hamming_to_wt"].median()),
            d_q25=float(group["hamming_to_wt"].quantile(0.25)),
            d_q75=float(group["hamming_to_wt"].quantile(0.75)),
            mean_gaps=float(group["n_gaps"].mean()),
            # --- (a), (b): direction
            along_frac=float(along.mean()),
            along_frac_sd=float(along.std(ddof=1)),
            orth_frac=float(orthogonal.mean()),
            # --- (g), (h): identity
            frac_nn_chymotrypsin=float((neighbour == 2).mean()),
            frac_nn_trypsin=float((neighbour == 1).mean()),
            frac_called_chymotrypsin=float(
                (group["LOR"] >= float(ref["threshold"])).mean()),
            mean_LOR=float(group["LOR"].mean()),
            mean_P_S=float(group["P_S"].mean()),
            mean_E_DCA=float(group["E_DCA"].mean()),
            # --- (i): the novelty guard
            nn_min=float(distance.min()),
            nn_q25=float(np.quantile(distance, 0.25)),
            nn_median=float(np.median(distance)),
            nn_q75=float(np.quantile(distance, 0.75)),
            frac_nn_under_floor=float((distance < RETRIEVAL_FLOOR).mean()),
        )
        # --- (c)-(f): the four refusals, each the mean over its columns of the
        # frequency of that column's chymotrypsin-specific residue.
        for name, _, columns in sites:
            hits = np.stack([profile["refusal_hits"][column_index[c]] for c in columns])
            row[f"refusal_{name}"] = float(hits.mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("t").reset_index(drop=True)


def refusal_reference(name: str) -> Tuple[float, float]:
    """``(chymotrypsin, trypsin)`` frequency of the residue a refusal is about.

    The ceiling and the floor of panels (c)-(f): what the answer would look like,
    and what the wild type's own family looks like. Averaged over the columns of
    a multi-column site exactly as the designs are.
    """
    ref = reference()
    index = {int(c): i for i, c in enumerate(ref["refusal_columns"])}
    columns = next(cols for n, _, cols in _refusal_sites() if n == name)
    rows = [index[c] for c in columns]
    return (float(ref["chymo_freq"][rows].mean()),
            float(ref["tryp_freq"][rows].mean()))


# --------------------------------------------------------------------------
# the figure
# --------------------------------------------------------------------------

def main() -> None:
    style.apply()
    config.ensure_dirs()
    ref = reference()

    arms: Dict[str, pd.DataFrame] = {}
    tables: Dict[str, pd.DataFrame] = {}
    for block in ARMS:
        frame = arm(block)
        if frame is None:
            print(f"  WARNING: {block} has no saturated run on disk -- it is the "
                  f"{'biased arm' if block == BIASED else 'MATCHED CONTROL'} and "
                  "the figure will be drawn without it.")
            continue
        arms[block] = frame
        tables[block] = summarise_arm(block, frame)

    if not tables:
        raise SystemExit("Neither arm has a saturated run; nothing to draw.")
    if CONTROL not in tables:
        print("\n  !! THE CONTROL ARM IS MISSING. Every claim this figure can "
              "support is therefore conditional: the restraint alone drags chains\n"
              "     toward the family average, and without a3_free_lam0 there is "
              "no way to tell that apart from the bias doing something.\n")

    # Whether the question can be answered at all with what has saturated. The
    # whole point of this figure is t < 0; if no negative rung survived the drift
    # test then nothing here is evidence about escape, only about the restraint
    # pulling chains IN, and the figure must say so on its face.
    negative = {block: sorted(table.loc[table["t"] < 0, "t"]) for block, table in tables.items()}
    escape_sampled = any(negative.values())
    for block, values in negative.items():
        print(f"  {block}: saturated rungs at t = "
              f"{', '.join(f'{t:g}' for t in tables[block]['t'])}"
              + (f"   (negative: {', '.join(f'{t:g}' for t in values)})" if values
                 else "   (NO negative t -- the escape question is untested)"))

    # The caveats are assembled BEFORE the grid, because two of them exist only
    # when something is missing and they change how much room the figure has to
    # leave at the bottom. A footnote that lands on top of an axis label is a
    # footnote nobody reads.
    missing = [block for block in ARMS if block not in tables]
    notes = [
        "Both arms are all 265 columns free with gap moves on, so $d_H$ counts indels; "
        "read from the UNSUBSTITUTED block, because the critic substitution shifts $d_H$. "
        "Only runs that pass the saturation drift test are shown.",
        "(b) measures the gap from the WILD TYPE to the natural-chymotrypsin centroid in "
        "the PC1–PC2 plane, so 1.0 is the whole gap and the dashed line is travel that "
        "closes none of it.  (c)–(f) count the residue most enriched in chymotrypsins "
        "over trypsins at each column, not the modal one — at 172 and 216 the modal "
        "chymotrypsin residue is the trypsin residue.",
        "(h) is NOT independent evidence for the biased arm: $\\lambda$ steers on exactly "
        "this log-odds, so a high call rate there is the bias working, not the design "
        "being a chymotrypsin. It is the $\\lambda=0$ arm, which steers on nothing, that "
        "makes the panel informative.  (i) is only a guard where the chains travel: a "
        "design that has barely left the wild type is close to the alignment because the "
        "wild type is IN the alignment, so the panel earns its name at $t<0$ and not "
        "at $t>0$.",
    ]
    if missing:
        notes.append(
            "MISSING ARM: " + ", ".join(missing) + " is not on disk, so the control "
            "that separates the bias from the restraint's own pull toward the family "
            "average is NOT in this figure. Nothing here may be attributed to the bias "
            "until it is.")
    if not escape_sampled:
        notes.append(
            "NO SATURATED RUN AT $t<0$: every rung drawn here PENALISES distance. This "
            "figure is currently a description of the island, not a test of escape "
            "from it.")
    notes = [line for note in notes for line in textwrap.wrap(note, width=255)]

    fig = plt.figure(figsize=(19.5, 16.8))
    bottom = 0.045 + 0.0145 * len(notes)
    grid = fig.add_gridspec(3, 12, left=0.055, right=0.985, top=0.905, bottom=bottom,
                            hspace=0.44, wspace=1.5)

    chymo_lo, chymo_hi = ref["chymo_distance_range"]

    # ----------------------------------------------------------------- (a) PCA
    ax = fig.add_subplot(grid[0, 0:6])
    natural_pc, natural_class = ref["natural_pc"], ref["natural_class"]
    # The natural families keep their semantic colours (C_TRYPSIN, C_CHYMO) but
    # are labelled IN PLACE rather than in the legend. The arm palette this
    # study inherited from h03 is blue/orange, which is the same blue and the
    # same orange as the two families -- unavoidable, since both are the
    # project's fixed roles -- so a legend naming four things in two colours
    # would be worse than useless here. Naming the clouds where they lie leaves
    # the legend to carry only the arms and the two fixed points.
    # The anchor is NOT each cloud's centroid: the two families overlap heavily
    # in this projection and their centroids are a few tenths of a unit apart,
    # so centroid-anchored labels land on top of each other. Each label is
    # pinned instead to the side of its own cloud that the other family does
    # not reach -- trypsins to the left, chymotrypsins to the right.
    for code, colour, label, anchor, align in (
            (0, style.C_OTHER, None, None, None),
            (1, style.C_TRYPSIN, "natural trypsins", (8, 50), "left"),
            (2, style.C_CHYMO, "natural chymotrypsins", (94, 70), "right")):
        points = natural_pc[natural_class == code]
        ax.scatter(points[:, 0], points[:, 1], s=2.0, c=colour, alpha=0.15,
                   linewidths=0, rasterized=True, zorder=1)
        if label is not None:
            x = np.percentile(points[:, 0], anchor[0])
            y = np.percentile(points[:, 1], anchor[1])
            ax.text(x, y, label, color=colour, fontsize=12.5, fontweight="bold",
                    ha=align, va="center", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                              alpha=0.82))

    # The designs, shaded light-to-dark within each arm's own hue as t falls, so
    # the direction of travel is readable without a second colour axis, and
    # ringed in white so that they read as foreground over the natural cloud
    # they share a hue with.
    handles = []
    for block, table in tables.items():
        label, colour, marker = ARMS[block]
        ts = list(table["t"])
        shades = dict(zip(sorted(ts, reverse=True), style.ramp(colour, len(ts))))
        for row in table.itertuples():
            points = run_profile(block, row.run)["pc"]
            ax.scatter(points[:, 0], points[:, 1], s=16, marker=marker,
                       facecolor=shades[row.t], edgecolor="white", linewidths=0.25,
                       alpha=0.80, rasterized=True, zorder=4)
        handles.append(Line2D([], [], marker=marker, ls="none", color=colour,
                              markeredgecolor="white", markersize=10,
                              label=f"designs, {label}"))
    ax.annotate("", xy=tuple(ref["chymo_pc"]), xytext=tuple(ref["wt_pc"]),
                arrowprops=dict(arrowstyle="-|>", color=style.INK, lw=2.0,
                                shrinkA=0, shrinkB=0, ls=(0, (4, 2))), zorder=7)
    ax.scatter(*ref["wt_pc"], s=260, marker="*", color=style.C_WT, zorder=8)
    ax.scatter(*ref["chymo_pc"], s=150, marker="P", color=style.C_CHYMO_REF, zorder=8)
    handles += [
        Line2D([], [], marker="*", ls="none", color=style.C_WT, markersize=15,
               label="wild-type trypsin"),
        Line2D([], [], marker="P", ls="none", color=style.C_CHYMO_REF,
               markersize=10, label="chymotrypsin centroid (the arrow is the gap)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=11, handletextpad=0.5,
              labelspacing=0.35, facecolor="white", framealpha=0.92, frameon=True,
              edgecolor=style.FAINT)
    share = 100 * ref["explained"] / float(ref["total_variance"])
    ax.set_xlabel(f"PC1  ({share[0]:.1f}% of total variance)")
    ax.set_ylabel(f"PC2  ({share[1]:.1f}%)")
    ax.set_title("where the designs sit, and where they would have to go",
                 fontsize=14, color=style.MUTED, pad=8)
    style.panel_letter(ax, "a", dx=-0.075)
    style.tidy(ax)

    # ------------------------------------------------- (b) along vs orthogonal
    ax = fig.add_subplot(grid[0, 6:12])
    ax.axvspan(chymo_lo, chymo_hi, color=style.C_OTHER, alpha=0.30, lw=0, zorder=0)
    ax.text(0.5 * (chymo_lo + chymo_hi), 0.905,
            f"how far natural chymotrypsins are from WT\n({chymo_lo:.0f}–{chymo_hi:.0f} substitutions)",
            transform=ax.get_xaxis_transform(), color=style.MUTED, fontsize=11.5,
            va="top", ha="center", linespacing=1.4)
    ax.axhline(1.0, color=style.C_CHYMO_REF, lw=1.6, ls="--", zorder=1)
    ax.text(0.99, 1.005, "chymotrypsin centroid ", transform=ax.get_yaxis_transform(),
            color=style.C_CHYMO_REF, fontsize=11.5, va="bottom", ha="right")
    ax.axhline(0.0, color=style.RULE, lw=1.2, zorder=1)
    for block, table in tables.items():
        label, colour, marker = ARMS[block]
        ax.plot(table["mean_d"], table["along_frac"], color=colour, lw=1.9,
                marker=marker, markersize=7, markeredgecolor="white",
                markeredgewidth=0.8, zorder=4, label=f"{label} — along the axis")
        ax.plot(table["mean_d"], table["orth_frac"], color=colour, lw=1.5,
                ls=(0, (2, 2)), marker=marker, markersize=5, markerfacecolor="white",
                zorder=3, label=f"{label} — orthogonal")
        # Label the ends with the t that produced them: the x axis is distance,
        # so without this the direction of the scan is invisible.
        for row in (table.iloc[0], table.iloc[-1]):
            ax.annotate(f"$t$={row.t:g}", xy=(row.mean_d, row.along_frac),
                        xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=10.5, color=colour)
    ax.set_xlabel(r"$\langle d_H(\sigma,\,\mathrm{WT})\rangle$   (mean distance travelled)")
    ax.set_ylabel("fraction of the WT $\\rightarrow$ chymotrypsin gap")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), fontsize=11)
    ax.set_title("direction of travel: along the axis, or off it?",
                 fontsize=14, color=style.MUTED, pad=8)
    style.panel_letter(ax, "b", dx=-0.062, dy=1.085)
    style.tidy(ax)

    # --------------------------------------------------- (c)-(f) four refusals
    for index, ((name, what, columns), letter) in enumerate(
            zip(_refusal_sites(), "cdef")):
        ax = fig.add_subplot(grid[1, 3 * index:3 * index + 3])
        chymo_level, tryp_level = refusal_reference(name)
        ax.axhline(chymo_level, color=style.C_CHYMO, lw=1.8, ls="--", zorder=2)
        ax.axhline(tryp_level, color=style.C_TRYPSIN, lw=1.5, ls=":", zorder=2)
        for block, table in tables.items():
            _, colour, marker = ARMS[block]
            ax.plot(table["t"], table[f"refusal_{name}"], color=colour, lw=1.9,
                    marker=marker, markersize=6.5, markeredgecolor="white",
                    markeredgewidth=0.8, zorder=4)
        ax.axvline(0.0, color=style.RULE, lw=1.1, zorder=1)
        residue = "".join(
            str(ref["target_token"][{int(c): i for i, c in
                                     enumerate(ref["refusal_columns"])}[c]])
            for c in columns)
        ax.set_ylim(-0.03, max(0.45, 1.25 * chymo_level))
        ax.set_xlabel("restraint strength $t$")
        if index == 0:
            ax.set_ylabel("frequency of the\nchymotrypsin residue")
        title = (f"3TGI {name} — {what}\ncol "
                 + (f"{columns[0]}, residue {residue}" if len(columns) == 1
                    else f"{columns[0]}–{columns[-1]}, mean over {len(columns)}"))
        ax.set_title(title, fontsize=12.5, color=style.MUTED, pad=8)
        style.panel_letter(ax, letter, dx=-0.30 if index == 0 else -0.13)
        style.tidy(ax)
        if index == 0:
            ax.legend(handles=[
                Line2D([], [], color=style.C_CHYMO, ls="--", lw=1.8,
                       label="natural chymotrypsins"),
                Line2D([], [], color=style.C_TRYPSIN, ls=":", lw=1.5,
                       label="natural trypsins")], loc="upper right", fontsize=10.5)

    # --------------------------------------------- (g) the nearest natural's family
    ax = fig.add_subplot(grid[2, 0:4])
    for block, table in tables.items():
        label, colour, marker = ARMS[block]
        ax.plot(table["t"], table["frac_nn_chymotrypsin"], color=colour, lw=1.9,
                marker=marker, markersize=6.5, markeredgecolor="white",
                markeredgewidth=0.8, zorder=4)
        ax.plot(table["t"], table["frac_nn_trypsin"], color=colour, lw=1.4,
                ls=(0, (2, 2)), zorder=3)
    ax.axvline(0.0, color=style.RULE, lw=1.1, zorder=1)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("restraint strength $t$")
    ax.set_ylabel("share of chains")
    ax.set_title("family of the nearest natural neighbour", fontsize=13,
                 color=style.MUTED, pad=8)
    # Colour already carries the arm -- it does so in every other panel -- so
    # this legend carries only the second encoding, which is the line style.
    # Naming both at once would be four entries in two hues over four points.
    ax.legend(handles=[
        Line2D([], [], color=style.MUTED, lw=1.9, label="nearest is a chymotrypsin"),
        Line2D([], [], color=style.MUTED, lw=1.4, ls=(0, (2, 2)),
               label="nearest is a trypsin")], loc="center left", fontsize=10.5)
    style.panel_letter(ax, "g", dx=-0.22)
    style.tidy(ax)

    # ------------------------------------------------------- (h) the discriminator
    ax = fig.add_subplot(grid[2, 4:8])
    ax.axhline(float(ref["tpr_at_threshold"]), color=style.C_CHYMO, lw=1.8,
               ls="--", zorder=2)
    ax.axhline(float(ref["fpr_at_threshold"]), color=style.C_TRYPSIN, lw=1.5,
               ls=":", zorder=2)
    for block, table in tables.items():
        label, colour, marker = ARMS[block]
        ax.plot(table["t"], table["frac_called_chymotrypsin"], color=colour,
                lw=1.9, marker=marker, markersize=6.5, markeredgecolor="white",
                markeredgewidth=0.8, zorder=4)
    ax.axvline(0.0, color=style.RULE, lw=1.1, zorder=1)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("restraint strength $t$")
    ax.set_ylabel("share called chymotrypsin")
    ax.set_title(f"held-out discriminator  (AUC {float(ref['auc']):.3f})",
                 fontsize=13, color=style.MUTED, pad=8)
    ax.legend(handles=[
        Line2D([], [], color=style.C_CHYMO, ls="--", lw=1.8,
               label=f"natural chymotrypsins ({float(ref['tpr_at_threshold']):.2f})"),
        Line2D([], [], color=style.C_TRYPSIN, ls=":", lw=1.5,
               label=f"natural trypsins ({float(ref['fpr_at_threshold']):.2f})")],
        loc="center left", fontsize=10.5)
    style.panel_letter(ax, "h", dx=-0.19)
    style.tidy(ax)

    # ------------------------------------------------------ (i) the novelty guard
    ax = fig.add_subplot(grid[2, 8:12])
    ax.axhspan(0, RETRIEVAL_FLOOR, color=style.RED, alpha=0.13, lw=0, zorder=0)
    ax.text(0.02, RETRIEVAL_FLOOR, " retrieval, not design",
            transform=ax.get_yaxis_transform(), color=style.RED, fontsize=11,
            va="bottom", ha="left")
    for block, table in tables.items():
        label, colour, marker = ARMS[block]
        ax.fill_between(table["t"], table["nn_q25"], table["nn_q75"],
                        color=colour, alpha=0.16, lw=0, zorder=2)
        ax.plot(table["t"], table["nn_median"], color=colour, lw=1.9,
                marker=marker, markersize=6.5, markeredgecolor="white",
                markeredgewidth=0.8, zorder=4)
        # The guard is about the CLOSEST design, not the typical one: one
        # retrieved sequence in a library is one retrieved sequence.
        ax.plot(table["t"], table["nn_min"], color=colour, lw=1.2,
                ls=(0, (1, 2)), marker="v", markersize=5, zorder=3)
    ax.axvline(0.0, color=style.RULE, lw=1.1, zorder=1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("restraint strength $t$")
    ax.set_ylabel(r"$d_H$ to nearest natural" "\n(all 265 columns)")
    ax.set_title(f"novelty guard: {int(ref['n_natural']):,}-sequence alignment",
                 fontsize=13, color=style.MUTED, pad=8)
    ax.legend(handles=[
        Line2D([], [], color=style.MUTED, lw=1.9, label="ensemble median"),
        Line2D([], [], color=style.MUTED, lw=1.2, ls=(0, (1, 2)), marker="v",
               markersize=5, label="ensemble minimum")],
        loc="upper right", fontsize=10.5)
    style.panel_letter(ax, "i", dx=-0.16)
    style.tidy(ax)

    # ----------------------------------------------------------------- framing
    fig.suptitle("If the chains are paid to leave the trypsin island, where do "
                 "they land?", y=0.962)
    style.caveat(fig, "\n".join(notes), y=0.010)
    style.save(fig, "h06_escape")

    # ----------------------------------------------------------- the numbers
    table = pd.concat(tables.values(), ignore_index=True)
    out = config.RESULTS_ROOT / "h06_escape_table.csv"
    table.to_csv(out, index=False)
    print(f"wrote {out}")

    print("\n  arm                  t    <d>   gap closed  off-axis   nn(S)  "
          "called S   min d to naturals")
    for block, frame in tables.items():
        for row in frame.itertuples():
            print(f"  {block:<18} {row.t:>5g} {row.mean_d:>6.1f} "
                  f"{row.along_frac:>10.3f} {row.orth_frac:>9.3f} "
                  f"{row.frac_nn_chymotrypsin:>7.3f} {row.frac_called_chymotrypsin:>8.3f} "
                  f"{row.nn_min:>14.0f}")
    print("\n  the four refusals (natural chymotrypsin level in brackets)")
    for name, _, _ in _refusal_sites():
        chymo_level, tryp_level = refusal_reference(name)
        line = f"    {name:<9} [chym {chymo_level:.3f} / tryp {tryp_level:.3f}]  "
        for block, frame in tables.items():
            values = frame[f"refusal_{name}"]
            line += f"{block}: {values.min():.3f}..{values.max():.3f}   "
        print(line)


if __name__ == "__main__":
    main()
