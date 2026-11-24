"""S8 -- how much of the budget is indels, and what happens to the gaps.

THE QUESTION, and why it is not cosmetic. Every run in this study is sampled
with ``--allow-gap-moves``, so ``d_H`` counts two physically different events:
a SUBSTITUTION (both residues present, different) and an INDEL (exactly one of
the two is a gap). An experimentalist commissioning "ten mutations" means ten
substitutions; ten positions where the chain opened or closed a gap is a
different construct and, in a real protein, a different backbone.

So a mutation budget is only a budget if its composition is known. This figure
is that composition, along each ladder and across all three sets.

IT IS ALSO THE CHECK ON THE SPLIT RESTRAINT. ``t_sub`` and ``t_indel`` exist
because a single ``t`` prices both at once: asking for a shorter substitution
budget silently also charges ``t`` per gained or lost residue. Every run here was
sampled with the single ``t`` (``t_sub`` and ``t_indel`` both falling back to
it), so panel (a) is the measurement of how much that conflation actually
mattered on these sets -- and the answer decides whether the split needs to be
exercised in production or merely to exist.

THE PANELS
  (a) the indel share of d_H along each ladder: what fraction of the budget is
      not a substitution at all.
  (b) absolute counts, substitutions and indels stacked, so a falling share can
      be told from a falling total.
  (c) gap abundance: the mean number of gaps per design against the wild type's
      own gap count, which is the quantity a backbone-length argument turns on.
  (d) the distribution of gap counts at each rung -- a mean of 42 gaps can be
      1000 designs at 42 or a mixture of 38 and 46, and only this says which.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data
import style

import s05_ladder

T_LADDER = s05_ladder.T_LADDER
#: One per mutation set, from the single definition in config.
SET_COLOUR = config.SET_COLOUR


def wild_type_gaps() -> int:
    """The wild type's own gap count -- the reference every gap number is read against."""
    from encoding import Alphabet

    return int((data.wild_type() == Alphabet.from_name("protein").gap_index).sum())


def decompose(set_name: str, beta: float, run: str) -> dict:
    """Split one run's ``d_H`` into substitutions and indels, over its OWN columns.

    Restricted to the mutable columns: outside them a design IS the wild type,
    so counting the rest would add the same constant to every run and hide what
    distinguishes them -- the same reasoning ``add_natural_neighbour.py`` gives
    for its own restriction.
    """
    from encoding import Alphabet, read_sequences

    alphabet = Alphabet.from_name("protein")
    gap = alphabet.gap_index
    mutable = [int(c) for c in config.column_spec(set_name).split(",")]
    for serine in (True, False):
        path = (config.block_dir(set_name, beta, serine=serine) / run
                / "final_sequences.fasta")
        if path.is_file():
            break
    else:
        raise FileNotFoundError(f"no FASTA for {set_name}/beta{beta:g}/{run}")
    encoded = np.stack([alphabet.encode(s) for _, s in read_sequences(path)])
    reference = data.wild_type()
    block, ref = encoded[:, mutable], reference[mutable]
    differs = block != ref[None, :]
    indel = differs & ((block == gap) | (ref[None, :] == gap))
    sub = differs & ~indel
    return dict(
        d_total=float(differs.sum(axis=1).mean()),
        d_sub=float(sub.sum(axis=1).mean()),
        d_indel=float(indel.sum(axis=1).mean()),
        indel_share=float(indel.sum() / differs.sum()) if differs.sum() else np.nan,
        # Gaps over the WHOLE sequence: a backbone-length argument is about the
        # construct, not about the mutable window.
        n_gaps=float((encoded == gap).sum(axis=1).mean()),
        n_gaps_sd=float((encoded == gap).sum(axis=1).std(ddof=1)),
        gap_counts=(encoded == gap).sum(axis=1),
    )


def main() -> None:
    style.apply()
    config.ensure_dirs()

    from ensemble import run_dirname

    beta = s05_ladder.BETA
    lam = config.LAMBDAS[-1]
    wt_gaps = wild_type_gaps()

    rows, clouds = [], {}
    for set_name in config.SETS:
        for t in T_LADDER:
            run = run_dirname(lam, t)
            try:
                info = decompose(set_name, beta, run)
            except FileNotFoundError:
                continue
            clouds[(set_name, t)] = info.pop("gap_counts")
            rows.append(dict(set=set_name, beta=beta, t=t, **info))
    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit("no ladder runs found; sample them first")

    table["gaps_vs_wt"] = table["n_gaps"] - wt_gaps
    print(f"\n  indel composition and gap abundance "
          f"(beta = {beta:g}, lambda = {lam:g}; wild type carries {wt_gaps} gaps)")
    print(table[["set", "t", "d_total", "d_sub", "d_indel", "indel_share",
                 "n_gaps", "gaps_vs_wt"]].to_string(
        index=False, float_format=lambda v: f"{v:8.3f}"))

    out = config.RESULTS_ROOT / f"s08_indels_beta{beta:g}.csv"
    table.to_csv(out, index=False)
    print(f"wrote {out}")

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.0))

    # (a) indel share of the budget
    ax = axes[0, 0]
    for set_name, group in table.groupby("set", sort=False):
        group = group.sort_values("t")
        ax.plot(group["t"], 100 * group["indel_share"], marker="o", ms=6, lw=1.8,
                color=SET_COLOUR[set_name], label=set_name)
    ax.set_xlabel("restraint strength $t$")
    ax.set_ylabel(r"indel share of $d_H$  (%)")
    ax.set_title("(a)  how much of the budget is not a substitution", loc="left")
    ax.legend(frameon=False, fontsize=9)
    style.tidy(ax)

    # (b) absolute counts, stacked
    ax = axes[0, 1]
    width = 0.26
    for index, (set_name, group) in enumerate(table.groupby("set", sort=False)):
        group = group.sort_values("t")
        x = np.arange(len(group)) + (index - 1) * width
        ax.bar(x, group["d_sub"], width, color=SET_COLOUR[set_name],
               label=f"{set_name}: substitutions", zorder=3)
        ax.bar(x, group["d_indel"], width, bottom=group["d_sub"],
               color=style.mix(SET_COLOUR[set_name], style.INK, 0.45),
               label=f"{set_name}: indels", zorder=3)
    ax.set_xticks(np.arange(len(T_LADDER)))
    ax.set_xticklabels([f"{t:g}" for t in T_LADDER])
    ax.set_xlabel("restraint strength $t$")
    ax.set_ylabel(r"$\langle d_H\rangle$, split")
    ax.set_title("(b)  a falling share, or a falling total?", loc="left")
    ax.legend(frameon=False, fontsize=7.5, ncol=1)
    style.tidy(ax)

    # (c) gap abundance against the wild type's own
    ax = axes[1, 0]
    for set_name, group in table.groupby("set", sort=False):
        group = group.sort_values("t")
        ax.errorbar(group["t"], group["n_gaps"], yerr=group["n_gaps_sd"],
                    marker="o", ms=6, lw=1.8, capsize=3,
                    color=SET_COLOUR[set_name], label=set_name)
    ax.axhline(wt_gaps, color=style.C_WT, ls=":", lw=1.3, zorder=1)
    ax.text(ax.get_xlim()[1], wt_gaps, f" wild type ({wt_gaps}) ",
            color=style.C_WT, fontsize=9, va="bottom", ha="right")
    ax.set_xlabel("restraint strength $t$")
    ax.set_ylabel("gaps per design  (whole sequence)")
    ax.set_title("(c)  gap abundance; bars are the ensemble sd", loc="left")
    style.tidy(ax)

    # (d) the distribution behind the mean
    ax = axes[1, 1]
    for set_name in config.SETS:
        shades = style.ramp(SET_COLOUR[set_name], len(T_LADDER))
        for t, shade in zip(T_LADDER, shades):
            counts = clouds.get((set_name, t))
            if counts is None:
                continue
            lo, hi = int(counts.min()), int(counts.max())
            ax.hist(counts, bins=np.arange(lo - 0.5, hi + 1.5), histtype="step",
                    color=shade, lw=1.3, zorder=3)
    ax.axvline(wt_gaps, color=style.C_WT, ls=":", lw=1.3, zorder=1)
    ax.set_xlabel("gaps per design")
    ax.set_ylabel("chains")
    ax.set_title("(d)  the distribution behind that mean", loc="left")
    style.tidy(ax)

    for ax, letter in zip(axes.ravel(), "abcd"):
        style.panel_letter(ax, letter)

    fig.suptitle(
        rf"S8 — indel composition and gap abundance along the $t$ ladders "
        rf"($\beta$={beta:g}, $\lambda$={lam:g})", fontsize=13)
    style.caveat(
        fig,
        r"all runs use $\mathtt{-\!-allow\text{-}gap\text{-}moves}$, so $d_H$ counts "
        "substitutions AND indels; the split is taken over each set's own mutable "
        "columns, gap counts over the whole sequence." "\n"
        r"every run was sampled with the single $t$ ($t_{\mathrm{sub}}$ and "
        r"$t_{\mathrm{indel}}$ both falling back to it), so panel (a) measures how "
        "much that conflation actually cost on these sets.",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    style.save(fig, f"s08_indels_beta{beta:g}", out_dir=config.OUT_DIR)


if __name__ == "__main__":
    main()
