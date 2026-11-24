"""S7 -- where the mutations land: red-sector enrichment along each ladder.

THE QUESTION. The ladders show that ``t`` sets how MANY mutations a design
carries. This asks whether it also changes WHICH columns they fall in -- and in
particular whether a tighter budget is spent preferentially on the red sector,
the 17 columns the sector decomposition marks as the specificity-determining
ones. If a shorter budget concentrates on the red sector, the restraint is doing
something a referee would call design; if the composition is flat in ``t``, then
``t`` buys fewer mutations of the same kind.

THE DENOMINATOR IS THE TRAP, and CLAUDE.md states it: *sector enrichment
denominators use the number of MUTABLE columns, never the number of columns --
otherwise a site restriction manufactures its own enrichment.* Every number here
is therefore a share of the mutable columns of that set:

    enrichment = (share of mutations in red) / (share of mutable columns in red)

so 1.0 means "exactly as the column set would give by chance" and 2.0 means
"twice as concentrated in red as the set itself is".

WHAT THIS MEANS PER SET, computed rather than assumed:

    segment      16 of 79 mutable columns are red (20.3 %)  -> informative
    coupling20    7 of 20 mutable columns are red (35.0 %)  -> informative
    redsector    16 of 16 mutable columns are red (100 %)   -> DEGENERATE

For the red-sector set the enrichment is identically 1.0 by construction: every
column it can touch is red. That is not a result and the figure says so in
words rather than drawing a flat line and letting a reader take it for one.

THE CURVE, panel (b). The red sector is RANKED -- ``sectors_iter_aln_v2.csv``
carries ``red_rank``, 1 = most important, from the sector decomposition, and
that ranking is NOT model-derived, so it carries between models (CLAUDE.md).
Panel (b) is the cumulative share of a run's mutations against that rank: a
curve bowing above the diagonal means the mutations concentrate in the columns
the sector calls most important, and the diagonal itself is what a set that
ignored the ranking would give.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data
import style

import s05_ladder

#: t rungs of the ladder, same as s05.
T_LADDER = s05_ladder.T_LADDER

#: One per mutation set, from the single definition in config.
SET_COLOUR = config.SET_COLOUR


def sectors() -> tuple[Dict[int, str], Dict[int, int]]:
    """``(column -> sector, column -> red rank)`` from the repository's own CSV.

    Read, never recomputed. The red sector is one of the few things in this
    study that is NOT model-derived, so it carries between models -- which is
    exactly why it must come from the file rather than from a fit.
    """
    frame = pd.read_csv(config.REPO / "data" / "sectors_iter_aln_v2.csv")
    sector = dict(zip(frame["msa_col"].astype(int), frame["sector"].astype(str)))
    ranks = pd.to_numeric(frame["red_rank"], errors="coerce")
    red_rank = {
        int(c): int(r) for c, r in zip(frame["msa_col"].astype(int), ranks)
        if np.isfinite(r)
    }
    return sector, red_rank


def mutation_counts(set_name: str, beta: float, run: str) -> np.ndarray:
    """Per-column count of chains that differ from the wild type, length L."""
    from encoding import Alphabet, read_sequences

    alphabet = Alphabet.from_name("protein")
    for serine in (True, False):
        path = (config.block_dir(set_name, beta, serine=serine) / run
                / "final_sequences.fasta")
        if path.is_file():
            encoded = np.stack([alphabet.encode(s) for _, s in read_sequences(path)])
            return (encoded != data.wild_type()[None, :]).sum(axis=0)
    raise FileNotFoundError(f"no FASTA for {set_name}/beta{beta:g}/{run}")


def enrichment_table(beta: float) -> pd.DataFrame:
    """Red-sector enrichment of every rung of every set's ladder."""
    from ensemble import run_dirname

    sector, _ = sectors()
    red = {c for c, s in sector.items() if s == "red"}
    rows = []
    for set_name in config.SETS:
        mutable = [int(c) for c in config.column_spec(set_name).split(",")]
        red_mutable = [c for c in mutable if c in red]
        baseline = len(red_mutable) / len(mutable)
        for t in T_LADDER:
            run = run_dirname(config.LAMBDAS[-1], t)
            try:
                counts = mutation_counts(set_name, beta, run)
            except FileNotFoundError:
                continue
            total = counts[mutable].sum()
            in_red = counts[red_mutable].sum()
            share = in_red / total if total else np.nan
            rows.append(dict(
                set=set_name, beta=beta, t=t, n_mutable=len(mutable),
                n_red_mutable=len(red_mutable), baseline=baseline,
                mutations=int(total), in_red=int(in_red), red_share=share,
                enrichment=share / baseline if baseline else np.nan,
                degenerate=len(red_mutable) == len(mutable),
            ))
    return pd.DataFrame(rows)


def cumulative_by_rank(set_name: str, beta: float, run: str,
                       red_rank: Dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative share of a run's red-sector mutations against red rank."""
    mutable = [int(c) for c in config.column_spec(set_name).split(",")]
    ranked = sorted((r, c) for c, r in red_rank.items() if c in mutable)
    if not ranked:
        return np.array([]), np.array([])
    counts = mutation_counts(set_name, beta, run)
    ordered = np.array([counts[c] for _, c in ranked], dtype=float)
    total = ordered.sum()
    if total == 0:
        return np.array([]), np.array([])
    x = np.arange(1, len(ordered) + 1) / len(ordered)
    return x, np.cumsum(ordered) / total


def main() -> None:
    style.apply()
    config.ensure_dirs()

    beta = s05_ladder.BETA
    table = enrichment_table(beta)
    if table.empty:
        raise SystemExit("no ladder runs found; sample them first")

    print(f"\n  red-sector enrichment, beta = {beta:g}, "
          f"lambda = {config.LAMBDAS[-1]:g}")
    print("  (share of mutations in red) / (share of MUTABLE columns in red)")
    print(table[["set", "t", "n_mutable", "n_red_mutable", "baseline",
                 "red_share", "enrichment", "degenerate"]].to_string(
        index=False, float_format=lambda v: f"{v:8.3f}"))

    out = config.RESULTS_ROOT / f"s07_sectors_beta{beta:g}.csv"
    table.to_csv(out, index=False)
    print(f"wrote {out}")

    _, red_rank = sectors()
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.4))

    # (a) enrichment against t, one line per set
    ax = axes[0]
    for set_name, group in table.groupby("set", sort=False):
        group = group.sort_values("t")
        degenerate = bool(group["degenerate"].iloc[0])
        ax.plot(group["t"], group["enrichment"],
                ls=":" if degenerate else "-",
                marker="o", ms=6, lw=1.8, color=SET_COLOUR[set_name],
                label=f"{set_name} "
                      f"({group['n_red_mutable'].iloc[0]}/{group['n_mutable'].iloc[0]} red"
                      + (", degenerate)" if degenerate else ")"))
    ax.axhline(1.0, color=style.RULE, ls="--", lw=1.2, zorder=1)
    ax.text(ax.get_xlim()[1], 1.0, " chance, given the set ", color=style.MUTED,
            fontsize=9, va="bottom", ha="right")
    ax.set_xlabel("restraint strength $t$")
    ax.set_ylabel("red-sector enrichment")
    ax.set_title("(a)  does a tighter budget go to the red sector?", loc="left")
    ax.legend(frameon=False, fontsize=9)
    style.tidy(ax)

    # (b) the curve: cumulative mutation share against red rank
    ax = axes[1]
    for set_name in ("segment", "coupling20"):
        if set_name not in set(table["set"]):
            continue
        shades = style.ramp(SET_COLOUR[set_name], len(T_LADDER))
        for t, shade in zip(T_LADDER, shades):
            from ensemble import run_dirname

            run = run_dirname(config.LAMBDAS[-1], t)
            try:
                x, y = cumulative_by_rank(set_name, beta, run, red_rank)
            except FileNotFoundError:
                continue
            if x.size == 0:
                continue
            ax.plot(np.concatenate([[0], x]), np.concatenate([[0], y]),
                    color=shade, lw=1.6,
                    label=f"{set_name}, $t$={t:g}" if t in (T_LADDER[0], T_LADDER[-1])
                    else None, zorder=3)
    ax.plot([0, 1], [0, 1], color=style.RULE, ls="--", lw=1.2, zorder=1)
    ax.text(0.98, 0.94, "ignores the ranking", color=style.MUTED, fontsize=9,
            ha="right", transform=ax.transAxes)
    ax.set_xlabel("red-sector columns, most important first  (fraction)")
    ax.set_ylabel("cumulative share of red-sector mutations")
    ax.set_title("(b)  within the red sector, do they follow its ranking?",
                 loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style.tidy(ax)

    for ax, letter in zip(axes, "ab"):
        style.panel_letter(ax, letter)

    fig.suptitle(
        rf"S7 — red-sector enrichment along the $t$ ladders "
        rf"($\beta$={beta:g}, $\lambda$={config.LAMBDAS[-1]:g})", fontsize=13)
    style.caveat(
        fig,
        "enrichment denominators are the number of MUTABLE columns of each set, "
        "never the number of columns — otherwise a site restriction manufactures "
        "its own enrichment (CLAUDE.md)." "\n"
        r"$\mathtt{redsector}$ is dotted because its enrichment is identically 1 "
        "by construction: all 16 of its mutable columns are red.  the red "
        "ranking is NOT model-derived and carries between models.",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.93))
    style.save(fig, f"s07_sectors_beta{beta:g}", out_dir=config.OUT_DIR)


if __name__ == "__main__":
    main()
