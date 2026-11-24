"""H2 -- does the restraint beat cooling? The test that justifies the term.

The ladder already has a way of keeping designs close to the wild type: cooling.
Turning beta up from 1 to 10 takes the mean Hamming distance from 12 to 7 and
buys about 9 in Q. But it does that by FREEZING the chain -- acceptance falls
from 0.034 to 0.0015, the beta=2 and beta=5 ensembles differ by two columns out
of 265, and RESULTS.md cannot tell a converged cold run from a stuck one. It
also collapses the library: 8 % unique designs at beta = 10.

The restraint keeps designs close a different way. It is a field, so it tilts
the equilibrium rather than slowing the sampler, and the chains stay warm.

So the question is not "does t reduce the mutation count" -- it does, by
construction -- but whether at MATCHED serine and MATCHED mutation count it
gives a better library than cooling does. Four panels, four axes of that
comparison:

  (a) Q against P(S)          quality at matched specificity
  (b) <d> against P(S)        mutation budget at matched specificity
  (c) unique designs against <d>   the library, at matched budget
  (d) accepted moves per site against <d>   settled, or merely frozen

Read (a) and (c) together. If cooling wins (a) and loses (c), the two protocols
are buying different things and neither is simply better -- which is the honest
answer, and the one this figure is built to be able to give.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data
import style

BLOCK = "a1_graft"

#: The restraint ladder: one line per lambda, t increasing along it.
RESTRAINT_COLOUR = style.BLUE
#: The ladder's cooling runs: one line per lam_tilde, beta increasing along it.
COOLING_COLOUR = style.ORANGE


def restraint_table() -> pd.DataFrame:
    """One row per (lambda, t) run of A1, with the library size attached."""
    frame = data.designs(BLOCK)
    table = data.summarise(frame, by=("lam", "t"))
    table["unique_pct"] = [
        100 * data.unique_fraction(BLOCK, _run_name(row.lam, row.t))
        for row in table.itertuples()
    ]
    table["accepted_per_site"] = [
        data.verdict(BLOCK, _run_name(row.lam, row.t)).accepted_per_site
        for row in table.itertuples()
    ]
    return table.sort_values(["lam", "t"])


def _run_name(lam: float, t: float) -> str:
    from ensemble import run_dirname

    return run_dirname(lam, t)


def cooling_table() -> pd.DataFrame:
    """One row per ladder run, warm and cooled, with the library size attached.

    These are the runs the restraint is measured against. They are read through
    ``ensembles_serine``, rescored at beta = 1, because a cooled run's own P(S)
    saturates at 1.0 and is not comparable (RESULTS.md section 5).
    """
    frame = data.cooling_ladder()
    # The gate, applied to the competitor as well as to the restraint. It used
    # to be applied only to A1, because load_ladder asserted the ladder runs were
    # saturated; it now measures them, and two of the 21 fail -- so without this
    # line the comparison would gate one side and not the other.
    dropped = sorted(set(frame.loc[~frame["saturated"], "run"]))
    if dropped:
        print(f"  cooling ladder: dropping {len(dropped)} unsaturated run(s): "
              f"{', '.join(dropped)}")
    frame = frame[frame["saturated"]]
    table = data.summarise(frame, by=("block", "lam", "beta"))
    uniques, accepted = [], []
    for row in table.itertuples():
        directory = (
            config.LADDER_GRAFTING
            if row.beta == 1.0
            else config.LADDER_COOLING[row.beta]
        )
        uniques.append(100 * _unique_from(directory / f"lam{row.lam:g}"))
        # The acceptance lives with the run that was SAMPLED, not with its
        # rescoring: ensembles/, one directory up from ensembles_serine/.
        accepted.append(
            _accepted_per_site(directory.parent / "ensembles" / f"lam{row.lam:g}")
        )
    table["unique_pct"] = uniques
    table["accepted_per_site"] = accepted
    # lam_tilde = lambda / beta is what the cooled ladder holds fixed down a
    # rung; it is the quantity being optimised, beta only says how cold.
    table["lam_tilde"] = table["lam"] / table["beta"]
    return table.sort_values(["lam_tilde", "beta"])


def _accepted_per_site(directory: Path) -> float:
    """Accepted moves per mutable site of a sampled run, from its summary.

    This is the axis on which cooling and the restraint differ in kind rather
    than in degree: cooling keeps a design near the wild type by refusing moves,
    the restraint by preferring a different equilibrium.
    """
    import json

    path = Path(directory) / "summary.json"
    if not path.is_file():
        return float("nan")
    with open(path, encoding="utf-8") as handle:
        summary = json.load(handle)
    sites = summary.get("n_allowed_sites") or 1
    return summary["acceptance_rate"] * summary["n_steps"] / sites


def _unique_from(directory: Path) -> float:
    from encoding import read_sequences

    path = Path(directory) / "final_sequences.fasta"
    if not path.is_file():
        return float("nan")
    sequences = [s for _, s in read_sequences(path)]
    return len(set(sequences)) / len(sequences) if sequences else float("nan")


def _shades(table, group, colour):
    """Group value -> shade, along a single-hue ramp ordered by the group."""
    values = sorted(table[group].unique())
    return dict(zip(values, style.ramp(colour, max(len(values), 2))))


def _draw_paths(ax, table, x, y, *, group, colour, marker):
    """One line per group, light to dark along the ladder within it.

    No inline labels: eight paths converge in the top-right corner of every
    panel and the labels collided there. The ramp is in the figure legend.
    """
    for value, shade in _shades(table, group, colour).items():
        part = table[table[group] == value].sort_values(x)
        ax.plot(part[x], part[y], color=shade, lw=1.8, marker=marker,
                markersize=6, markeredgecolor="white", markeredgewidth=0.8,
                zorder=3)


def main() -> None:
    style.apply()
    config.ensure_dirs()
    restraint = restraint_table()
    cooling = cooling_table()

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.6))

    # (a) quality at matched specificity
    ax = axes[0, 0]
    _draw_paths(ax, restraint, "P_S", "Q", group="lam", colour=RESTRAINT_COLOUR,
                marker="o")
    _draw_paths(ax, cooling, "P_S", "Q", group="lam_tilde", colour=COOLING_COLOUR,
                marker="s")
    ax.set_xlabel(style.P_S)
    ax.set_ylabel(style.LABELS["Q"])
    style.panel_letter(ax, "a")
    style.tidy(ax)

    # (b) mutation budget at matched specificity
    ax = axes[0, 1]
    _draw_paths(ax, restraint, "P_S", "hamming", group="lam", colour=RESTRAINT_COLOUR,
                marker="o")
    _draw_paths(ax, cooling, "P_S", "hamming", group="lam_tilde", colour=COOLING_COLOUR,
                marker="s")
    ax.set_xlabel(style.P_S)
    ax.set_ylabel(style.LABELS["hamming_to_wt"])
    style.panel_letter(ax, "b")
    style.tidy(ax)

    # (c) the library, at matched budget
    ax = axes[1, 0]
    _draw_paths(ax, restraint, "hamming", "unique_pct", group="lam",
                colour=RESTRAINT_COLOUR, marker="o")
    _draw_paths(ax, cooling, "hamming", "unique_pct", group="lam_tilde",
                colour=COOLING_COLOUR, marker="s")
    ax.set_xlabel(style.LABELS["hamming_to_wt"])
    ax.set_ylabel("unique designs (%)")
    ax.set_ylim(0, 104)
    style.panel_letter(ax, "c")
    style.tidy(ax)

    # (d) settled or merely frozen
    ax = axes[1, 1]
    for shade, lam in zip(style.ramp(RESTRAINT_COLOUR, len(restraint["lam"].unique())),
                          sorted(restraint["lam"].unique())):
        part = restraint[restraint["lam"] == lam].sort_values("hamming")
        ax.plot(part["hamming"], part["accepted_per_site"], color=shade, lw=1.8,
                marker="o", markersize=6, markeredgecolor="white",
                markeredgewidth=0.8, zorder=3)
    ax.axhline(5.0, color=style.RED, lw=1.5, ls=(0, (4, 3)))
    ax.text(ax.get_xlim()[0], 5.4, "below this a chain is frozen, not settled",
            color=style.RED, fontsize=11.5, va="bottom")
    ax.set_xlabel(style.LABELS["hamming_to_wt"])
    ax.set_ylabel("accepted moves per mutable site")
    ax.set_yscale("log")
    style.panel_letter(ax, "d")
    style.tidy(ax)

    # One legend for both ramps. Each entry is a path, and within a path the
    # marker darkens as the knob is turned up -- t for the restraint, beta for
    # cooling -- so the legend carries the ladder as well as the condition.
    handles = [
        plt.Line2D([], [], color=shade, marker="o", lw=1.8, markersize=6,
                   label=rf"$\lambda$={value:g}")
        for value, shade in _shades(restraint, "lam", RESTRAINT_COLOUR).items()
    ]
    handles += [
        plt.Line2D([], [], color=shade, marker="s", lw=1.8, markersize=6,
                   label=rf"$\tilde\lambda$={value:g}")
        for value, shade in _shades(cooling, "lam_tilde", COOLING_COLOUR).items()
    ]
    legend = fig.legend(
        handles=handles, loc="upper center", ncol=len(handles) // 2 + len(handles) % 2,
        bbox_to_anchor=(0.5, 0.995), fontsize=12.5, columnspacing=1.2,
        handletextpad=0.4,
    )
    legend.set_title(
        r"circles, blue: restraint ($\beta=1$, $t$ = 0 $\rightarrow$ 2, darkening)"
        "      "
        r"squares, orange: cooling ($t=0$, $\beta$ = 1 $\rightarrow$ 10, darkening)",
        prop={"size": 13},
    )
    fig.suptitle("Two ways to keep a design near the wild type", y=0.905)
    style.caveat(
        fig,
        "grafting window, 79 columns.  both sets rescored at "
        r"$\beta=1$ through substitute_critic, the only form in which a cooled run and a "
        r"warm one are comparable.  the cooled runs carry no acceptance trace, so (d) is "
        "the restraint alone.",
    )
    fig.tight_layout(rect=(0, 0.028, 1, 0.885))
    style.save(fig, "h02_pareto")

    # The table behind the figure, so a number in the README can be checked.
    out = config.RESULTS_ROOT / "h02_pareto_table.csv"
    columns = ["lam", "t", "P_S", "hamming", "Q", "unique_pct", "accepted_per_site",
               "E_DCA", "CDE", "LOR", "frac_switched"]
    restraint[columns].to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
