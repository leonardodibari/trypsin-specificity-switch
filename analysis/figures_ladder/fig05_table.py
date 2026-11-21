"""F5 -- the selected designs, side by side.

The rows are hand-picked (see ``selection.SELECTED``), not chosen by a formula.
Five desiderata pull against each other and any single score would bury the
trade-off under arbitrary weights; the honest version is to show the numbers and
let a reader see which run wins which column.

**In every column the best value is bold** -- highest or lowest depending on what
the column means (``selection.BETTER_HIGH``). Ties bold every tied row.

Every column is recomputed from ``final_sequences.fasta`` rather than scraped
from the three summary CSVs under results/ladder/, which carry three different
schemas and none of which holds the log-odds or the diversity ratio.
"""

from __future__ import annotations

from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import selection
import style


def _num(fmt: str):
    return lambda v: "—" if pd.isna(v) else format(v, fmt)


#: (dataframe column, header, width weight, formatter)
COLUMNS: List[tuple] = [
    ("condition", "condition", 2.30, lambda v: str(v)),
    ("beta", r"$\beta$", 0.58, _num("g")),
    ("lam", r"$\lambda$", 0.70, _num("g")),
    ("n_allowed_sites", "allowed\nsites", 0.85, _num(".0f")),
    # Substitutions and indels shown apart: --allow-gap-moves lets the sampler
    # delete or insert, and a single "mean Hamming" hides that 60% of the warm
    # grafting library carries an indel.
    ("subst_to_wt", "mean subst.\nto wild type", 1.15, _num(".1f")),
    ("indel_to_wt", "mean\nindels", 0.80, _num(".2f")),
    ("pct_pure_subst", "% chains\nindel-free", 1.00, _num(".0f")),
    ("Q", r"mean $Q_\mu$", 1.00, _num(".1f")),
    ("P_S", style.P_S, 1.12, _num(".3f")),
    ("log_odds", r"$\log\frac{P(\mathrm{S})}{P(\mathrm{D})}$", 0.88, _num(".2f")),
    ("pct_P_S_over_080", "% chains\n$P(\\mathrm{S})>0.8$", 1.05, _num(".0f")),
    ("pct_unique", "%\nunique", 0.78, _num(".1f")),
    ("mean_pair_hamming", "mean pairwise\nHamming", 1.15, _num(".1f")),
    ("pair_ratio", "pairwise /\nHamming", 1.05, _num(".2f")),
]

#: Printed under the header so the reader knows which way each column reads.
DIRECTION = {True: r"$\uparrow$", False: r"$\downarrow$"}


def render(table: pd.DataFrame, *, stem: str, title: str, subtitle: str = "",
           columns: Sequence[tuple] = tuple(COLUMNS), row_height: float = 0.52,
           note: str = "", group_rules: bool = True) -> None:
    """Draw a DataFrame as a figure-quality table, bolding the best cell per column.

    Hand-drawn rather than ``plt.table``: the stock helper cannot do per-cell
    weight, grouped rules, or maths in the headers.
    """
    style.apply()
    n_rows = len(table)
    weights = np.array([c[2] for c in columns], dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(weights / weights.sum())])
    centres = (edges[:-1] + edges[1:]) / 2
    best = {key: selection.best_in_column(table, key) for key, *_ in columns}

    header_h = 1.60
    note_lines = 1 + note.count("\n")
    # Reserve real height for the footnote: a two-line note at 11 pt ran into the
    # table's bottom rule when the axes were given a fixed fractional margin.
    note_h = 0.30 + 0.24 * note_lines
    fig_h = 1.95 + note_h + row_height * (n_rows + header_h)
    bottom = note_h / fig_h
    fig = plt.figure(figsize=(19.2, fig_h))
    ax = fig.add_axes([0.028, bottom, 0.942, 0.845 - bottom])
    ax.set_xlim(0, 1)
    ax.set_ylim(-(n_rows + header_h), 0.40)
    ax.axis("off")

    for (key, head, _, _), x in zip(columns, centres):
        ax.text(x, -0.40, head, ha="center", va="center", fontsize=13,
                fontweight="bold", color=style.INK, linespacing=1.45)
        if key in selection.BETTER_HIGH:
            ax.text(x, -1.22, DIRECTION[selection.BETTER_HIGH[key]], ha="center",
                    va="center", fontsize=13, color=style.MUTED)
    ax.plot([0, 1], [-header_h, -header_h], color=style.INK, lw=1.6)

    previous = None
    for row_index, (_, row) in enumerate(table.iterrows()):
        y = -(header_h + 0.5 + row_index)
        placeholder = bool(row.get("placeholder", False))
        base_colour = style.MUTED if placeholder else style.INK

        if row_index % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#f6f5f2", zorder=0)
        if group_rules and previous is not None and row["condition"] != previous:
            ax.plot([0, 1], [y + 0.5, y + 0.5], color=style.FAINT, lw=1.1)
        previous = row["condition"]

        for (key, _, _, fmt), x, left in zip(columns, centres, edges[:-1]):
            if key == "condition":
                text = str(row[key])
                if placeholder:
                    text += "\n(not sampled — placeholder)"
                ax.text(left + 0.010, y, text, ha="left", va="center",
                        fontsize=12.5, color=base_colour, linespacing=1.3, zorder=2)
                continue
            is_best = bool(best[key][row_index]) and not placeholder
            ax.text(x, y, fmt(row.get(key, np.nan)), ha="center", va="center",
                    fontsize=13.5 if is_best else 13,
                    color=style.INK if is_best else base_colour,
                    fontweight="bold" if is_best else "normal", zorder=2)

    ax.plot([0, 1], [-(n_rows + header_h), -(n_rows + header_h)],
            color=style.INK, lw=1.6)

    fig.suptitle(title, y=0.985)
    if subtitle:
        fig.text(0.5, 0.912, subtitle, ha="center", va="top", fontsize=13,
                 color=style.MUTED)
    if note:
        fig.text(0.5, 0.06 * note_h / fig_h, note, ha="center", va="bottom",
                 fontsize=11, color=style.MUTED, linespacing=1.5)
    style.save(fig, stem)


def main() -> None:
    table = selection.select_all()
    render(
        table,
        stem="fig05_table",
        title="The selected designs",
        subtitle=selection.describe_selection(),
        note=("↑ / ↓ marks which direction is better in each column; ties bold every tied row. "
              "Substitutions and indels are counted apart — the sampler may delete a "
              "wild-type residue or fill a wild-type gap, and those are not point mutations.\n"
              "'pairwise / Hamming' is library spread relative to distance travelled: "
              "~1 means chains differ from each other as much as they differ from the "
              "wild type, ~0 means the library has collapsed to a single sequence."),
    )
    show = ["condition", "beta", "lam", "n_allowed_sites", "subst_to_wt",
            "indel_to_wt", "pct_pure_subst", "Q", "P_S", "log_odds",
            "pct_P_S_over_080", "pct_unique", "mean_pair_hamming", "pair_ratio"]
    print(table[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nbest per column, among the selected rows:")
    for key in selection.BETTER_HIGH:
        mask = selection.best_in_column(table, key)
        for _, row in table[mask].iterrows():
            print(f"  {key:<20} {row[key]:>9.3f}   {row['condition']}"
                  f"  β={row['beta']:g}, λ={row['lam']:g}")


if __name__ == "__main__":
    main()
