"""S4 -- the selected runs side by side, with each requirement as its own column.

The analogue of ``figures_ladder/fig05_table.py``. The point of a table here,
rather than a ranking, is that the four requirements pull against each other:
a shorter mutation budget costs quality, more quality costs diversity, and the
specificity has to survive both. Any single score would bury that under weights
nobody chose and then report its own weighting as the result.

So: **in every column the best value is bold**, highest or lowest depending on
what the column means (``selection.BETTER_HIGH``), and no row is "the winner".
A reader sees which run wins which column and picks according to what they are
actually buying.

The four admissibility columns are shown as well, with the threshold in the
header, so it is visible that every listed run cleared all four -- and
``s02_means.csv`` holds the 48 rows that did not.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data
import selection
import style

#: column key -> (header, format, is this one of the four requirements)
COLUMNS = [
    ("set", "set", "{}", False),
    ("beta", r"$\beta$", "{:g}", False),
    ("lam", r"$\lambda$", "{:g}", False),
    ("t", "$t$", "{:g}", False),
    ("hamming", r"$\langle d\rangle$", "{:.1f}", True),
    ("hamming_sd", r"sd$(d)$", "{:.1f}", False),
    ("E_DCA", "$E$", "{:.1f}", False),
    ("Q", "$Q$", "{:.1f}", True),
    ("P_S", "$P(S)$", "{:.3f}", True),
    ("LOR", "LOR", "{:.1f}", False),
    ("n_unique", "distinct", "{:.0f}", True),
    ("divergence_ratio", "div/WT", "{:.2f}", True),
    ("accepted_per_site", "acc/site", "{:.0f}", False),
]


def main() -> None:
    style.apply()
    config.ensure_dirs()

    frame = data.load_all(serine=True, only_saturated=True)
    table = data.summarise(frame)
    q_wt = selection.wild_type_Q()
    chosen = selection.select(table, q_wt)
    if chosen.empty:
        raise SystemExit("no admissible run -- see s02_means.csv")

    # Which cell of each column is the best one, for the bolding.
    best: dict[str, set[int]] = {}
    for key, _, _, _ in COLUMNS:
        if key not in chosen or key not in selection.BETTER_HIGH:
            continue
        series = chosen[key].astype(float)
        target = series.max() if selection.BETTER_HIGH[key] else series.min()
        best[key] = set(series.index[np.isclose(series, target)])

    headers, cells, bolds = [], [], []
    for key, header, fmt, required in COLUMNS:
        if key not in chosen:
            continue
        mark = "*" if required else ""
        headers.append(f"{header}{mark}")
        column, bold = [], []
        for index, value in chosen[key].items():
            column.append(fmt.format(value))
            bold.append(index in best.get(key, set()))
        cells.append(column)
        bolds.append(bold)

    n_rows, n_cols = len(chosen), len(headers)
    fig, ax = plt.subplots(figsize=(1.35 * n_cols + 1.0, 0.62 * n_rows + 2.6))
    ax.axis("off")

    table_artist = ax.table(
        cellText=list(map(list, zip(*cells))),
        colLabels=headers,
        cellLoc="center",
        loc="upper center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(10.5)
    table_artist.scale(1.0, 1.55)

    for (row, col), cell in table_artist.get_celld().items():
        cell.set_edgecolor(style.FAINT)
        if row == 0:
            cell.set_text_props(weight="bold", color=style.INK)
            cell.set_facecolor(style.GRID)
        elif bolds[col][row - 1]:
            cell.set_text_props(weight="bold", color=style.INK)

    ax.set_title(
        f"S4 — the {n_rows} selected runs; in every column the best value is bold",
        fontsize=13, pad=18,
    )
    style.caveat(
        fig,
        "* marks one of the four requirements.  thresholds: "
        rf"$\langle d\rangle\leq$ per-set ceiling, $Q\leq Q_{{\mathrm{{WT}}}}={q_wt:.1f}$, "
        rf"$P(S)\geq{selection.P_S_FLOOR:.2f}$, "
        rf"distinct $\geq{selection.N_UNIQUE_FLOOR}$ OR divergence ratio $>{selection.DIVERGENCE_FLOOR}$."
        "\n"
        "no row is the winner — the requirements trade against each other, and a "
        "single score would hide which one each run paid with.  the 48 rows that "
        "did not clear all four are in s02_means.csv.",
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1.0))
    style.save(fig, "s04_table", out_dir=config.OUT_DIR)

    out = config.RESULTS_ROOT / "s04_table.csv"
    chosen.to_csv(out, index=False)
    print(f"wrote {out}")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(chosen[[c for c, *_ in COLUMNS if c in chosen]].to_string(index=False))


if __name__ == "__main__":
    main()
