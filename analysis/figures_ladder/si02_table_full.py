"""S2 -- every run, so the selection in F5 can be checked and re-made.

F5 shows a handful of runs picked by hand. This is the pool they were picked
from, in scan order, with the best value in each column bold. Reading down a
column shows what the whole scan can do on that axis; reading across a row shows
what one setting costs on the others.

Use this to change the selection: pick the rows you want and list them in
``selection.SELECTED``.
"""

from __future__ import annotations

import fig05_table
import selection


def main() -> None:
    pool = selection.full_pool()
    fig05_table.render(
        pool,
        stem="si02_table_full",
        title="Every sampled run — the pool behind figure F5",
        subtitle="bold = best value in that column, over the whole pool",
        row_height=0.46,
        note=("↑ / ↓ marks which direction is better in each column; ties bold every tied row.\n"
              "No run is 'best' outright: the column winners are spread across "
              "conditions, which is why the selection is made by hand rather than "
              "by a single score."),
    )
    show = ["condition", "beta", "lam", "n_allowed_sites", "subst_to_wt",
            "indel_to_wt", "pct_pure_subst", "Q", "P_S", "log_odds",
            "pct_P_S_over_080", "pct_unique", "mean_pair_hamming", "pair_ratio"]
    print(pool[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{len(pool)} runs in the pool")
    winners = {}
    for key in selection.BETTER_HIGH:
        mask = selection.best_in_column(pool, key)
        for _, row in pool[mask].iterrows():
            winners.setdefault(
                (row["condition"], row["beta"], row["lam"]), []).append(key)
    print("column winners by run:")
    for (condition, beta, lam), keys in winners.items():
        print(f"  {condition:<24} β={beta:g}, λ={lam:g}   {', '.join(keys)}")


if __name__ == "__main__":
    main()
