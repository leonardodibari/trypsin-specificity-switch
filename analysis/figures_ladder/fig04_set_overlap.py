"""F4 -- how much do the candidate mutation sets actually agree?

The sets have different sizes and some could in principle contain others, so a
raw intersection count is not comparable across pairs. Two normalisations:

(a) **Jaccard** |A n B| / |A u B| -- symmetric, and two sets of very different
    size can never score high.
(b) **Containment** |A n B| / |A| -- the share of the ROW set that lies inside the
    COLUMN set. 1.0 means the row set is contained in the column set. This is the
    panel to read for inclusion: every set scores 1.0 against the full segment,
    but the full segment scores only ~0.2 against each of them.

Each cell also carries the raw intersection size, and in the containment panel
the overlap expected for two random sets of the same sizes drawn from the
79-column grafting segment -- without which "7 shared columns" means nothing.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import config
import style

#: Drawn in this order. "1992 construct" is the set that was actually built and
#: assayed; it replaces the wider 17-position literature list, which was never a
#: single construct.


def main() -> None:
    style.apply()
    available = config.site_sets()
    ORDER = config.site_set_order()
    missing = [n for n in ORDER if n not in available]
    if missing:
        raise SystemExit(
            f"site set(s) not available for model {config.MODEL_NAME!r}: {missing}.\n"
            "The consensus set is derived from this model's own cold run — run\n"
            "  bash analysis/run_ladder.sh graft derive\n"
            "before building this figure.")
    sets = {name: set(available[name]) for name in ORDER}
    labels = [f"{name} ({len(sets[name])})" for name in ORDER]
    n = len(ORDER)
    segment = len(available["full segment"])

    intersection = np.array([[len(sets[a] & sets[b]) for b in ORDER] for a in ORDER])
    jaccard = np.array([[len(sets[a] & sets[b]) / len(sets[a] | sets[b])
                         for b in ORDER] for a in ORDER])
    containment = np.array([[intersection[i, j] / len(sets[a])
                             for j in range(n)] for i, a in enumerate(ORDER)])

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.8))
    fig.subplots_adjust(top=0.775, bottom=0.215, left=0.145, right=0.895, wspace=0.60)

    panels = [
        (axes[0], jaccard, "Jaccard\n" + r"$|A\cap B|\,/\,|A\cup B|$", False),
        (axes[1], containment,
         "Containment\n" + r"$|A\cap B|\,/\,|A|$   (share of row set inside column set)", True),
    ]
    for ax, matrix, title, annotate_random in panels:
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n), labels, rotation=35, ha="right")
        ax.set_yticks(range(n), labels)
        ax.set_title(title, fontsize=14, pad=12)
        for i in range(n):
            for j in range(n):
                value = matrix[i, j]
                text = f"{value:.2f}\n({intersection[i, j]})"
                if annotate_random and i != j and 0 not in (i, j):
                    expected = len(sets[ORDER[i]]) * len(sets[ORDER[j]]) / segment
                    text = (f"{value:.2f}\n{intersection[i, j]} sites\n"
                            f"rand {expected:.1f}")
                ax.text(j, i, text, ha="center", va="center", fontsize=10.5,
                        color="white" if value > 0.6 else "#222222")
        ax.set_xticks(np.arange(-0.5, n), minor=True)
        ax.set_yticks(np.arange(-0.5, n), minor=True)
        ax.grid(which="minor", color="white", lw=2.2)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # No "row set A" / "column set B" axis labels: panel (b)'s own title already
    # says which direction the containment runs, and the labels collided with the
    # long tick labels, the panel letters and the caveat line.
    bar = fig.colorbar(image, ax=axes, shrink=0.72, pad=0.025)
    bar.set_label("normalised overlap", fontsize=14)

    # Panel letters in FIGURE coordinates: the row labels are long enough that an
    # axes-relative offset puts the letter off the canvas.
    for letter, x in zip("ab", (0.012, 0.487)):
        fig.text(x, 0.885, letter, fontsize=21, fontweight="bold",
                 va="top", ha="left", color=style.INK)
    fig.suptitle("Overlap of the candidate mutation sets  "
                 f"(grafting segment, columns {config.GRAFTING_SEGMENT[0]}"
                 f"–{config.GRAFTING_SEGMENT[1] - 1})", y=0.965)
    style.caveat(fig, "'rand' is the overlap expected for two random sets of the "
                      "same sizes drawn from the 79-column segment.")

    style.save(fig, "fig04_set_overlap")

    small = [name for name in ORDER if name != "full segment"]
    for i, a in enumerate(small):
        for b in small[i + 1:]:
            shared = sorted(sets[a] & sets[b])
            print(f"  {a:<16} n {b:<16} {len(shared):>2}  {shared}")
    common = sorted(set.intersection(*[sets[name] for name in small]))
    print(f"  in all {len(small)} sets except the full segment: "
          f"{common if common else 'none'}")
    print(f"  union of those {len(small)}: "
          f"{len(set.union(*[sets[n] for n in small]))} columns")


if __name__ == "__main__":
    main()
