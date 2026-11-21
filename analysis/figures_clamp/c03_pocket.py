"""C3 -- the falsifiable test: does an imposed serine rebuild the S1 pocket?

This is the one figure in the project that can come out wrong.

In the natural alignment the specificity pocket differs between the two families
at more than the critic site. At MSA column 241 (3TGI residue 226) the trypsins
are G = 0.91, while the chymotrypsins spread over S = 0.28, A = 0.22, T = 0.21.
That is a real, measured difference, and it was never used to build anything --
no site set, no bias, no selection rule reads column 241.

So it is a prediction. If the model has learned the pocket's epistasis, then
clamping S at 189 and relaxing should pull column 226 off glycine and toward the
chymotrypsin spread, WITHOUT being told to. If the clamped runs look like the
control, the model knows about residue 189 and nothing about the pocket -- and
the honest conclusion is that P(S) was a local rewiring all along.

The control column is what the test turns on. Cold sampling alone moves these
frequencies; only the clamp-minus-control difference is evidence.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import config
import data
import style

#: MSA column -> 3TGI label. The critic itself is included as a sanity check:
#: it must read S = 1.00 in the clamped conditions by construction, and seeing
#: anything else means the clamp did not hold.
POCKET = [(201, "189  (the critic itself)"), (229, "216"), (241, "226"), (177, "172")]

#: How many residues to show per column before the rest is pooled into "other".
TOP_N = 4


def as_sampled(path):
    """The sequences as the CHAINS left them, before substitute_critic.py.

    Every figure elsewhere reads ensembles_serine/, where the critic column has
    been overwritten with S so that all conditions are scored with the same
    residue at i*. That is right for P(S | sigma_\\i*), which does not depend on
    the residue at i* at all -- but it is wrong here, because this figure plots
    the residue at i* directly. Read from ensembles_serine/ and the control row
    would show a serine the control never sampled, turning the clamp's one
    sanity check into a tautology that passes for every condition.
    """
    raw = path.parent.parent / "ensembles" / path.name
    return data.run_sequences(str(raw if (raw / "final_sequences.fasta").is_file()
                                  else path))


def composition(encoded: np.ndarray, column: int) -> dict:
    tokens = data.alphabet().tokens
    values, counts = np.unique(encoded[:, column], return_counts=True)
    return {tokens[v]: c / len(encoded) for v, c in zip(values, counts)}


def main() -> None:
    style.apply()
    classes = data.natural_classes()
    _, encoded = data.alignment()

    # Rows: the two natural families first (what the answer would look like),
    # then the control, then the clamped conditions. Beta = 1 only: the question
    # is what the model does, not what cooling does, and adding rungs here would
    # turn a four-row comparison into a wall.
    rows = [("natural trypsins", encoded[classes["trypsin"]], style.C_TRYPSIN),
            ("natural chymotrypsins", encoded[classes["chymotrypsin"]], style.C_CHYMO)]
    for condition in config.CONDITIONS:
        runs = [r for r in data.discover_runs(condition) if abs(r.beta - 1.0) < 1e-9]
        if not runs:
            continue
        config.check_run(runs[0].path)
        rows.append((condition, as_sampled(runs[0].path),
                     style.group(condition)["color"]))
    if len(rows) <= 2:
        raise SystemExit("No beta = 1 clamp ensembles found. Run: bash work/run_clamp.sh")

    fig, axes = plt.subplots(1, len(POCKET), figsize=(18.0, 5.8))
    fig.subplots_adjust(left=0.135, right=0.99, top=0.79, bottom=0.14, wspace=0.16)

    y = np.arange(len(rows))[::-1]
    for ax, (column, label) in zip(axes, POCKET):
        # The residues worth naming are those the two NATURAL families use; the
        # legend must not be redefined by whatever a design run happens to pick.
        pooled = {}
        for _, seqs, _ in rows[:2]:
            for residue, freq in composition(seqs, column).items():
                pooled[residue] = pooled.get(residue, 0.0) + freq
        keys = [k for k, _ in sorted(pooled.items(), key=lambda kv: -kv[1])[:TOP_N]]
        colours = dict(zip(keys, style.ramp(style.VIOLET, len(keys))))

        for index, (_, seqs, _) in enumerate(rows):
            comp = composition(seqs, column)
            left = 0.0
            for residue in keys:
                width = comp.get(residue, 0.0)
                if width > 0:
                    ax.barh(y[index], width, left=left, height=0.62,
                            color=colours[residue], edgecolor="white", linewidth=1.2,
                            zorder=3)
                    if width > 0.11:
                        ax.text(left + width / 2, y[index], residue, ha="center",
                                va="center", fontsize=12, color="white",
                                fontweight="bold", zorder=4)
                left += width
            if left < 0.999:
                ax.barh(y[index], 1.0 - left, left=left, height=0.62,
                        color=style.C_OTHER, edgecolor="white", linewidth=1.2, zorder=3)

        ax.set_xlim(0, 1)
        ax.set_yticks(y)
        ax.set_yticklabels([name for name, _, _ in rows] if ax is axes[0] else [],
                           fontsize=13)
        for tick, (_, _, colour) in zip(ax.get_yticklabels(), rows):
            tick.set_color(colour)
        ax.set_xlabel("frequency")
        ax.set_title(f"column {column}  —  3TGI {label}", fontsize=14,
                     color=style.MUTED, pad=10)
        # A rule at the trypsin glycine fraction: the control should sit on it,
        # and the question is whether the clamped rows move off it.
        style.tidy(ax, grid="x")

    fig.suptitle("Does an imposed serine rebuild the rest of the pocket?", y=0.965)
    style.caveat(
        fig,
        "Columns 216, 226 and 172 were never used to build anything -- no site set, "
        "no bias, no selection rule reads them. The clamped rows moving toward the "
        "chymotrypsin row is therefore a prediction coming true; matching the control "
        "row instead means the model knows residue 189 and not the pocket.",
        y=0.012)
    style.save(fig, "c03_pocket")

    tokens = data.alphabet().tokens
    for column, label in POCKET:
        print(f"\ncolumn {column} (3TGI {label}):")
        for name, seqs, _ in rows:
            comp = composition(seqs, column)
            top = sorted(comp.items(), key=lambda kv: -kv[1])[:4]
            print(f"  {name:<26} " + "  ".join(f"{r}={f:.2f}" for r, f in top))


if __name__ == "__main__":
    main()
