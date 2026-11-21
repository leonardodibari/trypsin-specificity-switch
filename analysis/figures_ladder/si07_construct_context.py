"""S7 -- why the literature constructs score so low, and what that implies for ours.

The reconstructed 1992 series reaches only P(S | sigma_\\i*) = 0.05 under this
model, and the obvious worry is that the model is simply unfamiliar with those
sequences -- that they fall somewhere the training data never went. This figure
tests that and finds the opposite.

(a) the train/test split by specificity class: is either class under-represented?
(b) how far each construct sits from the nearest training sequence, the nearest
    held-out sequence, and the nearest natural trypsin and chymotrypsin.
(c) the quantity that actually explains it: what FRACTION of the wild-type ->
    chymotrypsin sequence gap each construct and each of our designs closes,
    against the P(S) it is awarded.

The wild type sits 82 residues from the nearest natural chymotrypsin. The 1992
constructs close none of that gap -- and neither do our own designs, which close
at most ~5%. Every one of them is a trypsin with a locally rewired S1 site, and
P(S) is high because the local context at i* has changed, not because the
protein has moved toward the chymotrypsin family. That is the honest reading of
every P(S) number in this deck.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import config
import data
import selection
import style

MAX_DESIGN = 200       # sequences per design run for the distance sweep
CLASSES = [("trypsin", "natural trypsins (D at $i^*$)", style.C_TRYPSIN),
           ("chymotrypsin", "natural chymotrypsins (S at $i^*$)", style.C_CHYMO),
           ("other", "other naturals", style.C_OTHER)]


def main() -> None:
    style.apply()
    names, encoded = data.alignment()
    classes = data.natural_classes()
    held_out = data.split_mask()
    if held_out is None:
        raise SystemExit("This figure needs the train/test split; see data.split_mask.")
    train = ~held_out
    length = encoded.shape[1]
    trypsins = encoded[classes["trypsin"]]
    chymos = encoded[classes["chymotrypsin"]]

    series = data.construct_series()
    constructs = np.stack([s for _, s in series])
    construct_P = data.critic_conditional(constructs)
    s_state = data.S_STATE()
    wild = data.wild_type()
    base_gap = float(data.nearest_natural(wild[None, :], chymos)[0][0])

    fig = plt.figure(figsize=(18.5, 7.4))
    gs = fig.add_gridspec(1, 3, left=0.055, right=0.985, top=0.80, bottom=0.255,
                          wspace=0.30, width_ratios=[1.0, 1.25, 1.25])
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

    # --- (a) split composition -------------------------------------------------
    width = 0.38
    for offset, mask, label, hatch in ((-width / 2, train, "train", None),
                                       (width / 2, held_out, "held out", "///")):
        counts = [int((classes[key] & mask).sum()) for key, _, _ in CLASSES]
        ax_a.bar(np.arange(len(CLASSES)) + offset, counts, width,
                 color=[c for _, _, c in CLASSES], alpha=0.95 if hatch is None else 0.55,
                 edgecolor="white", hatch=hatch, linewidth=1.0, zorder=3)
        for x, value in zip(np.arange(len(CLASSES)) + offset, counts):
            ax_a.text(x, value, f"{value:,}", ha="center", va="bottom",
                      fontsize=10.5, color=style.MUTED, rotation=90)
    ax_a.set_xticks(range(len(CLASSES)),
                    ["trypsins\n(D at $i^*$)", "chymotrypsins\n(S at $i^*$)", "other"],
                    fontsize=12)
    ax_a.set_ylabel("sequences")
    ax_a.set_ylim(0, 1.32 * max(int((classes[k] & train).sum()) for k, _, _ in CLASSES))
    ax_a.set_title("the split is balanced across classes", fontsize=14,
                   color=style.MUTED, pad=10)
    ax_a.legend(handles=[
        Line2D([], [], marker="s", ls="none", color=style.MUTED, markersize=11,
               label="train"),
        Line2D([], [], marker="s", ls="none", markerfacecolor="white",
               markeredgecolor=style.MUTED, markersize=11, label="held out"),
    ], loc="upper right", fontsize=12)
    style.tidy(ax_a, grid="y")
    for key, _, _ in CLASSES:
        pass

    # --- (b) how far is each construct from what the model saw? ----------------
    pools = [("nearest\ntraining seq", train, style.VIOLET),
             ("nearest\nheld-out seq", held_out, style.AQUA)]
    group = np.arange(len(series))
    bar_w = 0.38
    for i, (label, mask, colour) in enumerate(pools):
        distances = [100 * float(data.nearest_natural(constructs[k:k + 1],
                                                      encoded[mask])[0][0]) / length
                     for k in range(len(series))]
        ax_b.bar(group + (i - 0.5) * bar_w, distances, bar_w, color=colour,
                 edgecolor="white", linewidth=1.0, zorder=3, label=label)
    # the two family clouds, as reference lines
    for pool, colour, label in ((trypsins, style.C_TRYPSIN, "nearest natural trypsin"),
                                (chymos, style.C_CHYMO, "nearest natural chymotrypsin")):
        values = [100 * float(data.nearest_natural(constructs[k:k + 1], pool)[0][0]) / length
                  for k in range(len(series))]
        ax_b.plot(group, values, marker="o", ms=8, lw=2.2, color=colour,
                  markeredgecolor="white", markeredgewidth=1.0, zorder=5, label=label)
    ax_b.set_xticks(group, [str(i + 1) for i in group], fontsize=13)
    ax_b.set_xlabel("construct  (1 = wild type, 2–4 = the 1992 series)")
    ax_b.set_ylabel("Hamming distance  (% of $L$)")
    ax_b.set_title("the constructs are NOT unfamiliar to the model", fontsize=14,
                   color=style.MUTED, pad=10)
    # Headroom above the chymotrypsin line so the legend does not sit on it.
    ax_b.set_ylim(0, 1.32 * 100 * float(
        data.nearest_natural(constructs[:1], chymos)[0][0]) / length)
    ax_b.legend(fontsize=11, loc="upper left", labelspacing=0.45, ncol=2,
                columnspacing=1.4)
    style.tidy(ax_b, grid="y")

    # --- (c) the explanation ---------------------------------------------------
    closed, p_s, labels = [], [], []
    for index, (label, _) in enumerate(series):
        gap = float(data.nearest_natural(constructs[index:index + 1], chymos)[0][0])
        closed.append(100 * (base_gap - gap) / base_gap)
        p_s.append(float(construct_P[index, s_state]))
        labels.append(str(index + 1))
    ax_c.plot(closed, p_s, ls=":", lw=1.8, color=style.C_1992, zorder=4)
    for x, y, text in zip(closed, p_s, labels):
        ax_c.scatter([x], [y], s=165, facecolor="white", edgecolor=style.INK,
                     linewidths=1.6, zorder=5)
        ax_c.annotate(text, xy=(x, y), ha="center", va="center", fontsize=10.5,
                      color=style.INK, zorder=6)

    table = selection.select_all()
    rng = np.random.default_rng(0)
    for _, row in table.iterrows():
        if bool(row["placeholder"]):
            continue
        run = next(r for r in data.discover_runs(row["condition"])
                   if r.beta == row["beta"] and r.lam == row["lam"])
        sequences = data.run_sequences(str(run.path))
        if len(sequences) > MAX_DESIGN:
            sequences = sequences[rng.choice(len(sequences), MAX_DESIGN, replace=False)]
        gap = float(data.nearest_natural(sequences, chymos)[0].mean())
        spec = style.group(row["condition"])
        ax_c.scatter([100 * (base_gap - gap) / base_gap], [float(row["P_S"])],
                     s=130, marker=spec["marker"], color=spec["color"],
                     edgecolor="white", linewidths=1.0, zorder=5)
    ax_c.axvline(0, color=style.RULE, lw=1.6, ls=(0, (5, 3)), zorder=1)
    ax_c.set_xlabel("% of the wild-type → chymotrypsin gap closed\n"
                    f"(the gap is {int(base_gap)} residues)")
    ax_c.set_ylabel(style.P_S)
    ax_c.set_xlim(-6, 20)
    ax_c.set_title("high $P(\\mathrm{S})$ without moving toward the family",
                   fontsize=14, color=style.MUTED, pad=10)
    handles = [Line2D([], [], marker="o", ls=":", color=style.C_1992,
                      markerfacecolor="white", markeredgecolor=style.INK,
                      markersize=10, label="1992 series (1–4)")]
    seen = set()
    for _, row in table.iterrows():
        condition = row["condition"]
        if bool(row["placeholder"]) or condition in seen:
            continue
        seen.add(condition)
        spec = style.group(condition)
        handles.append(Line2D([], [], marker=spec["marker"], ls="none",
                              color=spec["color"], markersize=10, label=condition))
    ax_c.legend(handles=handles, fontsize=10.5, loc="lower right", labelspacing=0.45)
    style.tidy(ax_c)

    for letter, ax in zip("abc", (ax_a, ax_b, ax_c)):
        style.panel_letter(ax, letter, dx=-0.115, dy=1.13)

    fig.suptitle("The literature constructs score low because they never leave "
                 "the trypsins — and neither do ours", y=0.965)
    style.caveat(
        fig,
        f"The wild type sits {int(base_gap)} residues ({100 * base_gap / length:.0f}% of $L$) "
        "from the nearest natural chymotrypsin. The 1992 constructs close none of that gap; "
        "our designs close at most ~5%.\n"
        "So $P(\\mathrm{S}\\mid\\sigma_{\\setminus i^*})$ is high because the LOCAL context at "
        "$i^*$ has been rewired, not because the protein has become a chymotrypsin — which is "
        "what every $P(\\mathrm{S})$ in this deck does and does not show.",
        y=0.010)
    style.save(fig, "si07_construct_context")

    print(f"{'construct':<28}{'P(S)':>8}{'->train':>10}{'->test':>9}{'gap closed':>12}")
    for index, (label, _) in enumerate(series):
        d_tr = 100 * float(data.nearest_natural(constructs[index:index + 1],
                                                encoded[train])[0][0]) / length
        d_te = 100 * float(data.nearest_natural(constructs[index:index + 1],
                                                encoded[held_out])[0][0]) / length
        print(f"{label:<28}{p_s[index]:>8.4f}{d_tr:>9.1f}%{d_te:>8.1f}%{closed[index]:>11.1f}%")


if __name__ == "__main__":
    main()
