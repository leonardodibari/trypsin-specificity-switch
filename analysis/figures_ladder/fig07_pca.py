"""F7 -- where the designs sit relative to the natural family, in sequence space.

Two principal components of the NATURAL alignment in one-hot space, with
everything else projected into that basis. Fitting on the naturals and projecting
the designs (rather than fitting on everything together) is what makes the axes
mean "natural family variation" -- otherwise a tight cluster of 1000 near-identical
designs would drag the components towards itself and the plot would describe the
sampler instead of the family.

(a) the natural families, the wild type, and the reconstructed 1992 series
(b) the natural families as background, with the selected designs on top

Implementation note: the one-hot matrix is 89,439 x 5,565 but has only 265
non-zeros per row, so it is held sparse. Centring would destroy that sparsity,
so the centred matrix is never materialised -- it is applied as a LinearOperator,
``(X - 1 mu^T) v = Xv - (mu.v) 1``, and the SVD is taken of that. Densifying
would cost ~2 GB for no benefit.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, svds

import config
import data
import selection
import style

N_COMPONENTS = 2
MAX_DESIGN_POINTS = 400   # per run, to keep the PDF and the eye from drowning


def _one_hot(encoded: np.ndarray, q: int) -> csr_matrix:
    n, length = encoded.shape
    indices = (np.arange(length)[None, :] * q + encoded).ravel()
    indptr = np.arange(n + 1) * length
    return csr_matrix((np.ones(n * length, dtype=np.float32), indices, indptr),
                      shape=(n, length * q))


def _fit_pca(X: csr_matrix):
    """Top components of the centred matrix, without densifying it."""
    mean = np.asarray(X.mean(axis=0)).ravel()
    n = X.shape[0]

    def matvec(v):
        return X @ v - float(mean @ v)

    def rmatvec(u):
        return X.T @ u - mean * float(u.sum())

    operator = LinearOperator(X.shape, matvec=matvec, rmatvec=rmatvec,
                              dtype=np.float32)
    # svds returns ascending singular values; reverse for PC1 first.
    _, singular, vt = svds(operator, k=N_COMPONENTS, random_state=0)
    order = np.argsort(-singular)
    components = vt[order]
    explained = singular[order] ** 2 / (n - 1)
    # TOTAL variance of the centred data, so the percentages are shares of all
    # the variance and not merely of the two components we kept. X is 0/1, so
    # sum(X^2) == nnz == n * L.
    total = (X.shape[0] * (X.nnz / X.shape[0]) - n * float(mean @ mean)) / (n - 1)
    return mean, components, explained, float(total)


def _project(encoded: np.ndarray, mean, components, q: int) -> np.ndarray:
    X = _one_hot(np.atleast_2d(encoded), q)
    return np.asarray(X @ components.T) - (mean @ components.T)[None, :]


def _pca_basis():
    """Fit (or reload) the natural-alignment PCA basis."""
    cache = config.cache_path("pca_basis")
    _, encoded = data.alignment()
    q = data.model().q
    if cache.is_file():
        blob = np.load(cache)
        return (blob["mean"], blob["components"], blob["explained"],
                blob["natural"], float(blob["total"]), q)
    mean, components, explained, total = _fit_pca(_one_hot(encoded, q))
    natural = _project(encoded, mean, components, q)
    data._save_npz(cache, mean=mean, components=components, explained=explained,
                   natural=natural, total=np.array(total))
    return mean, components, explained, natural, total, q


def _joint_axes(fig, gridspec):
    """A scatter axis with PC1/PC2 marginal histograms."""
    inner = gridspec.subgridspec(2, 2, width_ratios=[4.2, 1.0],
                                 height_ratios=[1.0, 4.2], wspace=0.06, hspace=0.06)
    main = fig.add_subplot(inner[1, 0])
    top = fig.add_subplot(inner[0, 0], sharex=main)
    right = fig.add_subplot(inner[1, 1], sharey=main)
    for marginal, axis in ((top, "x"), (right, "y")):
        marginal.spines[["top", "right"]].set_visible(False)
        marginal.tick_params(labelbottom=False, labelleft=False, length=0)
        marginal.grid(False)
        if axis == "x":
            marginal.spines["left"].set_visible(False)
            marginal.set_yticks([])
        else:
            marginal.spines["bottom"].set_visible(False)
            marginal.set_xticks([])
    style.tidy(main)
    return main, top, right


def _marginals(top, right, values, colour, weight=True, **kwargs):
    for axis, index in ((top, 0), (right, 1)):
        series = values[:, index]
        weights = np.full(len(series), 1 / max(len(series), 1)) if weight else None
        axis.hist(series, bins=60, weights=weights, color=colour,
                  orientation="vertical" if index == 0 else "horizontal", **kwargs)


def main() -> None:
    style.apply()
    mean, components, explained, natural, total, q = _pca_basis()
    classes = data.natural_classes()

    fig = plt.figure(figsize=(18.0, 10.4))
    outer = fig.add_gridspec(1, 2, left=0.055, right=0.985, top=0.855,
                             bottom=0.215, wspace=0.17)
    mainA, topA, rightA = _joint_axes(fig, outer[0, 0])
    mainB, topB, rightB = _joint_axes(fig, outer[0, 1])

    # --- (a) the natural families, wild type, 1992 series ----------------------
    for key, colour, label in (
        ("other", style.C_OTHER, "other naturals"),
        ("trypsin", style.C_TRYPSIN, "trypsins (D at $i^*$)"),
        ("chymotrypsin", style.C_CHYMO, "chymotrypsins (S at $i^*$)"),
    ):
        points = natural[classes[key]]
        mainA.scatter(points[:, 0], points[:, 1], s=2.0, c=colour, alpha=0.18,
                      linewidths=0, rasterized=True, label=label)
        _marginals(topA, rightA, points, colour, alpha=0.62)

    series = data.construct_series()
    series_points = _project(np.stack([s for _, s in series]), mean, components, q)
    wt_point = _project(data.wild_type(), mean, components, q)[0]
    chymo_points = np.vstack([
        _project(data.chymotrypsin_rat(), mean, components, q)[0],
        _project(data.chymotrypsin_bovine(), mean, components, q)[0],
    ])

    def draw_markers(ax, *, numbers: bool) -> None:
        ax.plot(series_points[:, 0], series_points[:, 1], color=style.C_1992,
                ls=":", lw=1.8, zorder=5)
        for index, point in enumerate(series_points):
            ax.scatter(*point, s=170 if numbers else 90, facecolor="white",
                       edgecolor=style.INK, linewidths=1.6, zorder=6)
            if numbers:
                ax.annotate(str(index + 1), xy=point, ha="center", va="center",
                            fontsize=11, color=style.INK, zorder=7)
        ax.scatter(*wt_point, s=240, marker="*", color=style.C_WT, zorder=8)
        ax.scatter(chymo_points[:, 0], chymo_points[:, 1], s=150, marker="P",
                   color=style.C_CHYMO_REF, zorder=8)

    draw_markers(mainA, numbers=False)

    # The wild type and all four constructs sit inside a region a few percent of
    # the plot wide, so numbering them in place is unreadable. Zoom it.
    pad_x, pad_y = 0.45, 0.22
    xs = np.concatenate([series_points[:, 0], [wt_point[0]]])
    ys = np.concatenate([series_points[:, 1], [wt_point[1]]])
    zoom = mainA.inset_axes([0.035, 0.62, 0.40, 0.33])
    draw_markers(zoom, numbers=True)
    zoom.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
    zoom.set_ylim(ys.min() - pad_y, ys.max() + pad_y)
    zoom.set_title("wild type and the 1992 series", fontsize=11, color=style.MUTED,
                   pad=4)
    zoom.tick_params(labelsize=9.5, colors=style.MUTED)
    zoom.grid(color=style.GRID, lw=0.7)
    zoom.set_axisbelow(True)
    zoom.set_facecolor("white")
    for spine in zoom.spines.values():
        spine.set_color(style.FAINT)
    mainA.indicate_inset_zoom(zoom, edgecolor=style.RULE, lw=1.0, alpha=1.0)

    # Build every handle explicitly at its own size. A legend-wide `markerscale`
    # would also scale the star, the plus and the ringed circle, which are
    # already sized for the plot, and they overflowed the legend box.
    handles_a = [
        Line2D([], [], marker="o", ls="none", color=style.C_OTHER, markersize=8,
               label="other naturals"),
        Line2D([], [], marker="o", ls="none", color=style.C_TRYPSIN, markersize=8,
               label="trypsins (D at $i^*$)"),
        Line2D([], [], marker="o", ls="none", color=style.C_CHYMO, markersize=8,
               label="chymotrypsins (S at $i^*$)"),
        Line2D([], [], marker="*", ls="none", color=style.C_WT, markersize=14,
               label="wild-type trypsin"),
        Line2D([], [], marker="P", ls="none", color=style.C_CHYMO_REF,
               markersize=10, label="chymotrypsin references"),
        Line2D([], [], marker="o", ls=":", color=style.C_1992,
               markerfacecolor="white", markeredgecolor=style.INK, markersize=9,
               label="1992 series (reconstructed)"),
    ]
    mainA.legend(handles=handles_a, loc="lower right", fontsize=11.5,
                 handletextpad=0.6, borderpad=0.45, labelspacing=0.38,
                 facecolor="white", framealpha=0.93, frameon=True,
                 edgecolor=style.FAINT)

    # --- (b) the selected designs over the natural background -------------------
    mainB.scatter(natural[:, 0], natural[:, 1], s=2.0, c=style.C_OTHER,
                  alpha=0.16, linewidths=0, rasterized=True, label="naturals")
    _marginals(topB, rightB, natural, style.C_OTHER, alpha=0.55)

    table = selection.select_all()
    rng = np.random.default_rng(0)
    handles = []
    for _, row in table.iterrows():
        condition = row["condition"]
        spec = style.group(condition)
        rank = int(row["rank_in_condition"])
        if bool(row["placeholder"]):
            continue
        run = next(r for r in data.discover_runs(condition)
                   if r.beta == row["beta"] and r.lam == row["lam"])
        sequences = data.run_sequences(str(run.path))
        if len(sequences) > MAX_DESIGN_POINTS:
            sequences = sequences[rng.choice(len(sequences), MAX_DESIGN_POINTS,
                                             replace=False)]
        points = _project(sequences, mean, components, q)
        filled = rank == 0
        mainB.scatter(points[:, 0], points[:, 1], s=26, marker=spec["marker"],
                      facecolor=spec["color"] if filled else "none",
                      edgecolor=spec["color"], linewidths=1.1, alpha=0.75,
                      rasterized=True, zorder=4)
        _marginals(topB, rightB, points, spec["color"], alpha=0.0,
                   histtype="step", lw=1.6)
        handles.append(Line2D([], [], marker=spec["marker"], ls="none",
                              markerfacecolor=spec["color"] if filled else "none",
                              markeredgecolor=spec["color"], markersize=9,
                              label=f"{condition}  β={row['beta']:g}, λ={row['lam']:g}"))
    missing = [c for c in config.CONDITIONS if not data.discover_runs(c)]
    for condition in missing:
        handles.append(Line2D([], [], marker=style.group(condition)["marker"],
                              ls="none", markerfacecolor="none",
                              markeredgecolor=style.FAINT, markersize=9,
                              label=f"{condition} — not sampled"))
    mainB.scatter(*wt_point, s=240, marker="*", color=style.C_WT, zorder=8)
    handles.append(Line2D([], [], marker="*", ls="none", color=style.C_WT,
                          markersize=15, label="wild type"))
    # The designs collapse into a tight blob near the wild type, so an in-axes
    # legend would sit on top of the only interesting region. Put it under the
    # figure instead.
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.74, 0.035),
               ncol=2, fontsize=11.5, handletextpad=0.5, labelspacing=0.4,
               columnspacing=1.6)

    for main_ax in (mainA, mainB):
        main_ax.set_xlabel(f"PC1  ({100 * explained[0] / total:.1f}% of total variance)")
        main_ax.set_ylabel(f"PC2  ({100 * explained[1] / total:.1f}%)")
    topA.set_title("naturals, wild type and the 1992 series", fontsize=15, pad=10)
    topB.set_title("naturals and the selected designs", fontsize=15, pad=10)
    # Figure coordinates: the marginal axes sit too high for an axes-relative
    # offset to land the letter anywhere sensible.
    for letter, x in zip("ab", (0.012, 0.512)):
        fig.text(x, 0.935, letter, fontsize=21, fontweight="bold", va="top",
                 ha="left", color=style.INK)

    fig.suptitle("Designs and naturals in the same sequence-space projection",
                 y=0.978)
    style.caveat(fig, "PCA fitted on the natural alignment only; everything else "
                      "is projected into that basis. The 1992 series is "
                      "reconstructed from column positions, not the published "
                      "sequences.", y=0.002)
    style.save(fig, "fig07_pca")

    print(f"  PC1 {100 * explained[0] / total:.2f}%   "
          f"PC2 {100 * explained[1] / total:.2f}%   of TOTAL variance")
    print(f"  wild type at PC1={wt_point[0]:.2f}, PC2={wt_point[1]:.2f}")
    for (name, _), point in zip(series, series_points):
        print(f"  {name:<30} PC1={point[0]:7.2f}  PC2={point[1]:7.2f}")


if __name__ == "__main__":
    main()
