#!/usr/bin/env python
"""4x3 overview of the MCMC trajectories: one column per lambda, four rows.

Rows
    1. ``P(S | critic site)`` vs accepted step, every chain plus the median
       ribbon, with the acceptance threshold as a dashed rule.
    2. ``E_DCA`` vs accepted step.
    3. ``E_DCA`` against ``P(S | critic site)``, one point per recorded step,
       with the wild type and the chimeric target marked.
    4. The effective energy ``E_DCA - plot_lambda * log P(S)``, evaluated at one
       *chosen* lambda for every column, so chains sampled at different lambdas
       are comparable on a single scale.

Example::

    python scripts/plot_trajectories.py \\
        --output-dir results/ \\
        --lambdas 1.0 5.0 10.0 \\
        --plot-lambda 2.0 \\
        --mapping data/mapping_ShoMSA_pdb3TGI.csv \\
        --wt data/wt_trypsin.fasta \\
        --model data/adabmdca_model \\
        --save fig.png
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import _SRC  # noqa: F401,E402  (puts src/ on sys.path)

from encoding import Alphabet, read_single_sequence  # noqa: E402
from mcmc import lam_dirname  # noqa: E402
from potts import PottsModel  # noqa: E402
from sites import (  # noqa: E402
    CRITIC_3TGI_POSITION,
    load_mapping,
    make_chimera,
    parse_column_spec,
    resolve_sites,
)

# Colours: chains share one sequential hue (they are exchangeable replicates,
# not distinct categories); the accent marks get their own hue *and* their own
# marker shape, so identity never rests on colour alone.
RUN_CMAP = plt.get_cmap("Blues")
RUN_RANGE = (0.40, 0.92)
MEDIAN_COLOR = "#eb6834"
THRESHOLD_COLOR = "#52514e"
WT_COLOR = "#008300"
TARGET_COLOR = "#e34948"
GRID_KWARGS = dict(color="#d8d7d2", linewidth=0.6, alpha=0.9)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="Directory holding the lam*/ run folders")
    parser.add_argument(
        "--lambdas", type=float, nargs="+", required=True, help="The lambdas to show, one per column"
    )
    parser.add_argument(
        "--plot-lambda",
        type=float,
        default=None,
        help="Lambda used for the effective energy in row 4 (default: the first --lambdas value)",
    )
    parser.add_argument("--mapping", default=None, help="CSV mapping 3TGI residues to MSA columns")
    parser.add_argument(
        "--critic-col", type=int, default=None, help="MSA column of the critic site; replaces --mapping"
    )
    parser.add_argument("--wt", required=True, help="FASTA with the aligned wild-type sequence")
    parser.add_argument("--model", required=True, help="adabmDCA parameter file or model directory")
    parser.add_argument("--save", default="fig.png", help="Output image (default: fig.png)")
    parser.add_argument(
        "--target",
        default=None,
        help="FASTA of the chimeric target; marks it in row 3 (red diamond)",
    )
    parser.add_argument(
        "--target-source",
        default=None,
        help="Aligned chymotrypsin FASTA; the chimera is built from it at --allowed-sites",
    )
    parser.add_argument(
        "--allowed-sites",
        default="sho",
        help="3TGI positions defining the chimera when --target-source is used (default: sho)",
    )
    parser.add_argument(
        "--allowed-cols",
        default=None,
        help="MSA columns defining the chimera, 0-based (e.g. '180-220'); replaces --allowed-sites",
    )
    parser.add_argument("--threshold", type=float, default=0.80, help="Threshold rule in row 1")
    parser.add_argument("--beta", type=float, default=1.0, help="Inverse temperature for P(S)")
    parser.add_argument("--critic-site", default=CRITIC_3TGI_POSITION, help="3TGI specificity site")
    parser.add_argument("--target-residue", default="S", help="Residue tracked at the critic site")
    parser.add_argument("--alphabet", default="protein", help="adabmDCA alphabet name or explicit tokens")
    parser.add_argument(
        "--cache-model", action="store_true", help="Reuse/write the <params>.cache.pt tensor dump"
    )
    parser.add_argument("--mapping-base", type=int, default=0, choices=(0, 1), help="msa_col index base")
    parser.add_argument("--device", default="cpu", help="torch device")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--dpi", type=int, default=200, help="Figure resolution (default: 200)")
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=(13.5, 13.0), help="Figure size in inches"
    )
    return parser


# --------------------------------------------------------------------- loading


def load_runs(output_dir: Path, lam: float) -> List[Tuple[str, pd.DataFrame]]:
    """Every chain recorded for ``lam``, as ``(label, trajectory)`` pairs."""
    lam_dir = _find_lam_dir(output_dir, lam)
    if lam_dir is None:
        return []
    runs: List[Tuple[str, pd.DataFrame]] = []
    for run_dir in sorted(lam_dir.glob("run*"), key=_run_index):
        path = run_dir / "trajectory.csv"
        if path.is_file():
            runs.append((run_dir.name, _read_trajectory(path)))
    if not runs and (lam_dir / "trajectory.csv").is_file():
        runs.append(("run0", _read_trajectory(lam_dir / "trajectory.csv")))
    return runs


def _find_lam_dir(output_dir: Path, lam: float) -> Path | None:
    exact = output_dir / lam_dirname(lam)
    if exact.is_dir():
        return exact
    for candidate in sorted(output_dir.glob("lam*")):
        if not candidate.is_dir():
            continue
        try:
            value = float(candidate.name[3:])
        except ValueError:
            continue
        if math.isclose(value, lam, rel_tol=1e-9, abs_tol=1e-12):
            return candidate
    return None


def _run_index(path: Path) -> int:
    digits = "".join(c for c in path.name if c.isdigit())
    return int(digits) if digits else 0


def _read_trajectory(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["P_S"] = np.exp(frame["logP_S"].to_numpy())
    return frame


def reference_points(args) -> Dict[str, Tuple[float, float]]:
    """``{'wild type': (P_S, E_DCA), ...}`` for the wild type and the target."""
    alphabet = Alphabet.from_name(args.alphabet)
    model = PottsModel.load(
        args.model, alphabet=alphabet, device=args.device, dtype=args.dtype, cache=args.cache_model
    )
    mapping = load_mapping(args.mapping, base=args.mapping_base) if args.mapping else None
    if args.critic_col is None and mapping is None:
        raise SystemExit("Give either --mapping (to resolve 3TGI 189) or --critic-col")
    critic_col = args.critic_col if args.critic_col is not None else mapping.column(args.critic_site)
    state = alphabet.index(args.target_residue)

    def point(sequence: np.ndarray) -> Tuple[float, float]:
        log_p = model.log_conditional(critic_col, sequence, state, beta=args.beta)
        return math.exp(log_p), model.energy(sequence)

    _, wt_string = read_single_sequence(args.wt)
    wt = alphabet.encode(wt_string)
    points = {"wild type": point(wt)}

    target = None
    if args.target:
        _, target_string = read_single_sequence(args.target)
        target = alphabet.encode(target_string)
    elif args.target_source:
        _, source_string = read_single_sequence(args.target_source)
        if args.allowed_cols is not None:
            columns = parse_column_spec(args.allowed_cols, length=model.L)
        elif mapping is not None:
            _, columns, _ = resolve_sites(args.allowed_sites, mapping)
        else:
            raise SystemExit("--target-source needs either --mapping or --allowed-cols")
        target, _ = make_chimera(wt, alphabet.encode(source_string), columns, alphabet)
    if target is not None:
        if target.shape[0] != wt.shape[0]:
            raise SystemExit("The target sequence is not aligned to the wild type")
        points["chimeric target"] = point(target)
    return points


# --------------------------------------------------------------------- drawing


def median_band(
    runs: Sequence[Tuple[str, pd.DataFrame]], column: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median and inter-quartile band of ``column`` on a common step grid.

    Chains stop at different steps; past the end of a chain it no longer counts
    toward the quantiles instead of being held at its last value.
    """
    grid = np.unique(np.concatenate([frame["step"].to_numpy() for _, frame in runs]))
    stacked = np.full((len(runs), grid.size), np.nan)
    for index, (_, frame) in enumerate(runs):
        steps = frame["step"].to_numpy()
        values = frame[column].to_numpy()
        inside = grid <= steps.max()
        stacked[index, inside] = np.interp(grid[inside], steps, values)
    alive = np.sum(~np.isnan(stacked), axis=0)
    # Past the point where fewer than half the chains are still running, the
    # "median" would be a single chain wearing the median's weight: drop it.
    enough = alive >= max(2, math.ceil(len(runs) / 2)) if len(runs) > 1 else alive >= 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns are expected
        median = np.nanmedian(stacked, axis=0)
        low = np.nanpercentile(stacked, 25, axis=0)
        high = np.nanpercentile(stacked, 75, axis=0)
    median[~enough] = np.nan
    low[~enough] = np.nan
    high[~enough] = np.nan
    return grid, median, low, high


def run_colors(n: int) -> List[tuple]:
    positions = np.linspace(*RUN_RANGE, n) if n > 1 else np.array([RUN_RANGE[1]])
    return [RUN_CMAP(p) for p in positions]


def crossing_step(frame: pd.DataFrame, threshold: float) -> int | None:
    crossed = frame.index[frame["P_S"].to_numpy() >= threshold]
    return int(frame["step"].iloc[crossed[0]]) if len(crossed) else None


def plot_series(
    axis,
    runs: Sequence[Tuple[str, pd.DataFrame]],
    column: str,
    colors: Sequence[tuple],
    threshold: float,
    mark_crossings: bool,
    values_fn=None,
) -> None:
    for (label, frame), color in zip(runs, colors):
        values = frame[column].to_numpy() if values_fn is None else values_fn(frame)
        axis.plot(frame["step"], values, color=color, linewidth=0.9, alpha=0.75, zorder=2)
        if mark_crossings:
            step = crossing_step(frame, threshold)
            if step is not None:
                position = int(np.searchsorted(frame["step"].to_numpy(), step))
                axis.plot(
                    step,
                    values[position],
                    marker="*",
                    markersize=13,
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    zorder=5,
                )
    if values_fn is not None:
        frames = [(label, frame.assign(_plot=values_fn(frame))) for label, frame in runs]
        grid, median, low, high = median_band(frames, "_plot")
    else:
        grid, median, low, high = median_band(runs, column)
    axis.fill_between(grid, low, high, color=MEDIAN_COLOR, alpha=0.16, linewidth=0, zorder=3)
    axis.plot(grid, median, color=MEDIAN_COLOR, linewidth=2.0, zorder=4)


def style(axis) -> None:
    axis.grid(True, **GRID_KWARGS)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#a9a8a2")
    axis.tick_params(colors="#52514e", labelsize=9)


def main() -> None:
    args = build_parser().parse_args()
    plot_lambda = args.plot_lambda if args.plot_lambda is not None else args.lambdas[0]
    output_dir = Path(args.output_dir)

    per_lambda = {lam: load_runs(output_dir, lam) for lam in args.lambdas}
    empty = [lam for lam, runs in per_lambda.items() if not runs]
    if empty:
        print(f"WARNING: no trajectory found for lambda {empty} under {output_dir}", file=sys.stderr)
    if all(not runs for runs in per_lambda.values()):
        raise SystemExit(f"No trajectory.csv found under {output_dir}")

    points = reference_points(args)

    n_columns = len(args.lambdas)
    figure, axes = plt.subplots(
        4, n_columns, figsize=tuple(args.figsize), squeeze=False, constrained_layout=True
    )
    figure.patch.set_facecolor("#fcfcfb")

    row_labels = [
        f"P({args.target_residue} | critic site)",
        r"$E_{\mathrm{DCA}}$",
        r"$E_{\mathrm{DCA}}$",
        rf"$E_{{\mathrm{{DCA}}}} - {plot_lambda:g}\,\log P$",
    ]

    for column_index, lam in enumerate(args.lambdas):
        runs = per_lambda[lam]
        colors = run_colors(max(len(runs), 1))
        column_axes = [axes[row][column_index] for row in range(4)]
        for axis in column_axes:
            style(axis)
            axis.set_facecolor("#fcfcfb")
        axes[0][column_index].set_title(rf"$\lambda = {lam:g}$", fontsize=12, pad=8)

        if not runs:
            for axis in column_axes:
                axis.text(0.5, 0.5, "no data", ha="center", va="center",
                          transform=axis.transAxes, color="#8a8984", fontsize=10)
            continue

        # Row 1 -- P(S | critic site)
        axis = column_axes[0]
        plot_series(axis, runs, "P_S", colors, args.threshold, mark_crossings=True)
        axis.axhline(args.threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.1, zorder=1)
        axis.set_ylim(0.0, 1.02)

        # Row 2 -- DCA energy
        plot_series(column_axes[1], runs, "E_DCA", colors, args.threshold, mark_crossings=True)

        # Row 3 -- energy against specificity
        axis = column_axes[2]
        for (label, frame), color in zip(runs, colors):
            axis.scatter(
                frame["P_S"], frame["E_DCA"], s=16, color=color, alpha=0.75,
                linewidths=0.5, edgecolors="white", zorder=2, label=label,
            )
        for name, (p_value, energy) in points.items():
            marker, color = ("o", WT_COLOR) if name == "wild type" else ("D", TARGET_COLOR)
            axis.scatter(
                p_value, energy, s=95, marker=marker, color=color,
                edgecolors="white", linewidths=1.2, zorder=6, label=name,
            )
        axis.axvline(args.threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.1, zorder=1)
        axis.set_xlabel(f"P({args.target_residue} | critic site)", fontsize=10)
        axis.set_xlim(-0.02, 1.02)

        # Row 4 -- effective energy at the chosen lambda
        plot_series(
            column_axes[3], runs, "E_DCA", colors, args.threshold, mark_crossings=True,
            values_fn=lambda frame: frame["E_DCA"].to_numpy() - plot_lambda * frame["logP_S"].to_numpy(),
        )
        column_axes[3].set_xlabel("accepted step", fontsize=10)

        for row in (0, 1):
            column_axes[row].set_xlabel("accepted step", fontsize=10)
        if column_index == 0:
            for row, text in enumerate(row_labels):
                axes[row][0].set_ylabel(text, fontsize=11)

    _add_legend(figure, axes, per_lambda, points, args)
    figure.suptitle(
        "Trypsin -> chymotrypsin specificity switch: biased MCMC trajectories",
        fontsize=14,
        y=1.015,
    )

    save_path = Path(args.save)
    if save_path.parent != Path(""):
        save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=args.dpi, bbox_inches="tight", facecolor=figure.get_facecolor())
    print(f"figure written to {save_path}")


def _add_legend(figure, axes, per_lambda, points, args) -> None:
    from matplotlib.lines import Line2D

    n_runs = max((len(runs) for runs in per_lambda.values()), default=0)
    colors = run_colors(max(n_runs, 1))
    handles = [
        Line2D([], [], color=color, linewidth=1.4, label=f"run {index}")
        for index, color in enumerate(colors)
    ]
    handles.append(Line2D([], [], color=MEDIAN_COLOR, linewidth=2.0, label="median (IQR band)"))
    handles.append(
        Line2D([], [], color=THRESHOLD_COLOR, linestyle="--", linewidth=1.1,
               label=f"threshold = {args.threshold:g}")
    )
    handles.append(
        Line2D([], [], color="#52514e", marker="*", markersize=12, linestyle="none",
               label="first crossing")
    )
    for name in points:
        marker, color = ("o", WT_COLOR) if name == "wild type" else ("D", TARGET_COLOR)
        handles.append(
            Line2D([], [], color=color, marker=marker, markersize=9, linestyle="none", label=name)
        )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 6),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.035),
    )


if __name__ == "__main__":
    main()
