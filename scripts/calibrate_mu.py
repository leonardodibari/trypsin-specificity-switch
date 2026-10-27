#!/usr/bin/env python
"""Calibrate mu, the weight of the context entropy in the sample quality.

The sample quality of a sequence is

    Q_mu(seq) = E_DCA(seq) - mu * CDE(seq)

with CDE the mean, over sites, of the Shannon entropy of the model's
conditional at that site given the rest of the sequence. A natural alignment is
a set of sequences the model should rate as uniformly good, so mu is chosen to
make Q as flat as possible across the natural family: it is the minimiser of
``Var(Q_mu)`` over the alignment, which is ``Cov(E, CDE) / Var(CDE)`` -- the
slope of the energy against the context entropy, i.e. the exchange rate between
the two that the family itself defines.

Example::

    python scripts/calibrate_mu.py \\
        --natural ../natural_msa.fasta \\
        --model ../bmDCA_params.dat \\
        --output calibration/ \\
        --cache-model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from _common import _SRC  # noqa: F401,E402  (puts src/ on sys.path)

from encoding import Alphabet, read_sequences, read_single_sequence  # noqa: E402
from observables import (  # noqa: E402
    context_dependent_entropy,
    critic_probability,
    energies,
    hamming_distance,
    optimal_mu,
    sample_quality,
)
from potts import PottsModel  # noqa: E402

NATURAL_COLOR = "#52514e"
ACCENT_COLOR = "#2a78d6"
OPTIMUM_COLOR = "#eb6834"
SURFACE = "#fcfcfb"
GRID_KWARGS = dict(color="#d8d7d2", linewidth=0.6, alpha=0.9)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--natural", required=True, help="FASTA of the natural alignment (aligned to the model)")
    parser.add_argument("--model", required=True, help="adabmDCA parameter file or model directory")
    parser.add_argument("--output", default="calibration", help="Output directory (default: calibration)")
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=5000,
        help="Random subsample of the alignment to score (0 = all; default: 5000)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed of that subsample")
    parser.add_argument("--beta", type=float, default=1.0, help="Inverse temperature")
    parser.add_argument("--alphabet", default="protein", help="adabmDCA alphabet name or explicit tokens")
    parser.add_argument("--device", default="cpu", help="torch device")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--cache-model", action="store_true", help="Reuse/write <params>.cache.pt")
    parser.add_argument("--chunk", type=int, default=256, help="Sequences scored per chunk")
    parser.add_argument(
        "--wt",
        default=None,
        help="Aligned wild type; adds the natural sequences' Hamming distance to it",
    )
    parser.add_argument(
        "--critic-col",
        type=int,
        default=None,
        help="Critic MSA column; adds P(target | critic) for the natural sequences",
    )
    parser.add_argument("--target-residue", default="S", help="Residue tracked at the critic site")
    parser.add_argument("--torch-threads", type=int, default=0, help="Intra-op torch threads (0 = default)")
    parser.add_argument("--save", default=None, help="Figure path (default: <output>/mu_calibration.png)")
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    alphabet = Alphabet.from_name(args.alphabet)
    model = PottsModel.load(
        args.model,
        alphabet=alphabet,
        device=args.device,
        dtype=args.dtype,
        cache=args.cache_model,
        verbose=True,
    )

    records = read_sequences(args.natural)
    names = [name for name, _ in records]
    lengths = {len(sequence) for _, sequence in records}
    if lengths != {model.L}:
        raise SystemExit(
            f"The alignment has sequence lengths {sorted(lengths)} but the model has L={model.L}"
        )
    if args.max_sequences and len(records) > args.max_sequences:
        index = np.random.default_rng(args.seed).choice(len(records), args.max_sequences, replace=False)
        index.sort()
        records = [records[i] for i in index]
        names = [names[i] for i in index]
        print(f"scoring a random subsample of {len(records)} of the alignment's sequences")
    else:
        print(f"scoring all {len(records)} sequences of the alignment")

    encoded = np.stack([alphabet.encode(sequence) for _, sequence in records])
    energy = energies(model, encoded, chunk=args.chunk)
    cde = context_dependent_entropy(model, encoded, beta=args.beta, chunk=args.chunk)

    mu = optimal_mu(energy, cde)
    quality = sample_quality(energy, cde, mu)
    baseline = float(np.var(energy))
    print(
        f"\nmu* = {mu:.4f}\n"
        f"  Var(E)            = {baseline:.3f}   (sd {np.sqrt(baseline):.3f})\n"
        f"  Var(E - mu* CDE)  = {quality.var():.3f}   (sd {quality.std():.3f})\n"
        f"  variance removed  = {100 * (1 - quality.var() / baseline):.1f}%\n"
        f"  corr(E, CDE)      = {np.corrcoef(energy, cde)[0, 1]:.3f}"
    )

    columns = {"name": names, "E_DCA": energy, "CDE": cde, "Q": quality}
    if args.wt:
        _, wt_string = read_single_sequence(args.wt)
        columns["hamming_to_wt"] = hamming_distance(encoded, alphabet.encode(wt_string))
    if args.critic_col is not None:
        columns["P_S"] = critic_probability(
            model, encoded, args.critic_col, alphabet.index(args.target_residue),
            beta=args.beta, chunk=args.chunk,
        )
    frame = pd.DataFrame(columns)
    frame.to_csv(output / "natural_observables.csv", index=False)
    payload = {
        "mu": mu,
        "n_sequences": int(len(energy)),
        "alignment": str(Path(args.natural).resolve()),
        "model": str(model.source),
        "beta": args.beta,
        "var_energy": baseline,
        "var_quality": float(quality.var()),
        "mean_quality": float(quality.mean()),
        "std_quality": float(quality.std()),
        "corr_energy_cde": float(np.corrcoef(energy, cde)[0, 1]),
        "mean_energy": float(energy.mean()),
        "mean_cde": float(cde.mean()),
    }
    with open(output / "mu.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    save = Path(args.save) if args.save else output / "mu_calibration.png"
    make_figure(energy, cde, mu, save, dpi=args.dpi)
    print(f"\nwritten to {output} (figure: {save})")


def make_figure(energy: np.ndarray, cde: np.ndarray, mu: float, save: Path, dpi: int = 200) -> None:
    """Four panels: the energy histogram, the trade-off, the scan, the result."""
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), constrained_layout=True)
    figure.patch.set_facecolor(SURFACE)
    for axis in axes.ravel():
        _style(axis)

    axis = axes[0][0]
    axis.hist(energy, bins=60, color=NATURAL_COLOR, alpha=0.85)
    axis.set_xlabel(r"$E_{\mathrm{DCA}}$")
    axis.set_ylabel("natural sequences")
    axis.set_title("Energy of the natural alignment", fontsize=11)
    axis.axvline(energy.mean(), color=OPTIMUM_COLOR, linestyle="--", linewidth=1.2)
    axis.text(
        0.97, 0.94, f"sd = {energy.std():.1f}", transform=axis.transAxes,
        ha="right", va="top", fontsize=9, color="#52514e",
    )

    axis = axes[0][1]
    axis.scatter(cde, energy, s=8, color=ACCENT_COLOR, alpha=0.35, linewidths=0)
    grid = np.linspace(cde.min(), cde.max(), 50)
    intercept = energy.mean() - mu * cde.mean()
    axis.plot(grid, intercept + mu * grid, color=OPTIMUM_COLOR, linewidth=2.0)
    axis.set_xlabel("CDE  (mean conditional entropy, nats)")
    axis.set_ylabel(r"$E_{\mathrm{DCA}}$")
    axis.set_title(rf"Trade-off: slope $\mu^* = {mu:.3g}$", fontsize=11)

    axis = axes[1][0]
    span = max(abs(mu), 1.0)
    scan = np.linspace(mu - 2.5 * span, mu + 2.5 * span, 400)
    variance = np.array([sample_quality(energy, cde, value).var() for value in scan])
    axis.plot(scan, variance, color=ACCENT_COLOR, linewidth=2.0)
    axis.axvline(mu, color=OPTIMUM_COLOR, linestyle="--", linewidth=1.2)
    axis.axhline(energy.var(), color=NATURAL_COLOR, linestyle=":", linewidth=1.2)
    axis.annotate(
        rf"$\mu^* = {mu:.3g}$",
        xy=(mu, variance.min()),
        xytext=(0.58, 0.82),
        textcoords="axes fraction",
        fontsize=10,
        color=OPTIMUM_COLOR,
        arrowprops=dict(arrowstyle="->", color=OPTIMUM_COLOR, linewidth=1.0),
    )
    axis.text(
        0.02, 0.93, r"dotted: Var$(E)$, i.e. $\mu = 0$", transform=axis.transAxes,
        fontsize=9, color="#52514e",
    )
    axis.set_xlabel(r"$\mu$")
    axis.set_ylabel(r"Var$(E_{\mathrm{DCA}} - \mu\,\mathrm{CDE})$")
    axis.set_title("The scan that picks mu", fontsize=11)

    axis = axes[1][1]
    quality = sample_quality(energy, cde, mu)
    scale = float(energy.std())
    axis.hist(
        _centre(energy, scale), bins=60, color=NATURAL_COLOR, alpha=0.55, label=r"$E_{\mathrm{DCA}}$"
    )
    axis.hist(
        _centre(quality, scale), bins=60, color=OPTIMUM_COLOR, alpha=0.7, label=r"$Q_{\mu^*}$"
    )
    axis.set_xlabel("centred and scaled by the energy's own sd")
    axis.set_ylabel("natural sequences")
    axis.set_title(
        f"Sample quality is {100 * (1 - quality.var() / energy.var()):.0f}% tighter than the energy",
        fontsize=11,
    )
    axis.legend(frameon=False, fontsize=9)

    figure.suptitle("Calibrating mu on the natural alignment", fontsize=13)
    save.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save, dpi=dpi, bbox_inches="tight", facecolor=figure.get_facecolor())


def _centre(values: np.ndarray, scale: float) -> np.ndarray:
    """Centre on the mean and divide by a common scale, so widths stay comparable."""
    return (values - values.mean()) / scale


def _style(axis) -> None:
    axis.grid(True, **GRID_KWARGS)
    axis.set_axisbelow(True)
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#a9a8a2")
    axis.tick_params(colors="#52514e", labelsize=9)


if __name__ == "__main__":
    main()
