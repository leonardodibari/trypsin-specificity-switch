"""H3 -- is the trypsin island a basin, or only a ramp?

THE QUESTION. RESULTS.md section 6 says the designs "stay inside the trypsins".
That is a qualitative statement about where an ensemble happened to land, and it
has two completely different explanations:

  a BASIN   there is a free-energy barrier between trypsin-like sequences and
            chymotrypsin-like ones, and the chains cannot cross it;
  a RAMP    there is no barrier at all, only a steadily rising cost, and the
            chains stop where the cost of another substitution exceeds what the
            bias pays for it.

The restraint settles this, because ``+ t * d_H`` makes t the chemical potential
conjugate to the Hamming distance. Two exact consequences follow, and this
figure is built on them:

    -d<d>/dt = Var(d)                                    (susceptibility)
    P_t(d) proportional to exp(-F(d)) * exp(-t*d)        (reweighting)

A BASIN shows up as a PEAK in Var(d) at some t*, a bimodal P(d) there, and a
non-convex F(d) with a hump between two minima. A RAMP shows up as a Var(d) that
varies smoothly with no peak, unimodal P(d) everywhere, and a convex F(d).

Either answer is a result the project does not currently have, and both replace
a sentence about where chains landed with a number.

THE CONTROL. a3 is the identical scan at lambda = 0. The restraint alone drags
chains toward the family average, and the family carries S at column 201 in
10.4 % of its members, so a negative t raises P(S) with no bias whatsoever. Any
structure seen in a2 that is also in a3 belongs to the family, not to the
design pressure. The two arms are drawn together in every panel for that reason.

THE PANELS
  (a) <d>(t)      the response. Must be monotone DECREASING in t -- that is
                  Var(d) >= 0, an identity, so a violation is not a result but
                  a non-equilibrium flag, and it is checked and reported.
  (b) Var(d)(t)   the susceptibility. The peak, if there is one.
  (c) P(d)        the distance histograms around the peak: one mode or two.
  (d) F(d)        the free-energy profile, reconstructed from the overlapping
                  histograms by WHAM, with its own consistency check.

WHAT d COUNTS. These are all-columns-free runs with --allow-gap-moves, so d_H
counts indels as well as substitutions, and the restraint therefore penalises
both. Panel (a) carries the indel part as a shaded band so that a reader can
never mistake a run that moved 100 residues for one that opened 40 gaps. The
split is also written to the CSV, column by column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data
import style

#: The biased arm and its matched control.
BIASED = "a2_free_lam100"
CONTROL = "a3_free_lam0"

ARMS = {
    BIASED: (r"$\lambda=100$  (biased)", style.BLUE, "o"),
    CONTROL: (r"$\lambda=0$  (matched control)", style.ORANGE, "s"),
}

#: Width of a Hamming-distance bin for the histograms and for F(d). The free
#: runs span d = 0..260, so a bin of 4 keeps ~65 bins -- enough to resolve a
#: two-mode structure, coarse enough that 1000 chains fill it.
BIN_WIDTH = 4


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def arm(block: str) -> pd.DataFrame:
    """Per-chain observables of one arm, saturated runs only.

    Read through the UNSUBSTITUTED block. The critic substitution changes the
    residue at column 201, which shifts every chain's d_H by one wherever the
    chain did not already carry S -- and d_H is the very coordinate this figure
    reconstructs a free energy along. The bias and the restraint both acted on
    the sampled sequence, so the sampled sequence is what P_t(d) is a
    distribution over.
    """
    frame = data.designs(block, serine=False)
    frame["block"] = block
    return frame


def response(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per t: the mean, the variance, and the indel share of d."""
    rows = []
    for t, group in frame.groupby("t", sort=True):
        d = group["hamming_to_wt"].to_numpy(float)
        # The indel part: how far the gap pattern has moved from the wild
        # type's. n_gaps is the design's gap count, so |n_gaps - wt| is a lower
        # bound on the indels and is what is comparable across runs.
        gaps = group["n_gaps"].to_numpy(float)
        rows.append(
            dict(
                t=float(t),
                n=len(d),
                mean_d=d.mean(),
                var_d=d.var(ddof=1),
                # Standard error of a variance, for a reader deciding whether a
                # peak is a peak: Var(s^2) ~ 2*sigma^4/(n-1) for a normal sample.
                var_d_se=d.var(ddof=1) * np.sqrt(2.0 / (len(d) - 1)),
                q25=np.quantile(d, 0.25),
                q75=np.quantile(d, 0.75),
                mean_gaps=gaps.mean(),
                mean_P_S=group["P_S"].mean(),
                mean_E=group["E_DCA"].mean(),
            )
        )
    return pd.DataFrame(rows).sort_values("t").reset_index(drop=True)


def monotonicity(table: pd.DataFrame) -> List[str]:
    """Where <d>(t) fails to decrease. An identity, so a failure is a flag.

    -d<d>/dt = Var(d) >= 0 holds for ANY equilibrium distribution of this
    family. If the sampled <d> rises with t somewhere, that pair of runs is not
    two samples of the same equilibrium ensemble -- one of them has not
    equilibrated -- and nothing may be read off it.
    """
    bad = []
    for (_, lo), (_, hi) in zip(table.iterrows(), table.iloc[1:].iterrows()):
        if hi.mean_d > lo.mean_d:
            bad.append(
                f"<d> rises from {lo.mean_d:.1f} to {hi.mean_d:.1f} "
                f"between t={lo.t:g} and t={hi.t:g}"
            )
    return bad


def fluctuation_response(table: pd.DataFrame) -> pd.DataFrame:
    """The identity ``-d<d>/dt = Var(d)``, checked rung by rung as a RATIO.

    ``monotonicity`` only asks whether the sign is right, which is the weakest
    form of the identity: it passes for any ensemble whose <d> happens to fall.
    The magnitude is the sharp test, and it is also the sharpest equilibrium
    flag available on a single arm, because the two sides fail in opposite ways:

      ratio ~ 1   the measured response matches the measured fluctuation, which
                  is what equilibrium means here.
      ratio < 1   Var(d) exceeds the response. An ensemble still expanding
                  outward from the wild type is a MIXTURE of chains at different
                  stages of that journey, so its spread is inflated, while <d>
                  has not yet reached its asymptote and the finite difference
                  between two such rungs understates the true response. Both
                  errors push the ratio the same way, so a ratio well under 1 is
                  the signature of exactly the pathology EQUILIBRATION_NOTE.md
                  records -- and it says so quantitatively, per rung.

    This matters because a Var(d) PEAK is the evidence for a basin, and a peak
    sitting in a region where this ratio is far from 1 is not evidence of
    anything: the fluctuation it is built from is not an equilibrium fluctuation.
    Central differences, so the first and last rung have no entry.
    """
    table = table.sort_values("t").reset_index(drop=True)
    rows = []
    for index in range(1, len(table) - 1):
        lo, mid, hi = table.iloc[index - 1], table.iloc[index], table.iloc[index + 1]
        response = -(hi.mean_d - lo.mean_d) / (hi.t - lo.t)
        rows.append(
            dict(
                t=mid.t,
                response=response,
                var_d=mid.var_d,
                var_d_se=mid.var_d_se,
                ratio=response / mid.var_d if mid.var_d > 0 else float("nan"),
            )
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# WHAM: the free energy along d
# --------------------------------------------------------------------------

def wham(
    samples: Dict[float, np.ndarray],
    *,
    bin_width: int = BIN_WIDTH,
    tol: float = 1e-10,
    max_iter: int = 100_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Free energy F(d) from histograms taken at several restraint strengths.

    The restraint is exactly linear in the order parameter, which is what makes
    this clean: run k samples ``P_k(d) = P(d) exp(-t_k d) / Z_k``, so the
    unbiased P(d) is recovered by the standard WHAM self-consistency

        P(d)   =  sum_k H_k(d)  /  sum_k N_k exp(f_k - t_k d)
        exp(-f_k) = sum_d P(d) exp(-t_k d)

    iterated to convergence. No model of F is assumed and nothing is fitted:
    the only inputs are the histograms and the t values that produced them.

    Returns ``(centres, F, P)`` with F = -log P, shifted to min 0, and bins that
    no run visited left as NaN.
    """
    ts = np.array(sorted(samples), dtype=float)
    everything = np.concatenate([samples[t] for t in ts])
    edges = np.arange(
        np.floor(everything.min()) - bin_width,
        np.ceil(everything.max()) + 2 * bin_width,
        bin_width,
        dtype=float,
    )
    centres = 0.5 * (edges[:-1] + edges[1:])

    counts = np.stack([np.histogram(samples[t], bins=edges)[0] for t in ts]).astype(float)
    totals = counts.sum(axis=1)                      # N_k
    occupied = counts.sum(axis=0) > 0

    # exp(-t_k * d) for every (run, bin), in logs throughout: t*d reaches ~260
    # at t = 1, and exp(-260) is not a number that survives in float.
    log_bias = -np.outer(ts, centres)                # (k, bins)
    log_N = np.log(totals)[:, None]
    log_counts_total = np.log(counts.sum(axis=0)[occupied])

    f = np.zeros_like(ts)                            # the free energies f_k
    log_P = np.full(centres.shape, -np.inf)
    for _ in range(max_iter):
        # log P(d) = log sum_k H_k(d) - logsumexp_k [ log N_k + f_k - t_k d ]
        denominator = _logsumexp(log_N + f[:, None] + log_bias, axis=0)
        new_log_P = np.full(centres.shape, -np.inf)
        new_log_P[occupied] = log_counts_total - denominator[occupied]
        new_log_P[occupied] -= _logsumexp(new_log_P[occupied])   # normalise

        # exp(-f_k) = sum_d P(d) exp(-t_k d)   ->   f_k = -logsumexp(log P - t d)
        new_f = -_logsumexp(new_log_P[None, :][:, occupied] + log_bias[:, occupied], axis=1)
        new_f -= new_f[0]                            # fix the gauge
        shift = np.max(np.abs(new_f - f))
        f, log_P = new_f, new_log_P
        if shift < tol:
            break
    else:
        print(f"  WHAM did not converge in {max_iter} iterations (last shift {shift:.2e})")

    free_energy = np.where(occupied, -log_P, np.nan)
    free_energy = free_energy - np.nanmin(free_energy)
    probability = np.where(occupied, np.exp(log_P), np.nan)
    return centres, free_energy, probability


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    peak = np.max(a, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    out = peak + np.log(np.sum(np.exp(a - peak), axis=axis, keepdims=True))
    if axis is None:
        return float(out.reshape(()))
    return np.squeeze(out, axis=axis)


def slope_check(
    centres: np.ndarray, free_energy: np.ndarray, table: pd.DataFrame
) -> pd.DataFrame:
    """The built-in validity test of the reconstruction.

    Run k samples ``exp(-F(d) - t_k d)``, so its distribution is peaked where

        dF/dd = -t_k

    Measuring dF/dd of the reconstructed profile at each run's own <d> and
    comparing it with -t_k is a check that costs nothing and uses no new data.
    A profile that fails it was built from histograms that are not samples of
    one common F -- which is what insufficient overlap, or a run that has not
    equilibrated, looks like here.
    """
    good = np.isfinite(free_energy)
    gradient = np.gradient(free_energy[good], centres[good])
    rows = []
    for row in table.itertuples():
        measured = float(np.interp(row.mean_d, centres[good], gradient))
        rows.append(
            dict(t=row.t, mean_d=row.mean_d, expected_slope=-row.t,
                 measured_slope=measured, residual=measured + row.t)
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# the figure
# --------------------------------------------------------------------------

def main() -> None:
    style.apply()
    config.ensure_dirs()

    arms, tables = {}, {}
    for block in ARMS:
        try:
            arms[block] = arm(block)
        except FileNotFoundError:
            print(f"  {block}: not sampled yet, skipping")
            continue
        tables[block] = response(arms[block])

    if BIASED not in tables:
        raise SystemExit(f"{BIASED} has no saturated run; nothing to draw")

    # The equilibrium flag, printed before anything is drawn.
    print("\n  monotonicity of <d>(t)   [-d<d>/dt = Var(d) >= 0, an identity]")
    for block, table in tables.items():
        problems = monotonicity(table)
        print(f"    {block}: " + ("OK" if not problems else "FAILS"))
        for problem in problems:
            print(f"      ! {problem}")

    # The same identity in magnitude, which is the test with teeth.
    print("\n  fluctuation-response ratio  [-d<d>/dt divided by Var(d); 1 = equilibrium]")
    for block, table in tables.items():
        check = fluctuation_response(table)
        if check.empty:
            print(f"    {block}: fewer than three rungs, cannot difference")
            continue
        path = config.RESULTS_ROOT / f"h03_fluctuation_response_{block}.csv"
        check.to_csv(path, index=False)
        worst = check.iloc[(check["ratio"] - 1.0).abs().argmax()]
        print(f"    {block}: ratio spans {check['ratio'].min():.2f}..."
              f"{check['ratio'].max():.2f}, worst {worst.ratio:.2f} at t={worst.t:g}"
              f"  -> {path.name}")

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.6))

    # (a) the response, with the indel part shaded underneath
    ax = axes[0, 0]
    for block, table in tables.items():
        label, colour, marker = ARMS[block]
        ax.plot(table["t"], table["mean_d"], color=colour, lw=1.8, marker=marker,
                markersize=6, markeredgecolor="white", markeredgewidth=0.8,
                label=label, zorder=3)
        ax.fill_between(table["t"], table["q25"], table["q75"], color=colour,
                        alpha=0.16, lw=0, zorder=2)
        # The gap drift: how much of d is the gap pattern rather than residues.
        ax.plot(table["t"], table["mean_gaps"], color=colour, lw=1.3,
                ls=(0, (2, 2)), zorder=3)
    ax.axhspan(81, 196, color=style.C_OTHER, alpha=0.30, lw=0, zorder=0)
    ax.text(ax.get_xlim()[1], 196, " naturals", color=style.MUTED, fontsize=11,
            va="top", ha="right")
    ax.set_xlabel("restraint strength $t$")
    ax.set_ylabel(r"$\langle d_H(\sigma,\,\mathrm{WT})\rangle$")
    ax.legend(loc="upper right", fontsize=12)
    style.panel_letter(ax, "a")
    style.tidy(ax)

    # (b) the susceptibility
    ax = axes[0, 1]
    for block, table in tables.items():
        label, colour, marker = ARMS[block]
        ax.errorbar(table["t"], table["var_d"], yerr=table["var_d_se"],
                    color=colour, lw=1.8, marker=marker, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.8,
                    capsize=3, elinewidth=1.0, zorder=3)
        peak = table.loc[table["var_d"].idxmax()]
        ax.annotate(f"$t^*$ = {peak.t:g}", xy=(peak.t, peak.var_d),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", color=colour, fontsize=12.5)
    ax.set_xlabel("restraint strength $t$")
    ax.set_ylabel(r"$\mathrm{Var}(d_H) = -\,\mathrm{d}\langle d\rangle/\mathrm{d}t$")
    style.panel_letter(ax, "b")
    style.tidy(ax)

    # (c) the distance histograms -- one mode or two
    ax = axes[1, 0]
    table = tables[BIASED]
    shown = _bracket(table)
    for shade, t in zip(style.ramp(style.BLUE, len(shown)), shown):
        d = arms[BIASED].loc[arms[BIASED]["t"] == t, "hamming_to_wt"].to_numpy(float)
        ax.hist(d, bins=np.arange(d.min() - BIN_WIDTH, d.max() + 2 * BIN_WIDTH,
                                  BIN_WIDTH),
                histtype="step", lw=1.8, color=shade, density=True,
                label=f"$t$ = {t:g}")
    ax.set_xlabel(r"$d_H(\sigma,\,\mathrm{WT})$")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=11.5, title=r"$\lambda=100$")
    style.panel_letter(ax, "c")
    style.tidy(ax)

    # (d) the reconstructed free energy
    ax = axes[1, 1]
    checks = {}
    for block, table in tables.items():
        label, colour, marker = ARMS[block]
        samples = {
            float(t): group["hamming_to_wt"].to_numpy(float)
            for t, group in arms[block].groupby("t")
        }
        centres, free_energy, _ = wham(samples)
        ax.plot(centres, free_energy, color=colour, lw=2.0, zorder=3, label=label)
        checks[block] = slope_check(centres, free_energy, table)
    ax.axvspan(81, 196, color=style.C_OTHER, alpha=0.30, lw=0, zorder=0)
    ax.set_xlabel(r"$d_H(\sigma,\,\mathrm{WT})$")
    ax.set_ylabel(r"$F(d) = -\log P(d)$   (unbiased, WHAM)")
    ax.legend(loc="upper left", fontsize=12)
    style.panel_letter(ax, "d")
    style.tidy(ax)

    fig.suptitle("Is the trypsin island a basin, or a ramp?", y=0.955)
    style.caveat(
        fig,
        "all 265 columns free, gap moves on, so $d_H$ counts indels too -- the dashed "
        "line in (a) is the gap drift alone.  shaded band in (a) is the chain-to-chain "
        "IQR; grey band is the range natural chymotrypsins occupy.  (d) is reconstructed "
        "from the overlapping histograms of every $t$ by WHAM, with nothing fitted; its "
        r"consistency check ($\mathrm{d}F/\mathrm{d}d = -t$) is in h03_wham_check.csv.",
    )
    fig.tight_layout(rect=(0, 0.030, 1, 0.935))
    style.save(fig, "h03_susceptibility")

    # The tables behind the figure.
    out = config.RESULTS_ROOT / "h03_response.csv"
    pd.concat([table.assign(block=block) for block, table in tables.items()],
              ignore_index=True).to_csv(out, index=False)
    print(f"wrote {out}")

    out = config.RESULTS_ROOT / "h03_wham_check.csv"
    pd.concat([check.assign(block=block) for block, check in checks.items()],
              ignore_index=True).to_csv(out, index=False)
    print(f"wrote {out}")
    for block, check in checks.items():
        worst = check.loc[check["residual"].abs().idxmax()]
        print(f"  {block}: worst dF/dd residual {worst.residual:+.3f} at t={worst.t:g} "
              f"(expected {worst.expected_slope:+.3f}, measured {worst.measured_slope:+.3f})")


def _bracket(table: pd.DataFrame, n: int = 5) -> List[float]:
    """The t values around the susceptibility peak -- where a split would show.

    A bimodal P(d) appears, if it appears at all, within a window of t around
    the maximum of Var(d); away from it one basin dominates and the histogram is
    unimodal whatever the underlying F looks like.
    """
    ts = list(table["t"])
    peak = int(table["var_d"].to_numpy().argmax())
    lo = max(0, min(peak - n // 2, len(ts) - n))
    return ts[lo:lo + n]


if __name__ == "__main__":
    main()
