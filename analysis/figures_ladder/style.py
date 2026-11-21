"""The one place colours, fonts and notation are defined.

Every figure script imports from here. No script defines a colour of its own --
that is what makes the set read as one piece of work rather than ten.

Palette provenance: the categorical slots are the validated reference palette of
the dataviz design system (blue / orange / aqua / yellow / magenta / green /
violet / red). For *adjacent* pairs (lines, bars, histograms) the full order
passes the colourblind and normal-vision separation gates. For *all-pairs* forms
(scatter, PCA) only the first three slots are validated, so anything with more
than three groups must carry a second encoding -- marker shape plus a direct
label. ``SCATTER_GROUPS`` below enforces that by pairing colour with marker.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --- categorical slots ---------------------------------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#6250d6"
RED = "#e34948"

# --- ink and surfaces ----------------------------------------------------------
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#ebe9e4"
RULE = "#c9c7c0"
FAINT = "#dedcd6"
SURFACE = "#ffffff"

# --- semantic roles, fixed across every figure ---------------------------------
C_TRYPSIN = BLUE          # natural sequences carrying D at i*
C_CHYMO = ORANGE          # natural sequences carrying S at i*
C_OTHER = "#b8b6b0"       # every other natural sequence (background)
C_DESIGN = AQUA           # model scores / designed ensembles
C_WT = INK                # the wild type, always labelled
C_CHYMO_REF = "#9c3d16"   # the chymotrypsin reference sequences
C_1992 = INK              # the 1992 construct series: ink, dotted, numbered

#: Colour + marker for the four table conditions. Colour alone is not enough at
#: four groups in an all-pairs form, so the marker is part of the identity.
#: Keyed by stable ROLE, not by display name: the consensus names carry a
#: derived size that changes with the threshold, and a palette that breaks when
#: a threshold moves is a palette that will break silently.
SCATTER_GROUPS = {
    "grafting":            {"color": BLUE,   "marker": "o"},
    "grafting_cool":       {"color": ORANGE, "marker": "s"},
    "consensus_tight":     {"color": AQUA,   "marker": "^"},
    "consensus_wide":      {"color": VIOLET, "marker": "D"},
    # Magenta, not green: green sits next to the aqua used for the tight
    # consensus set, and "^" vs "v" is a weak second cue when the hues are
    # already close -- the two were genuinely confusable in F6.
    "consensus_wide_cool": {"color": MAGENTA, "marker": "v"},
}


def group(condition: str) -> dict:
    """Colour and marker for a condition, resolved through its role."""
    import config
    return SCATTER_GROUPS[config.CONDITION_ROLE[condition]]

#: Rank within a condition -> marker fill. Best is solid, runner-up is hollow.
RANK_FILL = {0: "full", 1: "none"}

# --- notation ------------------------------------------------------------------
# One spelling of every symbol, used in axis labels, legends and the README.
I_STAR = r"i^*"
SIG_REST = r"\sigma_{\setminus i^*}"
P_S = r"$P(\mathrm{S}\mid\sigma_{\setminus i^*})$"
P_D = r"$P(\mathrm{D}\mid\sigma_{\setminus i^*})$"
P_S_PLAIN = "P(S | sigma_{\\i*})"
LOG_ODDS = (r"$\log\,P(\mathrm{S}\mid\sigma_{\setminus i^*})/"
            r"P(\mathrm{D}\mid\sigma_{\setminus i^*})$")
NEG_P_D = r"$-P(\mathrm{D}\mid\sigma_{\setminus i^*})$"
CRITIC_NOTE = r"at the critic site $i^*$ (residue 189)"

LABELS = {
    "hamming_to_wt": "Hamming to wild type",
    "P_S": P_S,
    "E_DCA": r"$E_{\mathrm{DCA}}$",
    "Q": r"$Q_\mu$   (lower is better)",
    "CDE": "CDE (nats)",
    "min_hamming_natural_S_pct": "to nearest natural S (%)",
    "log_odds": LOG_ODDS,
    "n_gaps": "gap columns",
}

#: The corner-plot variable lists. F2 uses all five; F6 drops the natural-neighbour
#: distance, which is not comparable across site sets of different size.
CORNER_VARS_FULL = ["hamming_to_wt", "P_S", "E_DCA", "Q", "min_hamming_natural_S_pct"]
CORNER_VARS_SELECTED = ["hamming_to_wt", "P_S", "E_DCA", "Q"]


def apply() -> None:
    """Install the shared rcParams. Call once at the top of every figure script."""
    mpl.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13.5,
        "figure.titlesize": 19,
        "axes.linewidth": 1.1,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "axes.edgecolor": "#3a3a38",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": "#3a3a38",
        "ytick.color": "#3a3a38",
        "axes.grid": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "legend.frameon": False,
    })


def mix(colour: str, towards: str, amount: float) -> tuple:
    """Blend two hex colours. Used to build single-hue ramps."""
    import matplotlib.colors as mcolors
    a = np.array(mcolors.to_rgb(colour))
    b = np.array(mcolors.to_rgb(towards))
    return tuple(a + (b - a) * amount)


def ramp(base: str, n: int) -> list:
    """``n`` steps of ONE hue, light to dark.

    An ordered variable (lambda, beta) must not be coloured with categorical
    hues -- that implies categories where there is a scale. This blends the base
    towards the surface for the light end and towards the ink for the dark end,
    so no new hue is introduced.
    """
    if n <= 1:
        return [base]
    out = []
    for i in range(n):
        t = i / (n - 1)
        out.append(mix(SURFACE, base, 0.35 + 0.65 * t) if t < 0.5
                   else mix(base, INK, (t - 0.5) * 0.75))
    return out


def tidy(ax, *, grid: str = "both") -> None:
    """Drop the top/right spines and put a recessive grid behind the data."""
    ax.spines[["top", "right"]].set_visible(False)
    if grid != "none":
        ax.grid(axis=grid if grid in ("x", "y") else "both", color=GRID, lw=1.0)
    ax.set_axisbelow(True)


def panel_letter(ax, letter: str, *, dx: float = -0.145, dy: float = 1.045) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=21,
            fontweight="bold", va="top", ha="left", color=INK)


def save(fig, stem: str, *, out_dir: Path | None = None) -> Path:
    """Write ``<stem>.png`` (300 dpi) and ``<stem>.pdf``; return the PNG path."""
    from config import OUT_DIR, ensure_dirs
    ensure_dirs()
    out_dir = Path(out_dir or OUT_DIR)
    png = out_dir / f"{stem}.png"
    fig.savefig(png, dpi=300)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"wrote {png}")
    return png


def caveat(fig, text: str, *, y: float = 0.005) -> None:
    """A one-line caveat along the bottom of a figure. Used where a figure would
    otherwise overstate its claim (F1's training-set circularity, the 1992
    construct being a reconstruction, the placeholder condition)."""
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=11, color=MUTED)
