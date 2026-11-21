"""SI 1 -- controls that bound how much of the F1 result is memorisation.

F1 reports that the Potts conditional ``P(S | sigma_{\\i*})`` at the critic column
separates natural chymotrypsins (S at i*) from natural trypsins (D at i*) at
AUC ~0.99. The model was fitted on this same alignment, so the obvious objection
is circularity: the separation could be lookup rather than generalisation.
Retraining is out of reach, so this figure *bounds* the objection with three
controls computed from existing data.

(a, b)  ROC and precision-recall for three scores on ONE common evaluation set:

  * **Potts** -- ``P(S | sigma_{\\i*})``. The critic column is never read: the
    conditional at i* is a function of the other 264 columns only. No labels.

  * **1-NN (the memorisation ceiling)** -- for each evaluated sequence, the
    Hamming distance to the nearest *other* natural sequence carrying D minus
    the distance to the nearest *other* natural sequence carrying S. Positive
    means "an S-carrier sits closer than any D-carrier". This is pure lookup in
    the training alignment, and it *uses the labels*, so it is the strongest
    thing a memoriser could do. If Potts only matched it, F1 would say little.

  * **Class profile (the no-epistasis floor)** -- independent-site profiles
    f^S_i(a), f^D_i(a) fitted separately to the S- and D-carriers (Laplace
    pseudocount, column i* excluded), scored as the log-likelihood ratio
    sum_{i != i*} log f^S_i(x_i)/f^D_i(x_i). It conditions on context, but with
    no couplings at all, so the Potts-minus-profile gap is what the couplings
    buy. It is fitted on a TRAIN split disjoint from the evaluation set.

  A couplings-free *generative* model cannot condition on context at i* at all:
  its conditional is the column frequency f_i*(a), constant across sequences,
  hence AUC = 0.5 exactly. That degenerate case is the chance diagonal.

(c)  Potts AUC stratified by novelty: the Hamming distance from each evaluated
  sequence to its nearest *other* natural sequence, as a percentage of length.
  If the AUC holds up for sequences far from anything else in the alignment, the
  classifier is not interpolating near-duplicates.

AUC is the Mann-Whitney rank statistic (ties get half credit), not an integral of
the drawn curve: tens of thousands of sequences pile up at P(S) ~ 0 and trapezoid
integration is biased there. Average precision is sum (R_n - R_{n-1}) P_n over
distinct thresholds, with no interpolation.

WHAT IT ACTUALLY SHOWS -- stated plainly, because the figure is a control and a
control that is reported selectively is worthless. On the common evaluation set
the Potts conditional (AUC 0.992, AP 0.968) does NOT beat 1-NN lookup (0.996,
0.989); label-free though it is, it sits just below the memorisation ceiling
overall. It does beat lookup in the one regime where lookup has nothing to copy
from -- sequences more than 25% divergent from any other natural sequence (0.984
vs 0.973) -- and its AUC is flat across the novelty bins, within the bootstrap
intervals. So the F1 separation is not carried by near-duplicates, but this
figure does not establish that the Potts conditional is doing something a
nearest-neighbour table could not. The couplings buy roughly +0.007 AUC and
+0.018 AP over the no-epistasis class profile.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

import config
import data
import style

from observables import min_hamming_to_set  # noqa: E402  (data.py puts src/ on the path)

# --- control parameters ---------------------------------------------------------
N_EVAL = 5_000                 # stratified subsample scored by all three methods
SEED = 20240517
PSEUDOCOUNT = 1.0              # Laplace, for the class profiles
REF_BLOCK = 8_000              # reference sequences per GPU block
NOVELTY_EDGES = (0.0, 5.0, 10.0, 25.0, 100.0)
NOVELTY_LABELS = ("< 5%", "5–10%", "10–25%", "> 25%")


# --- metrics --------------------------------------------------------------------
def auc_rank(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney U / (n+ n-). Ties share rank, so they score exactly 0.5."""
    ranks = rankdata(score)
    n_pos = int(label.sum())
    n_neg = int(label.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _threshold_counts(score: np.ndarray, label: np.ndarray):
    """(tp, fp) at every distinct score threshold, highest score first."""
    order = np.argsort(-score, kind="mergesort")
    sorted_score, sorted_label = score[order], label[order]
    tp = np.cumsum(sorted_label)
    fp = np.cumsum(1 - sorted_label)
    last = np.r_[np.flatnonzero(np.diff(sorted_score)), sorted_score.size - 1]
    return tp[last], fp[last]


def roc_curve(score: np.ndarray, label: np.ndarray):
    tp, fp = _threshold_counts(score, label)
    n_pos, n_neg = label.sum(), label.size - label.sum()
    return np.r_[0.0, fp / n_neg], np.r_[0.0, tp / n_pos]


def pr_curve(score: np.ndarray, label: np.ndarray):
    """Recall, precision and AP = sum (R_n - R_{n-1}) P_n -- no interpolation."""
    tp, fp = _threshold_counts(score, label)
    recall = tp / label.sum()
    precision = tp / (tp + fp)
    average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return recall, precision, average_precision


# --- the controls ----------------------------------------------------------------
def blocked_min_hamming(queries: np.ndarray, reference: np.ndarray,
                        exclude: np.ndarray | None = None) -> np.ndarray:
    """Hamming distance to the nearest member of ``reference``, excluding self.

    ``data.nearest_natural`` cannot be used as it stands (it forwards the model
    object into ``min_hamming_to_set``'s ``sequences`` slot, which raises), so the
    GPU kernel is called directly here. The reference set is walked in blocks
    because one-hot-ing all 89,439 natural sequences at once is 2 GB.

    ``exclude`` is the index into ``reference`` that each query must ignore --
    its own row -- or -1 for none. The alignment holds no exact duplicates
    (89,439 unique of 89,439), so excluding the query's own row is the same as
    discarding every zero distance: no sequence can be its own neighbour.
    """
    best = np.full(len(queries), reference.shape[1] + 1, dtype=np.int64)
    for start in range(0, len(reference), REF_BLOCK):
        block = reference[start:start + REF_BLOCK]
        skip = None
        if exclude is not None:
            local = exclude - start
            skip = np.where((local >= 0) & (local < len(block)), local, -1)
        distance, _ = min_hamming_to_set(queries, block, device=config.DEVICE,
                                         exclude=skip, chunk=512)
        best = np.minimum(best, distance)
    return best


def class_profile_llr(train_sequences: np.ndarray, train_label: np.ndarray,
                      eval_sequences: np.ndarray, n_states: int) -> np.ndarray:
    """Log-likelihood ratio under two independent-site profiles. No couplings."""
    columns = np.setdiff1d(np.arange(train_sequences.shape[1]), [config.CRITIC_COL])
    log_profile = []
    for positive in (1, 0):
        block = train_sequences[train_label == positive][:, columns]
        counts = np.zeros((columns.size, n_states))
        for column in range(columns.size):
            counts[column] = np.bincount(block[:, column], minlength=n_states)
        counts += PSEUDOCOUNT
        log_profile.append(np.log(counts / counts.sum(axis=1, keepdims=True)))
    delta = log_profile[0] - log_profile[1]                      # (columns, states)
    rows = np.arange(columns.size)
    return delta[rows[None, :], eval_sequences[:, columns]].sum(axis=1)


def compute() -> dict:
    """Every number in the figure. Cached per model fingerprint."""
    cache = config.cache_path("si01_controls")
    if cache.is_file():
        blob = np.load(cache)
        return {key: blob[key] for key in blob.files}

    _, encoded = data.alignment()
    P, _ = data.natural_conditional()
    classes = data.natural_classes()
    n_states = len(data.alphabet().tokens)

    is_s, is_d = classes["chymotrypsin"], classes["trypsin"]

    # Evaluate on the HELD-OUT split where there is one. The whole point of this
    # figure is to bound how much of F1 is memorisation, so scoring it on the
    # sequences the model was fitted on would defeat it. The 1-NN and profile
    # baselines still get the WHOLE alignment as their reference/training pool --
    # that is deliberate: it makes them as strong as possible, so beating them
    # means something.
    held_out = data.split_mask()
    evaluable = held_out if held_out is not None else np.ones(len(is_s), dtype=bool)
    labelled = np.flatnonzero((is_s | is_d) & evaluable)
    label_all = is_s[labelled].astype(np.int64)

    # --- a stratified subsample, so both classes are represented --------------
    rng = np.random.default_rng(SEED)
    share = int(round(N_EVAL * label_all.mean()))
    picks = []
    for value, size in ((1, share), (0, N_EVAL - share)):
        pool = labelled[label_all == value]
        picks.append(rng.choice(pool, size=min(size, pool.size), replace=False))
    evaluated = np.sort(np.concatenate(picks))
    label = is_s[evaluated].astype(np.int64)

    # --- (i) 1-NN: distance to the nearest OTHER S- and D-carrier -------------
    queries = encoded[evaluated]
    s_index, d_index = np.flatnonzero(is_s), np.flatnonzero(is_d)
    self_in_s = np.searchsorted(s_index, evaluated)
    self_in_s = np.where((self_in_s < s_index.size) &
                         (s_index[np.minimum(self_in_s, s_index.size - 1)] == evaluated),
                         self_in_s, -1)
    self_in_d = np.searchsorted(d_index, evaluated)
    self_in_d = np.where((self_in_d < d_index.size) &
                         (d_index[np.minimum(self_in_d, d_index.size - 1)] == evaluated),
                         self_in_d, -1)
    nearest_s = blocked_min_hamming(queries, encoded[s_index], self_in_s)
    nearest_d = blocked_min_hamming(queries, encoded[d_index], self_in_d)
    knn_score = (nearest_d - nearest_s).astype(float)

    # --- (iii) novelty: nearest OTHER natural sequence, any class -------------
    novelty = blocked_min_hamming(queries, encoded, evaluated)
    novelty_pct = 100.0 * novelty / encoded.shape[1]

    # --- (ii) class profiles, fitted off the evaluation set -------------------
    # Fitted on every labelled sequence the evaluation set does not contain --
    # including the model's training sequences, which makes this baseline as
    # strong as it can be. (Renamed from `held_out`, which now means the model's
    # own train/test split and would have shadowed it.)
    profile_fit_rows = np.setdiff1d(np.flatnonzero(is_s | is_d), evaluated)
    profile_score = class_profile_llr(encoded[profile_fit_rows],
                                      is_s[profile_fit_rows].astype(np.int64),
                                      queries, n_states)

    result = {
        "evaluated": evaluated,
        "label": label,
        "potts": P[evaluated, data.S_STATE()],
        # The log-odds is F1's headline discriminant. Scoring only P(S) here
        # compared the baselines against the WEAKER of the model's two scores,
        # which stacked the comparison against the model.
        "logodds": data.log_odds(P)[evaluated],
        "knn": knn_score,
        "profile": profile_score,
        "nearest_s": nearest_s,
        "nearest_d": nearest_d,
        "novelty_pct": novelty_pct,
        "n_train_profile": np.array([profile_fit_rows.size]),
        "potts_full": P[labelled, data.S_STATE()],
        "label_full": label_all,
        "length": np.array([encoded.shape[1]]),
    }
    np.savez_compressed(cache, **result)
    return result


def bootstrap_auc(score: np.ndarray, label: np.ndarray, n_resample: int = 2000,
                  seed: int = SEED) -> tuple[float, float]:
    """95% percentile interval for the AUC, resampling each class separately.

    A bin holds a few hundred sequences and only ~100-300 of them carry S, so the
    third decimal of a per-bin AUC is not real. The interval says so on the figure
    rather than in a footnote nobody reads.
    """
    positive = score[label == 1]
    negative = score[label == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resample)
    for repeat in range(n_resample):
        resampled = np.r_[rng.choice(positive, positive.size, replace=True),
                          rng.choice(negative, negative.size, replace=True)]
        draws[repeat] = auc_rank(resampled,
                                 np.r_[np.ones(positive.size, dtype=np.int64),
                                       np.zeros(negative.size, dtype=np.int64)])
    return tuple(np.percentile(draws, [2.5, 97.5]))


def novelty_table(result: dict):
    """Per-bin AUC for Potts and 1-NN, with n, the class split and a CI."""
    rows = []
    for low, high, name in zip(NOVELTY_EDGES[:-1], NOVELTY_EDGES[1:], NOVELTY_LABELS):
        inside = (result["novelty_pct"] >= low) & (result["novelty_pct"] < high)
        label = result["label"][inside]
        # Held-out sequences are far from anything the model saw -- the median
        # is ~30% of L -- so the near-duplicate bins can be empty. An empty or
        # single-class bin has no AUC; drop it rather than plot a nan bar.
        if label.size == 0 or label.min() == label.max():
            continue
        rows.append({
            "label": name,
            "n": int(inside.sum()),
            "n_s": int(label.sum()),
            "n_d": int(label.size - label.sum()),
            # Panel (c) uses the log-odds: it is F1's headline discriminant,
            # and ranking the novelty bins by the weaker P(S) while quoting the
            # log-odds in the legend would compare two different things.
            "potts": auc_rank(result["logodds"][inside], label),
            "potts_ps": auc_rank(result["potts"][inside], label),
            "knn": auc_rank(result["knn"][inside], label),
            "ci": bootstrap_auc(result["logodds"][inside], label),
        })
    return rows


# --- the figure -------------------------------------------------------------------
SCORES = (
    ("Potts  " + style.P_S, "potts", style.BLUE, "-", 2.2, 3),
    ("Potts  " + style.LOG_ODDS, "logodds", style.C_DESIGN, "-", 2.6, 4),
    ("1-NN lookup", "knn", style.MUTED, (0, (5, 2)), 1.8, 2),
    ("class profile", "profile", style.C_OTHER, (0, (1, 1.8)), 2.4, 2),
)


def draw(result: dict, rows) -> None:
    import matplotlib.pyplot as plt

    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 6.9),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.2]})
    ax_roc, ax_pr, ax_bin = axes

    label = result["label"]
    prevalence = label.mean()
    roc_handles, pr_handles = [], []

    for name, key, colour, dash, width, order in SCORES:
        score = result[key]
        fpr, tpr = roc_curve(score, label)
        recall, precision, average_precision = pr_curve(score, label)
        area = auc_rank(score, label)
        roc_handles += ax_roc.plot(fpr, tpr, color=colour, lw=width, ls=dash,
                                   zorder=order, solid_capstyle="round",
                                   label=f"{name}   {area:.3f}")
        pr_handles += ax_pr.plot(recall, precision, color=colour, lw=width, ls=dash,
                                 zorder=order, solid_capstyle="round",
                                 label=f"{name}   {average_precision:.3f}")

    roc_handles += ax_roc.plot([0, 1], [0, 1], color=style.RULE, lw=1.2,
                               ls=(0, (3, 3)), zorder=1, label="chance   0.500")
    pr_handles.append(ax_pr.axhline(prevalence, color=style.RULE, lw=1.2,
                                    ls=(0, (3, 3)), zorder=1,
                                    label=f"chance   {prevalence:.3f}"))

    for ax in (ax_roc, ax_pr):
        style.tidy(ax)
        ax.set_xlim(-0.015, 1.015)
        ax.set_ylim(-0.02, 1.03)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    ax_roc.set_xlabel("false positive rate")
    ax_roc.set_ylabel("true positive rate")
    ax_roc.set_title("S vs D at $i^*$", pad=12)
    legend = ax_roc.legend(handles=roc_handles, loc="lower right", title="ROC AUC",
                           handlelength=2.3, labelspacing=0.55, borderaxespad=0.6,
                           fontsize=12.5, alignment="left")
    legend.get_title().set_fontsize(12.5)

    ax_pr.set_xlabel("recall")
    ax_pr.set_ylabel("precision")
    ax_pr.set_title(f"{int(label.sum()):,} S  vs  {int(label.size - label.sum()):,} D",
                    pad=12)
    legend = ax_pr.legend(handles=pr_handles, loc="lower left", title="average precision",
                          handlelength=2.3, labelspacing=0.55, borderaxespad=0.6,
                          fontsize=12.5, alignment="left")
    legend.get_title().set_fontsize(12.5)

    # --- (c) AUC by novelty ---------------------------------------------------
    x = np.arange(len(rows))
    potts_auc = np.array([row["potts"] for row in rows])
    knn_auc = np.array([row["knn"] for row in rows])
    floor = 0.90

    ax_bin.bar(x, potts_auc - floor, bottom=floor, width=0.56, color=style.C_DESIGN,
               edgecolor="none", zorder=2, label="Potts log-odds")
    ax_bin.plot(x + 0.19, knn_auc, ls="none", marker="o", ms=10, mfc=style.SURFACE,
                mec=style.MUTED, mew=1.8, zorder=4, label="1-NN lookup")

    low = potts_auc - np.array([row["ci"][0] for row in rows])
    high = np.array([row["ci"][1] for row in rows]) - potts_auc
    ax_bin.errorbar(x, potts_auc, yerr=np.vstack([low, high]), fmt="none",
                    ecolor=style.INK, elinewidth=1.6, capsize=5, capthick=1.6,
                    zorder=5)
    for position, row, top in zip(x, rows, potts_auc + high):
        ax_bin.text(position, max(top, row["knn"]) + 0.0035, f"{row['potts']:.3f}",
                    ha="center", va="bottom", fontsize=13, color=style.INK,
                    fontweight="bold")
    style.tidy(ax_bin, grid="y")
    ax_bin.set_xticks(x)
    ax_bin.set_xticklabels([f"{row['label']}\n$n$ = {row['n']:,}\n"
                            f"{row['n_s']:,} S" for row in rows])
    ax_bin.tick_params(axis="x", labelsize=11, length=0)
    ax_bin.set_ylim(floor, 1.020)
    ax_bin.set_yticks([0.90, 0.95, 1.00])
    ax_bin.set_xlim(-0.62, len(rows) - 0.38)
    ax_bin.set_ylabel("AUC within bin   (0.5 = chance)")
    ax_bin.set_xlabel("nearest-neighbour distance (% of $L$)", labelpad=9)
    ax_bin.set_title("the signal survives novelty", pad=42)
    ax_bin.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                  columnspacing=1.6, handletextpad=0.6, fontsize=12.5)

    for ax, letter in zip(axes, "abc"):
        style.panel_letter(ax, letter, dx=-0.175, dy=1.20)

    fig.subplots_adjust(left=0.062, right=0.985, bottom=0.325, top=0.845, wspace=0.34)
    style.caveat(fig,
                 "Bins give $n$ and, of those, the number carrying S; bars carry a 95% "
                 "stratified-bootstrap interval. 1-NN score = Hamming "
                 "to the nearest other natural D-carrier minus Hamming to the\nnearest other "
                 "natural S-carrier; the query is always excluded and the alignment holds no "
                 "exact duplicates. Class profiles are independent-site\n(column "
                 f"{config.CRITIC_COL} omitted), fitted on the "
                 f"{int(result['n_train_profile'][0]):,} labelled sequences outside the "
                 "evaluation subsample. Both baselines read the class labels and may use the "
                 "model's\ntraining sequences, which makes them as strong as possible; the "
                 "Potts conditional reads neither.\n" + data.split_status().capitalize()
                 + ".", y=0.012)
    style.save(fig, "si01_controls")


def main() -> None:
    result = compute()
    rows = novelty_table(result)

    label = result["label"]
    print(f"\nevaluation set: {label.size:,} sequences "
          f"({int(label.sum()):,} S / {int(label.size - label.sum()):,} D), "
          f"L = {int(result['length'][0])}")
    print(f"profile train split: {int(result['n_train_profile'][0]):,} sequences\n")
    full_label = result["label_full"]
    print(f"{'score':<34}{'AUC':>8}{'AP':>9}")
    print(f"{'Potts, FULL labelled alignment':<34}"
          f"{auc_rank(result['potts_full'], full_label):>8.4f}"
          f"{pr_curve(result['potts_full'], full_label)[2]:>9.4f}"
          f"   (n = {full_label.size:,})")
    for name, key, *_ in SCORES:
        flat = name.split("  ")[0]
        print(f"{flat:<34}{auc_rank(result[key], label):>8.4f}"
              f"{pr_curve(result[key], label)[2]:>9.4f}")
    print("\nnovelty bin        n     n_S     n_D   Potts AUC          95% CI   1-NN AUC")
    for row in rows:
        print(f"{row['label']:<12}{row['n']:>6,}{row['n_s']:>8,}{row['n_d']:>8,}"
              f"{row['potts']:>12.4f}  [{row['ci'][0]:.4f}, {row['ci'][1]:.4f}]"
              f"{row['knn']:>11.4f}")
    print(f"\nnovelty (% of L): median {np.median(result['novelty_pct']):.2f}, "
          f"max {result['novelty_pct'].max():.2f}")
    draw(result, rows)


if __name__ == "__main__":
    main()
