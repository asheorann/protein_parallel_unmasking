"""Figures for the blocks/ results.

Local helper, not part of Anushka's code (blocks/results/ is gitignored).
Writes PNG + PDF + a CSV of every plotted number into blocks/figures/.

    python blocks/results/slurm/figures_blocks.py

Five figures:
  fig1  estimator validation   bias and precision of the three variants
  fig2  the sign flip          why the shipped sampling defaults cannot be used
  fig3  bound by strategy      the headline, k-uniform, with overlap visible
  fig4  mechanism              coupling vs bound, raw and protein-centred
  fig5  paired contrasts       the effect that survives, paired within protein

Colour: slots from the validated reference palette, not the Okabe-Ito set in
blocks/plots.py -- that one fails the gate on a white surface (yellow #F0E442
sits at 1.29:1 contrast, and black/grey read as achromatic). Sets used here
were each run through scripts/validate_palette.js --pairs all on #ffffff:
  three variants  #2a78d6 #eb6834 #1baf7a   PASS (aqua 2.82:1 -> direct labels)
  diverging sign  #2a78d6 #d03b3b           PASS
Strategy identity is always carried by an axis label, never by colour alone.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from blocks.evaluate import bootstrap_ci  # noqa: E402

RES = ROOT / "blocks" / "results"
OUT = ROOT / "blocks" / "figures"

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "lines.linewidth": 1.6, "figure.facecolor": "white", "axes.facecolor": "white",
})

ROWS: list[dict] = []     # every plotted number, for the CSV


def rec(fig, panel, label, value, lo=None, hi=None, note=""):
    ROWS.append(dict(figure=fig, panel=panel, label=label, value=value,
                     ci_low=lo, ci_high=hi, note=note))
    return value


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def tidy(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def forest(ax, labels, means, los, his, note=None):
    """Horizontal dot-and-interval against zero. Sign is positional; colour
    reinforces it with the validated diverging pair."""
    y = np.arange(len(labels))[::-1]
    ax.axvline(0, color=INK, linewidth=1.0, zorder=2)
    for yi, m, lo, hi in zip(y, means, los, his):
        c = BLUE if m > 0 else RED
        ax.plot([lo, hi], [yi, yi], color=c, linewidth=2.0, solid_capstyle="round", zorder=3)
        ax.plot([m], [yi], "o", color=c, markersize=7,
                markeredgecolor="white", markeredgewidth=1.6, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    tidy(ax, "x")
    if note:
        ax.set_title(note, loc="left", color=INK2, fontsize=7.5, pad=4)


# ---------------------------------------------------------------- fig 1
def fig1_estimator():
    d = json.loads((RES / "diag_estimator.json").read_text())
    modes = [("forced", "forced mask\n(as shipped)", ORANGE),
             ("bernoulli", "Bernoulli(t)\n(the fix)", BLUE),
             ("kuniform", "k-uniform\n(lower variance)", AQUA)]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.9, 2.7))

    # Panel A: ratio to the exact autoregressive NLL. Truth is 1.
    ar = [r for r in d if r["test"] == "ar"]
    for i, (m, lab, c) in enumerate(modes):
        vals = np.concatenate([r["ratios"] for r in ar if r["mode"] == m])
        x = np.full(len(vals), i) + np.linspace(-0.17, 0.17, len(vals))
        axA.plot(x, vals, "o", color=c, markersize=4.5, alpha=0.75,
                 markeredgecolor="white", markeredgewidth=0.7, zorder=3)
        med = float(np.median(vals))
        axA.plot([i - 0.22, i + 0.22], [med, med], color=c, linewidth=2.2, zorder=4)
        axA.annotate(f"{med:.2f}x", (i, float(vals.max())), xytext=(0, 7),
                     textcoords="offset points", fontsize=8, color=c, ha="center",
                     va="bottom", fontweight="bold", zorder=5)
        rec("fig1", "A", f"{m} median ratio", med,
            float(vals.min()), float(vals.max()), "estimate / exact AR NLL")
    axA.axhline(1.0, color=INK, linewidth=1.0, linestyle="--", zorder=2)
    axA.annotate("exact (ratio = 1)", (-0.42, 1.0), fontsize=7, color=INK,
                 va="bottom", ha="left")
    axA.set_yscale("log")
    axA.set_yticks([1, 2, 4, 8])
    axA.set_yticklabels(["1", "2", "4", "8"])
    axA.set_xticks(range(3))
    axA.set_xticklabels([l for _, l, _ in modes])
    axA.set_xlim(-0.62, 2.52)
    axA.set_ylim(0.72, 13)
    axA.set_ylabel("estimated / exact")
    axA.set_title("A  Bias: does it reduce to autoregression?", loc="left", fontsize=8.5)
    tidy(axA)

    # Panel B: Monte-Carlo spread across seeds, contiguous blocks of 16.
    cg = [r for r in d if r["test"] == "contiguous"]
    n_ts = sorted({r["n_t"] for r in cg})
    w = 0.26
    top = 0.0
    for i, (m, lab, c) in enumerate(modes):
        sds = [float(np.mean([np.std(r["per_residue"], ddof=1)
                              for r in cg if r["mode"] == m and r["n_t"] == nt]))
               for nt in n_ts]
        xs = np.arange(len(n_ts)) + (i - 1) * w
        axB.bar(xs, sds, width=w * 0.86, color=c, zorder=3,
                label=lab.replace("\n", " "), edgecolor="white", linewidth=0.8)
        for x, s in zip(xs, sds):
            axB.annotate(f"{s:.03f}", (x, s), fontsize=6.3, color=INK2,
                         ha="center", va="bottom", xytext=(0, 1.5),
                         textcoords="offset points")
        for nt, s in zip(n_ts, sds):
            rec("fig1", "B", f"{m} sd at n_t={nt}", s, note="nats/residue across 5 seeds")
        top = max(top, max(sds))
    axB.set_xticks(range(len(n_ts)))
    axB.set_xticklabels([f"$n_t$ = {nt}" for nt in n_ts])
    axB.set_ylim(0, top * 1.38)
    axB.set_ylabel("MC spread over seeds (nats/residue, sd)")
    axB.set_title("B  Precision: spread over repeat seeds", loc="left", fontsize=8.5)
    axB.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.99))
    tidy(axB)

    fig.suptitle("Estimator validation on 6 CATH domains, frozen ESM-2 150M",
                 x=0.005, ha="left", fontsize=9, y=1.04)
    save(fig, "fig1_estimator_validation")


# ---------------------------------------------------------------- shared
def paired(rows, a, b, seeds=False):
    names = sorted({r["name"] for r in rows})
    def v(n, s):
        xs = [r["nll_per_residue"] for r in rows if r["name"] == n and r["strategy"] == s]
        return float(np.mean(xs)) if xs else None
    d = [v(n, a) - v(n, b) for n in names if v(n, a) is not None and v(n, b) is not None]
    lo, hi = bootstrap_ci(d)
    return float(np.mean(d)), lo, hi, np.array(d)


# ---------------------------------------------------------------- fig 2
def fig2_signflip():
    cfgs = [("bound_cath.jsonl", "shipped defaults\n$n_t$=8, max blocks=16"),
            ("bound_cath_nt32.jsonl", "more samples\n$n_t$=32, max blocks=32"),
            ("bound_kuniform.jsonl", "k-uniform\n$n_t$=32, 2 seeds")]
    labels, means, los, his = [], [], [], []
    for f, lab in cfgs:
        m, lo, hi, _ = paired(jsonl(RES / f), "cluster", "anticluster")
        labels.append(lab); means.append(m); los.append(lo); his.append(hi)
        rec("fig2", "", lab.replace("\n", " "), m, lo, hi, "cluster - anticluster")

    fig, ax = plt.subplots(figsize=(4.6, 2.35))
    forest(ax, labels, means, los, his)
    for yi, (m, lo, hi) in zip(np.arange(len(labels))[::-1], zip(means, los, his)):
        ax.annotate(f"{m:+.3f}", (hi, yi), xytext=(5, 0), textcoords="offset points",
                    fontsize=7.5, va="center", color=INK, fontweight="bold")
    ax.set_xlabel("cluster $-$ anticluster  (nats / residue)")
    ax.annotate("hypothesis predicts $>$ 0  $\\rightarrow$", (0.96, 0.95),
                xycoords="axes fraction", fontsize=7, color=INK2,
                ha="right", va="top")
    ax.set_xlim(-0.19, 0.125)
    ax.set_title("The decisive contrast changes sign with the sampling budget",
                 loc="left", fontsize=8.5, pad=6)
    save(fig, "fig2_sign_flip")


# ---------------------------------------------------------------- fig 3
def fig3_by_strategy():
    rows = jsonl(RES / "bound_kuniform.jsonl")
    names = sorted({r["name"] for r in rows})
    exact = {"single", "autoregressive"}
    stats = []
    for s in sorted({r["strategy"] for r in rows}):
        per = [float(np.mean([r["nll_per_residue"] for r in rows
                              if r["name"] == n and r["strategy"] == s])) for n in names]
        lo, hi = bootstrap_ci(per)
        stats.append((float(np.mean(per)), s, lo, hi))
        rec("fig3", "", s, float(np.mean(per)), lo, hi, "k-uniform, 100 CATH domains")
    stats.sort()

    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    y = np.arange(len(stats))[::-1]
    for yi, (m, s, lo, hi) in zip(y, stats):
        c = ORANGE if s in exact else BLUE
        ax.plot([lo, hi], [yi, yi], color=c, linewidth=2.0,
                solid_capstyle="round", zorder=3)
        ax.plot([m], [yi], "o", color=c, markersize=7,
                markeredgecolor="white", markeredgewidth=1.6, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([s for _, s, _, _ in stats])
    # extra headroom so the legend sits in clear space, not over the top interval
    ax.set_ylim(y.min() - 0.6, y.max() + 1.45)
    ax.set_xlabel("bound (nats / residue)  $\\downarrow$ tighter")
    tidy(ax, "x")
    h = [plt.Line2D([], [], color=BLUE, marker="o", linewidth=2, markersize=6,
                    markeredgecolor="white", label="partition strategy"),
         plt.Line2D([], [], color=ORANGE, marker="o", linewidth=2, markersize=6,
                    markeredgecolor="white", label="exact special case (MDLM / AR)")]
    ax.legend(handles=h, frameon=False, loc="upper right", fontsize=6.8)
    ax.set_title("Every interval overlaps: no strategy is separated\n"
                 "k-uniform estimator, 100 CATH S35 domains",
                 loc="left", fontsize=8.5, pad=6)
    save(fig, "fig3_bound_by_strategy")


# ---------------------------------------------------------------- fig 4
def fig4_mechanism():
    rows = jsonl(RES / "bound_kuniform.jsonl")
    names = sorted({r["name"] for r in rows})
    strategies = sorted({r["strategy"] for r in rows})

    def v(n, s):
        xs = [r["nll_per_residue"] for r in rows if r["name"] == n and r["strategy"] == s]
        return float(np.mean(xs)) if xs else np.nan
    def cm(n, s):
        xs = [r["coupling_mass"] for r in rows if r["name"] == n and r["strategy"] == s]
        return float(np.mean(xs)) if xs else np.nan

    M = np.array([[v(n, s) for s in strategies] for n in names])     # protein x strategy
    C = np.array([[cm(n, s) for s in strategies] for n in names])

    # `autoregressive` and `single` are EXACT special cases of the method, not
    # partitions competing on slack: AR has zero slack by construction and sits
    # at coupling 0, so including it in the fit manufactures a positive slope
    # that says nothing about coupling. Both are plotted, neither is fitted.
    EXACT = {"single", "autoregressive"}
    fit_j = [j for j, s in enumerate(strategies) if s not in EXACT]
    xmu = np.array([C[:, j].mean() for j in range(len(strategies))])

    def label_means(ax, ys):
        """4-level stagger in x-order. Three strategies sit within 0.05 of each
        other on this axis, so two levels are not enough to keep their boxes
        apart."""
        levels = [(0, 15), (0, 32), (0, -17), (0, -34)]
        for rank, j in enumerate(np.argsort(xmu)):
            dx, dy = levels[rank % 4]
            ha = "left" if xmu[j] < 0.05 else ("right" if xmu[j] > 0.95 else "center")
            ax.annotate(strategies[j], (xmu[j], ys[j]), xytext=(dx + (3 if ha == "left" else 0), dy),
                        textcoords="offset points", fontsize=6.3, color=INK,
                        ha=ha, va="center", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.13", fc="white",
                                  ec="none", alpha=0.9))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.6, 3.1))

    # Panel A: as the data come. All eight shown, fit on the six partitions.
    axA.scatter(C.ravel(), M.ravel(), s=6, color=MUTED, alpha=0.3,
                linewidths=0, zorder=2)
    ys = []
    for j, s in enumerate(strategies):
        mu = float(M[:, j].mean())
        sem = float(M[:, j].std(ddof=1) / np.sqrt(len(names)))
        c = ORANGE if s in EXACT else BLUE
        axA.errorbar([xmu[j]], [mu], yerr=[1.96 * sem], fmt="o", color=c,
                     markersize=7, markeredgecolor="white", markeredgewidth=1.5,
                     elinewidth=1.6, capsize=2.5, ecolor=c, zorder=4)
        ys.append(mu)
        rec("fig4", "A", f"{s} mean bound", mu, mu - 1.96 * sem, mu + 1.96 * sem,
            "exact special case, not fitted" if s in EXACT else "")
    label_means(axA, ys)
    xf, yf = C[:, fit_j].ravel(), M[:, fit_j].ravel()
    sl, ic = np.polyfit(xf, yf, 1)
    r = float(np.corrcoef(xf, yf)[0, 1])
    g = np.linspace(xf.min(), xf.max(), 50)
    axA.plot(g, sl * g + ic, "--", color=INK, linewidth=1.2, zorder=3)
    axA.annotate(f"fit on the 6 partitions:\nslope {sl:+.3f},  $r$ = {r:+.3f}",
                 (0.03, 0.04), xycoords="axes fraction", fontsize=7.2, color=INK)
    rec("fig4", "A", "between+within slope", float(sl),
        note=f"r={r:.4f}, n={len(xf)}, exact special cases excluded")
    axA.set_xlabel("within-block coupling mass")
    axA.set_ylabel("bound (nats / residue)")
    axA.set_xlim(-0.07, 1.14)
    axA.set_title("A  As measured: protein identity dominates", loc="left", fontsize=8.5)
    tidy(axA, "both")

    # Panel B: the within-protein (fixed-effects) relation. BOTH axes are
    # centred on each protein's own mean over the six partitions -- centring
    # only the outcome leaves the regressor carrying between-protein variance
    # and attenuates the slope. The six partitions only, since the centring
    # reference is their mean.
    Cs, Ms = C[:, fit_j], M[:, fit_j]
    Cc = Cs - Cs.mean(axis=1, keepdims=True)
    Mc = Ms - Ms.mean(axis=1, keepdims=True)
    axB.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    axB.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
    axB.scatter(Cc.ravel(), Mc.ravel(), s=6, color=MUTED, alpha=0.3,
                linewidths=0, zorder=2)
    xmu_c = np.array([Cc[:, k].mean() for k in range(len(fit_j))])
    ysb = []
    for k, j in enumerate(fit_j):
        mu = float(Mc[:, k].mean())
        sem = float(Mc[:, k].std(ddof=1) / np.sqrt(len(names)))
        axB.errorbar([xmu_c[k]], [mu], yerr=[1.96 * sem], fmt="o", color=BLUE,
                     markersize=7, markeredgecolor="white", markeredgewidth=1.5,
                     elinewidth=1.6, capsize=2.5, ecolor=BLUE, zorder=4)
        ysb.append(mu)
        rec("fig4", "B", f"{strategies[j]} centred", mu,
            mu - 1.96 * sem, mu + 1.96 * sem, "within-protein deviation")
    levels = [(0, 12), (0, 24), (0, -16)]
    for rank, k in enumerate(np.argsort(xmu_c)):
        dx, dy = levels[rank % 3]
        axB.annotate(strategies[fit_j[k]], (xmu_c[k], ysb[k]), xytext=(dx, dy),
                     textcoords="offset points", fontsize=6.5, color=INK,
                     ha="center", va="center", zorder=6,
                     bbox=dict(boxstyle="round,pad=0.13", fc="white",
                               ec="none", alpha=0.88))
    slc, icc = np.polyfit(Cc.ravel(), Mc.ravel(), 1)
    rc = float(np.corrcoef(Cc.ravel(), Mc.ravel())[0, 1])
    gc = np.linspace(Cc.min(), Cc.max(), 50)
    axB.plot(gc, slc * gc + icc, "--", color=INK, linewidth=1.2, zorder=3)
    axB.annotate(f"within-protein fit:\nslope {slc:+.3f},  $r$ = {rc:+.3f}",
                 (0.03, 0.04), xycoords="axes fraction", fontsize=7.2, color=INK)
    rec("fig4", "B", "within-protein slope", float(slc),
        note=f"r={rc:.4f}, n={Cc.size}, both axes protein-centred")
    axB.set_xlabel("coupling mass $-$ protein mean")
    axB.set_ylabel("bound $-$ protein mean")
    axB.set_title("B  Within protein: both axes centred", loc="left", fontsize=8.5)
    tidy(axB, "both")

    h = [plt.Line2D([], [], color=BLUE, marker="o", linestyle="", markersize=6,
                    markeredgecolor="white", label="partition (fitted)"),
         plt.Line2D([], [], color=ORANGE, marker="o", linestyle="", markersize=6,
                    markeredgecolor="white", label="exact special case (not fitted)")]
    axA.legend(handles=h, frameon=False, loc="upper left", fontsize=6.6,
               bbox_to_anchor=(-0.01, 1.02))
    fig.suptitle("Does bound slack rise with within-block coupling mass?",
                 x=0.005, ha="left", fontsize=9, y=1.05)
    save(fig, "fig4_mechanism")
    return sl, r, slc, rc


# ---------------------------------------------------------------- fig 5
def fig5_contrasts():
    """Revised 2026-11-16. Two things the first version got wrong:

    * its title quoted an "MC noise floor per protein" of 0.082. That number is
      the sd of the raw seed-0 minus seed-1 difference pooled over all eight
      strategies, and it is carried by `autoregressive`, whose 32-of-n position
      subsample alone has sd 0.145. For the multi-block partitions the
      Monte-Carlo sd of one estimate is 0.021-0.027, and of a two-seed paired
      contrast about 0.022. It is now computed per contrast, not hard-coded.
    * it starred `single - anticluster` on the two-seed mean. Each seed reuses
      ONE random stream for all 100 proteins, so a seed is not 100 independent
      replicates: `single` moves by 0.030 between the seeds, in the same
      direction on 67 proteins, and that contrast is -0.005 in one seed and
      -0.033 in the other. The star now needs the interval to exclude zero in
      each seed separately, and each seed is drawn.
    """
    rows = jsonl(RES / "bound_kuniform.jsonl")
    seeds = sorted({r["seed"] for r in rows})
    pairs = [("cluster", "anticluster", "most vs least coupled"),
             ("cluster", "contiguous", "clustered vs BD3-LM"),
             ("contiguous", "anticluster", "BD3-LM vs least coupled"),
             ("single", "anticluster", "MDLM vs least coupled"),
             ("single", "autoregressive", "MDLM vs AR")]
    labels, means, los, his, per_seed, robust = [], [], [], [], [], []
    for a, b, why in pairs:
        m, lo, hi, d = paired(rows, a, b)
        by_seed = [paired([r for r in rows if r["seed"] == s], a, b) for s in seeds]
        # Monte-Carlo sd of the two-seed mean contrast on one protein
        mc_sd = float(np.std(by_seed[0][3] - by_seed[-1][3], ddof=1) / 2)
        ok = all(not (l < 0 < h) for _, l, h, _ in by_seed)
        labels.append(f"{a} $-$ {b}\n{why}")
        means.append(m); los.append(lo); his.append(hi)
        per_seed.append([s[0] for s in by_seed]); robust.append(ok)
        rec("fig5", "", f"{a} - {b}", m, lo, hi, why)
        for s, (ms, ls, hs, _) in zip(seeds, by_seed):
            rec("fig5", "", f"{a} - {b}, seed {s}", ms, ls, hs, "one seed alone")
        rec("fig5", "", f"{a} - {b}, proteins with a positive difference",
            float(np.mean(d > 0)), note="fraction of 100")
        rec("fig5", "", f"{a} - {b}, MC sd per protein", mc_sd,
            note="sd of the two-seed mean contrast, from the seed difference")

    fig, ax = plt.subplots(figsize=(5.0, 3.35))
    forest(ax, labels, means, los, his)
    ys = np.arange(len(labels))[::-1]
    for yi, ms in zip(ys, per_seed):
        ax.plot(ms, [yi - 0.27] * len(ms), "|", color=INK2, markersize=7,
                markeredgewidth=1.3, zorder=5)
    # label on the outer side of the interval, so it never sits on the zero line
    for yi, m, lo, hi, ok in zip(ys, means, los, his, robust):
        neg = m < 0
        ax.annotate(f"{'*  ' if ok and neg else ''}{m:+.3f}{'  *' if ok and not neg else ''}",
                    (lo if neg else hi, yi), xytext=(-5 if neg else 5, 0),
                    textcoords="offset points", fontsize=7.5, va="center",
                    ha="right" if neg else "left", color=INK, fontweight="bold")
    ax.set_xlabel("difference in bound (nats / residue),  paired within protein")
    ax.set_xlim(-0.088, 0.105)
    ax.set_title("Paired contrasts, k-uniform, 100 domains\n"
                 "ticks: each seed alone;  * interval excludes zero in both seeds",
                 loc="left", fontsize=8.5, pad=6)
    save(fig, "fig5_paired_contrasts")


if __name__ == "__main__":
    print("drawing into", OUT)
    fig1_estimator()
    fig2_signflip()
    fig3_by_strategy()
    fig4_mechanism()
    fig5_contrasts()
    with open(OUT / "figure_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["figure", "panel", "label", "value",
                                          "ci_low", "ci_high", "note"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"  wrote figure_data.csv ({len(ROWS)} rows)")
