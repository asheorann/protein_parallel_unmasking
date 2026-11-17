"""Figures for Experiment 2 (generation quality vs partition).

Local helper, not part of Anushka's code (blocks/results/ is gitignored).
Companion to figures_blocks.py, same house style and palette.

What these four figures have to carry:

  fig6  quality against MEASURED NFE, with the composition null drawn on the
        recovery panel. Requested steps are not compute: a block gets at most
        one call per residue, so the 128-step budget delivers ~0.91 L.
  fig7  the confound. pLDDT falls steeply with sequence entropy, so a
        partition that produces lower-complexity sequences scores higher
        without being better.
  fig8  the paired pLDDT contrasts before and after removing that confound.
  fig9  the question Experiment 2 exists to answer: does a tighter bound give
        better samples? Per-protein Spearman over the eight strategies.

Palette validated with the dataviz validator (light surface, categorical):
  #2a78d6 #eb6834 #1baf7a #d03b3b -> all checks PASS, worst adjacent pair
  #1baf7a/#eb6834 deltaE 9.2 deutan / 27.6 normal. #1baf7a sits at 2.74:1
  against the surface, which the validator flags as requiring relief, so every
  series is directly labelled and identity is never carried by colour alone.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "blocks" / "results"
OUT = ROOT / "blocks" / "figures"

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"
GRID, FAINT = "#e4e3df", "#c9c8c3"

# Three partitions carry the argument; the rest are context, drawn recessive.
HILITE = {"contiguous": BLUE, "anticluster": ORANGE, "cluster": AQUA}
AR = "autoregressive"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "lines.linewidth": 1.6, "figure.facecolor": "white", "axes.facecolor": "white",
})

ROWS: list[dict] = []


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


def tidy(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def right_labels(ax, items, fontsize=6.6, gap_frac=0.055):
    """Direct-label every series in a column at the right edge.

    Eight series will not fit as labels pinned to their own marks: in the first
    draft `contiguous`, `cluster` and `anticluster` landed within 0.2 pLDDT of
    each other and the text overprinted. So the labels are spread vertically to
    a minimum gap and a hairline leader connects each back to its mark, which
    keeps identity on the label rather than on colour (the validator also flags
    #1baf7a at 2.74:1, so colour alone was never enough).

    items: list of (x_data, y_data, text, colour).
    """
    if not items:
        return
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    gap = gap_frac * span
    items = sorted(items, key=lambda t: t[1])
    ys = [it[1] for it in items]
    for _ in range(500):
        moved = False
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < gap:
                d = (gap - (ys[i] - ys[i - 1])) / 2.0
                ys[i - 1] -= d
                ys[i] += d
                moved = True
        lo = min(ys)
        if lo < y0 + 0.5 * gap:
            ys = [v + (y0 + 0.5 * gap - lo) for v in ys]
        hi = max(ys)
        if hi > y1 - 0.5 * gap:
            ys = [v - (hi - (y1 - 0.5 * gap)) for v in ys]
        if not moved:
            break
    xr = ax.get_xlim()[1]
    for (xd, yd, text, col), ylab in zip(items, ys):
        ax.annotate("", xy=(xr, ylab), xytext=(xd, yd),
                    textcoords="data", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=FAINT, linewidth=0.6,
                                    shrinkA=2, shrinkB=0))
        ax.annotate(text, xy=(xr, ylab), xytext=(4, 0),
                    textcoords="offset points", fontsize=fontsize,
                    color=INK if col else MUTED, va="center",
                    annotation_clip=False)


def boot(x, n=20000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def entropy(s):
    c = Counter(s)
    p = np.array([v / len(s) for v in c.values()])
    return float(-(p * np.log(p)).sum())


# ---------------------------------------------------------------- data
rows = [json.loads(l) for l in open(RES / "folded.jsonl") if l.strip()]
for r in rows:
    r["entropy"] = entropy(r["sequence"])
STRATS = sorted({r["strategy"] for r in rows})
BUDGETS = sorted({r["steps_requested"] for r in rows})
CELLS = defaultdict(list)
for r in rows:
    CELLS[(r["name"], r["steps_requested"], r["seed"])].append(r)

# composition null for recovery
_gen = Counter("".join(r["sequence"] for r in rows))
_nat = Counter("".join({r["name"]: r["native"] for r in rows}.values()))
_g, _n = sum(_gen.values()), sum(_nat.values())
NULL = sum((_gen[a] / _g) * (_nat[a] / _n) for a in "ACDEFGHIKLMNPQRSTVWY")

# within-cell slope of pLDDT on entropy
_p, _e = [], []
for c, rs in CELLS.items():
    p = np.array([x["plddt"] for x in rs])
    e = np.array([x["entropy"] for x in rs])
    _p += list(p - p.mean())
    _e += list(e - e.mean())
SLOPE = float(np.polyfit(_e, _p, 1)[0])
SLOPE_R = float(np.corrcoef(_e, _p)[0, 1])


# ---------------------------------------------------------------- fig 6
def fig6_quality_vs_nfe():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.2, 2.9))
    fig.subplots_adjust(wspace=0.52)

    for ax, met, ylab in ((axA, "plddt", "ESMFold pLDDT"),
                          (axB, "recovery", "sequence recovery")):
        labels = []
        for s in STRATS:
            if s == AR:
                continue
            xs, ys = [], []
            for b in BUDGETS:
                sub = [r for r in rows if r["strategy"] == s
                       and r["steps_requested"] == b]
                xs.append(float(np.mean([r["nfe"] for r in sub])))
                ys.append(float(np.mean([r[met] for r in sub])))
            c = HILITE.get(s)
            ax.plot(xs, ys, "-o", color=c or FAINT, markersize=4 if c else 3,
                    linewidth=1.8 if c else 1.0, zorder=4 if c else 2,
                    markeredgecolor="white", markeredgewidth=0.8 if c else 0)
            labels.append((xs[-1], ys[-1], s, c))
        sub = [r for r in rows if r["strategy"] == AR]
        ax_, ay_ = (float(np.mean([r["nfe"] for r in sub])),
                    float(np.mean([r[met] for r in sub])))
        ax.plot([ax_], [ay_], "D", color=RED, markersize=6,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)
        labels.append((ax_, ay_, "autoregressive", RED))
        ax.set_xscale("log")
        ax.set_xlabel("measured NFE (denoiser calls, log scale)")
        ax.set_ylabel(ylab)
        tidy(ax, "both")
        ax.margins(y=0.14)
        right_labels(ax, labels)

    axB.axhline(NULL, color=INK, linewidth=1.0, linestyle=(0, (4, 2)), zorder=3)
    axB.annotate(f"composition null {NULL:.3f}", (0.03, 0.04),
                 xycoords="axes fraction", fontsize=6.8, color=INK,
                 bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none",
                           alpha=0.92))
    axA.annotate("autoregressive is one point:\nits NFE ignores the budget",
                 (0.03, 0.94), xycoords="axes fraction", fontsize=6.4,
                 color=INK2, va="top")
    axA.set_title("A  pLDDT: every partition folds badly", loc="left", fontsize=8.5)
    axB.set_title("B  recovery: pinned to the null", loc="left", fontsize=8.5)
    fig.suptitle("Sample quality against measured compute, 30 CATH domains, 5 seeds",
                 x=0.005, ha="left", fontsize=9, y=1.04)

    for s in STRATS:
        for b in BUDGETS:
            sub = [r for r in rows if r["strategy"] == s and r["steps_requested"] == b]
            rec("fig6", "A", f"{s} steps={b} pLDDT",
                float(np.mean([r["plddt"] for r in sub])),
                note=f"measured NFE {np.mean([r['nfe'] for r in sub]):.0f}")
            rec("fig6", "B", f"{s} steps={b} recovery",
                float(np.mean([r["recovery"] for r in sub])),
                note=f"measured NFE {np.mean([r['nfe'] for r in sub]):.0f}")
    rec("fig6", "B", "composition null", NULL, note="sum_a p_gen(a) p_nat(a)")
    save(fig, "fig6_quality_vs_nfe")


# ---------------------------------------------------------------- fig 7
def fig7_confound():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.4, 3.1))
    fig.subplots_adjust(wspace=0.46)

    # Panel A is the relationship over all 6000 sequences. The eight strategy
    # means span only 0.36 nats and 6.6 pLDDT, so plotting them here buried the
    # labels on top of each other; they get their own zoomed panel instead.
    x = np.array([r["entropy"] for r in rows])
    y = np.array([r["plddt"] for r in rows])
    axA.scatter(x, y, s=4, color=MUTED, alpha=0.07, linewidths=0, zorder=2)
    sl, ic = np.polyfit(x, y, 1)
    r_all = float(np.corrcoef(x, y)[0, 1])
    g = np.linspace(x.min(), x.max(), 50)
    axA.plot(g, sl * g + ic, "--", color=INK, linewidth=1.3, zorder=5)
    axA.annotate(f"all 6000 sequences: slope {sl:+.1f} pLDDT / nat,  $r$ = {r_all:+.3f}\n"
                 f"within one protein, budget and seed: {SLOPE:+.1f},  "
                 f"$r$ = {SLOPE_R:+.3f}",
                 (0.03, 0.06), xycoords="axes fraction", fontsize=7, color=INK)
    axA.set_xlabel("sequence entropy (nats)")
    axA.set_ylabel("ESMFold pLDDT")
    axA.set_title("A  Low-complexity sequences fold 'confidently'",
                  loc="left", fontsize=8.5)
    tidy(axA, "both")
    rec("fig7", "A", "raw slope", float(sl), note=f"r={r_all:.3f}, n={len(x)}")
    rec("fig7", "A", "within-cell slope", SLOPE,
        note=f"r={SLOPE_R:.3f}, n={len(_e)}, both axes centred on the cell mean")

    # Panel B: the same axes, zoomed to the strategy means. The dashed line is
    # the within-cell slope, not a fit to these eight points.
    mx = {s: float(np.mean([r["entropy"] for r in rows if r["strategy"] == s]))
          for s in STRATS}
    my = {s: float(np.mean([r["plddt"] for r in rows if r["strategy"] == s]))
          for s in STRATS}
    labels = []
    for s in STRATS:
        col = HILITE.get(s, RED if s == AR else None)
        axB.plot([mx[s]], [my[s]], "o", color=col or INK2, markersize=7.5,
                 markeredgecolor="white", markeredgewidth=1.5, zorder=6)
        labels.append((mx[s], my[s], s, col))
        rec("fig7", "B", f"{s} mean entropy", mx[s])
        rec("fig7", "B", f"{s} mean pLDDT", my[s])
    cx = float(np.mean(list(mx.values())))
    cy = float(np.mean(list(my.values())))
    gx = np.linspace(min(mx.values()) - 0.03, max(mx.values()) + 0.03, 20)
    axB.plot(gx, cy + SLOPE * (gx - cx), "--", color=INK, linewidth=1.2, zorder=3)
    axB.set_xlabel("mean sequence entropy (nats)")
    axB.set_ylabel("mean ESMFold pLDDT")
    axB.set_xlim(min(mx.values()) - 0.06, max(mx.values()) + 0.06)
    axB.margins(y=0.2)
    axB.set_title(f"B  The eight partitions, zoomed\n"
                  f"dashed: the within-cell slope {SLOPE:+.1f}, not a fit to these 8",
                  loc="left", fontsize=8.5)
    tidy(axB, "both")
    right_labels(axB, labels, gap_frac=0.085)

    fig.suptitle("Is the pLDDT ranking of partitions a ranking of sequence degeneracy?",
                 x=0.005, ha="left", fontsize=9, y=1.05)
    save(fig, "fig7_degeneracy_confound")


# ---------------------------------------------------------------- fig 8
def fig8_adjusted_contrasts():
    pairs = [("cluster", "anticluster", "most vs least coupled"),
             ("contiguous", "anticluster", "BD3-LM vs least coupled"),
             ("cluster", "contiguous", "clustered vs BD3-LM"),
             (AR, "contiguous", "AR vs BD3-LM")]
    labels, raws, adjs = [], [], []
    for a, b, why in pairs:
        d, de = [], []
        for c, rs in CELLS.items():
            m = {r["strategy"]: r for r in rs}
            if a in m and b in m:
                d.append(m[a]["plddt"] - m[b]["plddt"])
                de.append(m[a]["entropy"] - m[b]["entropy"])
        d, de = np.array(d), np.array(de)
        res = d - SLOPE * de
        raws.append(boot(d))
        adjs.append(boot(res))
        labels.append(f"{a} $-$ {b}\n{why}")
        rec("fig8", "", f"{a} - {b} raw", *raws[-1], note="pLDDT, paired in cell")
        rec("fig8", "", f"{a} - {b} entropy-adjusted", *adjs[-1],
            note=f"minus {SLOPE:.2f} x mean entropy gap {de.mean():+.3f} nats")
        rec("fig8", "", f"{a} - {b} entropy gap", float(de.mean()), note="nats")

    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    y = np.arange(len(labels))[::-1].astype(float)
    ax.axvline(0, color=INK, linewidth=1.0, zorder=2)
    for yi, (m1, l1, h1), (m2, l2, h2) in zip(y, raws, adjs):
        for dy, (m, lo, hi), face in ((0.16, (m1, l1, h1), True),
                                      (-0.16, (m2, l2, h2), False)):
            c = BLUE if m > 0 else RED
            ax.plot([lo, hi], [yi + dy] * 2, color=c, linewidth=2.0,
                    solid_capstyle="round", zorder=3)
            ax.plot([m], [yi + dy], "o", color=c if face else "white",
                    markersize=7, markeredgecolor=c if not face else "white",
                    markeredgewidth=1.6, zorder=4)
            ax.annotate(f"{m:+.2f}", (hi, yi + dy), xytext=(5, 0),
                        textcoords="offset points", fontsize=7, va="center",
                        color=INK, fontweight="bold" if face else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(y.min() - 0.55, y.max() + 0.55)
    ax.set_xlabel("difference in pLDDT, paired within protein, budget and seed")
    tidy(ax, "x")
    h = [plt.Line2D([], [], color=INK2, marker="o", linewidth=2, markersize=6,
                    markeredgecolor="white", label="as measured"),
         plt.Line2D([], [], color=INK2, marker="o", linewidth=2, markersize=6,
                    markerfacecolor="white", label="entropy-adjusted")]
    ax.legend(handles=h, frameon=False, loc="upper right", fontsize=6.8)
    ax.set_title("Removing the entropy gap removes the result\n"
                 f"adjustment subtracts {SLOPE:.1f} pLDDT per nat of entropy difference",
                 loc="left", fontsize=8.5, pad=6)
    save(fig, "fig8_plddt_adjusted")


# ---------------------------------------------------------------- fig 9
def fig9_bound_vs_quality():
    from scipy.stats import spearmanr
    bound = defaultdict(dict)
    for l in open(RES / "bound_kuniform.jsonl"):
        r = json.loads(l)
        bound[r["name"]].setdefault(r["strategy"], []).append(r["nll_per_residue"])
    bound = {n: {s: float(np.mean(v)) for s, v in d.items()} for n, d in bound.items()}

    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    rng = np.random.default_rng(0)
    for i, (met, lab) in enumerate((("plddt", "pLDDT"), ("recovery", "recovery"))):
        q = defaultdict(dict)
        for r in rows:
            q[r["name"]].setdefault(r["strategy"], []).append(r[met])
        rhos = []
        for n, d in q.items():
            ss = [s for s in d if s in bound.get(n, {})]
            if len(ss) < 4:
                continue
            rho = spearmanr([bound[n][s] for s in ss],
                            [float(np.mean(d[s])) for s in ss]).statistic
            if rho == rho:
                rhos.append(float(rho))
        rhos = np.array(rhos)
        yj = i + rng.uniform(-0.11, 0.11, len(rhos))
        ax.scatter(rhos, yj, s=16, color=MUTED, alpha=0.65, linewidths=0, zorder=3)
        m, lo, hi = boot(rhos)
        ax.plot([lo, hi], [i, i], color=BLUE, linewidth=2.4,
                solid_capstyle="round", zorder=4)
        ax.plot([m], [i], "o", color=BLUE, markersize=8,
                markeredgecolor="white", markeredgewidth=1.6, zorder=5)
        # above the row, not beside it: the 30 jittered points reach past the
        # interval on both sides and the text sat on top of them.
        ax.annotate(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]", (m, i), xytext=(0, 17),
                    textcoords="offset points", fontsize=7.2, va="center",
                    ha="center", color=INK, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                              alpha=0.92))
        rec("fig9", "", f"bound vs {met}, mean Spearman rho", m, lo, hi,
            f"{len(rhos)} proteins, over the 8 strategies within each")
    ax.axvline(0, color=INK, linewidth=1.0, zorder=2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["pLDDT", "recovery"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(-1.35, 1.5)
    ax.set_xlabel("Spearman $\\rho$ between bound and sample quality, within protein")
    ax.annotate("$\\leftarrow$ hypothesis: tighter bound, better samples",
                (0.01, 0.97), xycoords="axes fraction", fontsize=6.8,
                color=INK2, va="top")
    tidy(ax, "x")
    ax.set_title("Does a tighter bound give better samples?  No.\n"
                 "one point per protein, 30 domains",
                 loc="left", fontsize=8.5, pad=6)
    save(fig, "fig9_bound_vs_quality")


if __name__ == "__main__":
    print("drawing into", OUT)
    fig6_quality_vs_nfe()
    fig7_confound()
    fig8_adjusted_contrasts()
    fig9_bound_vs_quality()
    with open(OUT / "figure_data_samples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["figure", "panel", "label", "value",
                                          "ci_low", "ci_high", "note"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"  wrote figure_data_samples.csv ({len(ROWS)} rows)")
