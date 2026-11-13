"""Figures.

Sized and styled for a two-column ICML page: vector PDF, 7pt-equivalent labels,
colourblind-safe palette, distinct markers so the figures survive greyscale
printing. Every function takes a dataframe-like list of dicts and writes one
file.

Figure 2 is the one that carries the argument. Figures 1 and 3 report what
happened; figure 2 tests why.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
})

# Okabe-Ito, colourblind safe
PALETTE = {
    "single": "#000000",
    "mdlm": "#000000",
    "autoregressive": "#E69F00",
    "contiguous": "#56B4E9",
    "random": "#999999",
    "burial": "#009E73",
    "sse": "#F0E442",
    "anticluster": "#0072B2",
    "cluster": "#D55E00",
}
MARKERS = {
    "single": "o", "autoregressive": "s", "contiguous": "^", "random": "x",
    "burial": "D", "sse": "v", "anticluster": "P", "cluster": "*",
}

LABELS = {
    "single": "Single block (MDLM)",
    "autoregressive": "Block size 1 (AR)",
    "contiguous": "Contiguous (BD3-LM)",
    "random": "Random permutation",
    "burial": "Burial (core first)",
    "sse": "Secondary structure",
    "anticluster": "Anti-clustered",
    "cluster": "Spatially clustered",
}


def _group(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return out


def _order(strategies) -> list[str]:
    pref = ["autoregressive", "anticluster", "burial", "sse", "contiguous",
            "random", "cluster", "single"]
    known = [s for s in pref if s in strategies]
    return known + sorted(set(strategies) - set(known))


# --------------------------------------------------------------------------

def plot_bound_by_strategy(rows: list[dict], out: str | Path) -> None:
    """Figure 1: bound per residue for each partition strategy.

    Lower is a tighter bound. Error bars are the standard error across
    proteins, which is the right unit because proteins are the sampling unit.
    """
    groups = _group(rows, "strategy")
    strategies = _order(groups.keys())
    means, sems = [], []
    for s in strategies:
        v = np.array([r["nll_per_residue"] for r in groups[s]])
        means.append(v.mean())
        sems.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    xs = np.arange(len(strategies))
    ax.bar(
        xs, means, yerr=sems, capsize=2,
        color=[PALETTE.get(s, "#666666") for s in strategies],
        edgecolor="black", linewidth=0.5,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS.get(s, s) for s in strategies], rotation=35, ha="right")
    ax.set_ylabel("Bound (nats / residue) $\\downarrow$")
    ax.set_ylim(bottom=min(means) * 0.95 if means else 0)
    fig.savefig(out)
    plt.close(fig)


def plot_mechanism(rows: list[dict], out: str | Path) -> None:
    """Figure 2: bound against within-block coupling mass.

    The hypothesis predicts a positive slope: the more true contacts a
    partition buries inside its blocks, where they must be handled by a
    factorized posterior, the looser the bound. A flat or negative slope
    falsifies the mechanism even if figure 1 shows a difference, and that
    outcome should be reported rather than buried.
    """
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    xs_all, ys_all = [], []

    for s, rs in _group(rows, "strategy").items():
        x = np.array([r["coupling_mass"] for r in rs])
        y = np.array([r["nll_per_residue"] for r in rs])
        xs_all.extend(x)
        ys_all.extend(y)
        ax.scatter(
            x, y, s=14, alpha=0.65,
            color=PALETTE.get(s, "#666666"),
            marker=MARKERS.get(s, "o"),
            label=LABELS.get(s, s), linewidths=0.4, edgecolors="none",
        )

    xs_all, ys_all = np.array(xs_all), np.array(ys_all)
    if len(xs_all) > 2 and xs_all.std() > 1e-9:
        slope, intercept = np.polyfit(xs_all, ys_all, 1)
        grid = np.linspace(xs_all.min(), xs_all.max(), 50)
        r = np.corrcoef(xs_all, ys_all)[0, 1]
        ax.plot(grid, slope * grid + intercept, "k--", linewidth=1.0,
                label=f"fit: slope={slope:.2f}, $r$={r:.2f}")

    ax.set_xlabel("Within-block coupling mass (fraction of native contacts)")
    ax.set_ylabel("Bound (nats / residue) $\\downarrow$")
    ax.legend(frameon=False, loc="best", ncol=1)
    fig.savefig(out)
    plt.close(fig)


def plot_quality_cost(
    rows: list[dict], out: str | Path, metric: str = "recovery_mean",
    ylabel: str = "Sequence recovery $\\uparrow$",
) -> None:
    """Figure 3: sample quality against NFE, one line per partition.

    Budget is compute, not steps. The autoregressive partition has a floor of
    L calls, so it contributes a single point rather than a curve; that is a
    property of the method and should be stated in the caption rather than
    hidden by interpolation.
    """
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for s, rs in _group(rows, "strategy").items():
        pts = sorted((r["nfe_mean"], r[metric], r.get(metric.replace("mean", "sem"), 0.0))
                     for r in rs if metric in r)
        if not pts:
            continue
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        e = [p[2] for p in pts]
        style = dict(color=PALETTE.get(s, "#666666"), marker=MARKERS.get(s, "o"),
                     label=LABELS.get(s, s))
        if len(pts) == 1:
            ax.errorbar(x, y, yerr=e, fmt=MARKERS.get(s, "o"), capsize=2, **{
                k: v for k, v in style.items() if k != "marker"})
        else:
            ax.errorbar(x, y, yerr=e, capsize=2, **style)
    ax.set_xscale("log")
    ax.set_xlabel("NFE (denoiser calls, log scale) $\\downarrow$")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="best")
    fig.savefig(out)
    plt.close(fig)


def write_markdown_table(rows: list[dict], out: str | Path, keys: list[str]) -> None:
    """Dump a table so numbers in the paper can be pasted, not retyped."""
    lines = ["| " + " | ".join(keys) + " |", "|" + "|".join(["---"] * len(keys)) + "|"]
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k, "")
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    Path(out).write_text("\n".join(lines) + "\n")
