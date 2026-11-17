"""Protein-level check of two Experiment 2 statements.

Reuses the definitions in figures_samples.py (sequence entropy, the within-cell
slope of pLDDT on entropy, the per-protein Spearman correlation between bound
and sample quality) and changes one thing: intervals resample the 30 proteins,
not the 750 protein x budget x seed cells that fig8 resamples, which are 25 per
protein and so not independent. Also repeats the bound-vs-quality correlation
with entropy-adjusted pLDDT. Reads folded.jsonl and bound_kuniform.jsonl and
writes nothing.

    python blocks/results/slurm/check_exp2_protein_level.py [results_dir]
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

RES = sys.argv[1] if len(sys.argv) > 1 else str(
    Path(__file__).resolve().parents[3] / "blocks" / "results")
rows = [json.loads(l) for l in open(f"{RES}/folded.jsonl") if l.strip()]

def entropy(s):
    c = Counter(s); p = np.array([v / len(s) for v in c.values()])
    return float(-(p * np.log(p)).sum())

def boot(x, n=20000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

for r in rows:
    r["entropy"] = entropy(r["sequence"])
CELLS = defaultdict(list)
for r in rows:
    CELLS[(r["name"], r["steps_requested"], r["seed"])].append(r)
_p, _e = [], []
for c, rs in CELLS.items():
    p = np.array([x["plddt"] for x in rs]); e = np.array([x["entropy"] for x in rs])
    _p += list(p - p.mean()); _e += list(e - e.mean())
SLOPE = float(np.polyfit(_e, _p, 1)[0])
print(f"within-cell slope {SLOPE:+.2f} pLDDT/nat, n cells {len(CELLS)}, proteins {len({r['name'] for r in rows})}")
print(f"mean pLDDT {np.mean([r['plddt'] for r in rows]):.1f}, frac >70 {np.mean([r['plddt']>70 for r in rows]):.3f}, mean recovery {np.mean([r['recovery'] for r in rows]):.3f}")
for r in rows:
    r["plddt_adj"] = r["plddt"] - SLOPE * r["entropy"]

bound = defaultdict(dict)
for l in open(f"{RES}/bound_kuniform.jsonl"):
    r = json.loads(l)
    bound[r["name"]].setdefault(r["strategy"], []).append(r["nll_per_residue"])
bound = {n: {s: float(np.mean(v)) for s, v in d.items()} for n, d in bound.items()}

print("\n== per-protein Spearman, bound vs sample quality (negative = tighter bound, better samples) ==")
for excl, tag in (((), "all 8 strategies"), (("autoregressive",), "without autoregressive"),
                  (("autoregressive", "single"), "6 multi-block partitions")):
    for met in ("plddt", "plddt_adj", "recovery"):
        q = defaultdict(dict)
        for r in rows:
            if r["strategy"] in excl: continue
            q[r["name"]].setdefault(r["strategy"], []).append(r[met])
        rhos = []
        for n, d in q.items():
            ss = [s for s in d if s in bound.get(n, {})]
            rho = spearmanr([bound[n][s] for s in ss], [float(np.mean(d[s])) for s in ss]).statistic
            if rho == rho: rhos.append(float(rho))
        m, lo, hi = boot(rhos)
        print(f"  {tag:26s} {met:10s} rho {m:+.3f} [{lo:+.3f},{hi:+.3f}]  n={len(rhos)}  negative on {int(np.sum(np.array(rhos)<0))}")

print("\n== paired pLDDT contrasts: cell bootstrap (as in fig8, 750 cells) vs protein bootstrap (30 proteins) ==")
for a, b in (("cluster","anticluster"),("contiguous","anticluster"),("cluster","contiguous"),
             ("random","anticluster"),("sse","contiguous"),("autoregressive","contiguous")):
    for met in ("plddt", "plddt_adj"):
        cell, byprot = [], defaultdict(list)
        for c, rs in CELLS.items():
            m = {r["strategy"]: r for r in rs}
            if a in m and b in m:
                d = m[a][met] - m[b][met]; cell.append(d); byprot[c[0]].append(d)
        pm = [float(np.mean(v)) for v in byprot.values()]
        c0 = boot(cell); p0 = boot(pm)
        print(f"  {a+' - '+b:28s} {met:10s} {c0[0]:+.2f}  cell CI [{c0[1]:+.2f},{c0[2]:+.2f}]  protein CI [{p0[1]:+.2f},{p0[2]:+.2f}]  {a} higher on {int(np.sum(np.array(pm)>0))}/{len(pm)} proteins")
