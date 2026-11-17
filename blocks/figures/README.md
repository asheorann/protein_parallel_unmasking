# Figures for the blocks/ results

Drawn 2026-11-15 by `blocks/results/slurm/figures_blocks.py`. PNG (300 dpi) and
vector PDF, sized for a two-column page. Every plotted number is in
`figure_data.csv`. All runs use the frozen `facebook/esm2_t30_150M_UR50D`
denoiser; nothing is trained.

**These numbers come from the fixed code (commit `1781db2`).** Anything
measured before that fix is not comparable: `bound.sh --check` crashed outright
on numpy 2.x, and when it did run the forced-mask estimator overstated the bound
by about 8x.

| file | what it shows |
|---|---|
| `fig1_estimator_validation` | the estimator is now unbiased, and which variant to use |
| `fig2_sign_flip` | why the shipped sampling defaults cannot support a conclusion |
| `fig3_bound_by_strategy` | the headline comparison, with the overlap visible |
| `fig4_mechanism` | coupling mass vs bound, as measured and within protein |
| `fig5_paired_contrasts` | the paired contrasts, by seed (revised 2026-11-16) |
| `fig6_quality_vs_nfe` | Experiment 2: sample quality against measured compute |
| `fig7_degeneracy_confound` | why the pLDDT ranking is not a ranking of partitions |
| `fig8_plddt_adjusted` | the pLDDT contrasts before and after that confound |
| `fig9_bound_vs_quality` | does a tighter bound give better samples? |

Figures 1–5 are Experiment 1 (the bound), drawn by `figures_blocks.py` into
`figure_data.csv`. Figures 6–9 are Experiment 2 (generation), drawn by
`figures_samples.py` into `figure_data_samples.csv`.

## Suggested captions

**Fig 1 — Estimator validation (6 CATH domains).** At block size one the
estimator must return the exact autoregressive NLL. The forced-mask form keeps
the `1/t` weight while making the mask indicator fire with probability one, and
overstates it by **7.86x** (median; range 6.4–11.6). Masking Bernoulli(t) and
letting an empty draw contribute zero gives **1.03x**; drawing the masked-set
size `k ~ U{1..n}` with weight `n/k` is **exact, with zero variance**. Panel B:
spread over repeat seeds on contiguous blocks of 16. k-uniform is 4–5x tighter
than Bernoulli at matched cost, and the forced form gets *worse* with more
samples (0.302 → 0.529) because extra strata give more chances to draw the
tiny-`t`, huge-weight term.

**Fig 2 — The contrast changes sign with the sampling budget.** `cluster −
anticluster`, paired within protein, 100 CATH S35 domains. The hypothesis
predicts a positive value. At the shipped defaults it is **−0.086**
[−0.145, −0.024]; at `n_t`=32 / 32 blocks, **+0.046** [+0.013, +0.084]; with the
k-uniform estimator, **+0.019** [+0.003, +0.034]. The bootstrap resamples
*proteins*, so it never sees the Monte-Carlo error of the estimator, which here
is the dominant term — it reports a confident interval around whichever way the
noise fell. Do not quote a contrast computed at `NT=8 MAXB=16`.

**Fig 3 — Bound by partition strategy** (k-uniform, 100 domains, 2 seeds).
Every marginal interval overlaps every other. On this evidence no strategy is
separated from any other, which is why the paired design in Fig 5 is the one
that can answer the question. `single` (MDLM) and `autoregressive` are exact
special cases of the method, not competitors on slack. The `autoregressive`
value is itself an estimate: with `--max-blocks 32` it is scaled up from 32 of
the n positions, unbiased but with an sd of 0.145 per domain and seed.

**Fig 4 — Does slack rise with within-block coupling mass?** Fitted on the six
genuine partitions: `autoregressive` has zero slack *by construction* and sits
at coupling 0, so including it manufactures a positive slope that says nothing
about coupling. (A) As measured, slope **+0.111**, r = **+0.095** — but the
scatter is dominated by which protein a point belongs to, not which partition.
(B) Centring both axes on each protein's own mean leaves the within-protein
relation the hypothesis is actually about: slope **+0.011**, r = **+0.038**.
About nine tenths of the apparent relationship in (A) is a between-protein
confound — proteins whose partitions happen to carry more coupling also have
higher bounds, for reasons unrelated to the mechanism.

**Fig 5 — Paired contrasts** (k-uniform, 100 domains; revised 2026-11-16).
Within-protein differences, so the protein effect cancels. Dot and bar: two-seed
mean with a bootstrap interval over proteins. Ticks: each seed alone. A star
needs the interval to exclude zero in *both* seeds.

- `cluster − contiguous` **+0.054** [+0.036, +0.072], 76 of 100 domains, seeds
  +0.050 / +0.058. Robust.
- `contiguous − anticluster` **−0.035** [−0.054, −0.016], contiguous tighter on
  74 of 100 domains, seeds −0.037 / −0.034. Robust, and the same sign under all
  three estimator settings (−0.047 at the shipped defaults, −0.021 at `n_t`=32).
  **This runs against the hypothesis:** the partition predicted to be tightest
  (coupling 0.004) is looser than plain adjacent windows (coupling 0.331).
- `cluster − anticluster` **+0.019** [+0.003, +0.034] in the predicted
  direction, but only 56 of 100 domains (sign test p = 0.27) and the seed-0
  interval includes zero (+0.014 [−0.002, +0.030]; seed 1 +0.023). Borderline.
- `single − anticluster` **−0.019** is **not** a reliable result: −0.005
  [−0.018, +0.009] in seed 0 and −0.033 [−0.047, −0.019] in seed 1.
- `single − autoregressive` **+0.043**, seeds +0.049 / +0.038.

Why the effects are invisible in Fig 3: the bound varies between proteins with
an sd of about 0.26 nats/residue, against 0.06–0.10 for a paired difference.
Monte-Carlo error is the smaller term: the sd of one estimate is 0.021–0.027
for the multi-block partitions and about 0.022 for a two-seed paired contrast.

Not in the figure, same data: `random − anticluster` is **+0.003**
[−0.013, +0.019], 50 of 100 domains — minimising within-block contacts does no
better than a size-matched random partition. `sse − contiguous` is −0.009
(61 of 100). The order is `sse` 2.650 < `contiguous` 2.659 < `anticluster`
2.695 ≈ `random` 2.698 < `burial` 2.712 ≈ `cluster` 2.713: the two partitions
made of sequence-contiguous segments are the tightest, whatever their coupling.

### Two corrections to the 2026-11-15 version

1. It quoted a "per-protein Monte-Carlo noise floor" of **0.082**. That is the
   sd of the raw seed-0 minus seed-1 difference pooled over all eight
   strategies, and it is carried by `autoregressive`, which is estimated from 32
   of the n positions per seed (sd 0.145 on its own). It is not the noise on any
   partition contrast; those values are in the paragraph above.
2. It called `single − anticluster` a significant counterexample.
   `mechanism_kuniform.py` seeds its generator with the seed number alone, so
   every protein and strategy in a seed shares **one random stream** and a seed
   is not 100 independent replicates. `single` moves by 0.030 between the two
   seeds, in the same direction on 67 of 100 domains (z = 4.8); no multi-block
   strategy shows this (|z| ≤ 1.1, except `cluster` at 2.4). The protein
   bootstrap cannot see that shared error, which is the same blind spot Fig 2
   describes. Any contrast involving `single` needs a rerun with an independent
   stream per protein and more seeds before it is quoted.

## The honest one-line summary (Experiment 1)

Coupling mass is not what orders these partitions. Adjacent-window blocks are
tighter than every non-contiguous partition by 0.035–0.054 nats/residue (1–2 %
of the bound), the low-coupling partition is no better than a random one, and
the predicted `cluster` > `anticluster` gap is +0.019 and borderline. All of it
is detectable only in a paired design, and all of it is for a frozen ESM-2
denoiser that was never trained as a block model.

## Experiment 2: generation (run 2026-11-16)

6,000 sequences: 30 CATH S35 domains (all of them inside the 100 used for the
bound, so the two experiments pair per protein) × 8 partitions × 5 step budgets
× 5 seeds, then every one scored with ESMFold. Nothing is trained and nothing
is conditioned: the denoiser sees only a length and a partition, never the
structure that partition came from.

**Fig 6 — Quality against measured compute.** The x-axis is measured NFE, not
the requested budget: `_allocate_steps` gives every block at least one step and
at most one per residue, so the 128-step setting delivers about 0.91 L calls and
`autoregressive` is a single point at exactly L whatever is requested.
(A) pLDDT sits between 41 and 48 for every partition at every budget; only
3.6 % of all 6,000 sequences clear pLDDT 70. (B) Recovery is **0.060** against a
composition null of **0.060**. Because generation is unconditional, a sample has
no channel through which to match the particular native it is scored against, so
recovery is at its floor by construction. Her README names recovery the primary
metric precisely because pLDDT is circular; as the experiment is built, recovery
cannot rank anything and pLDDT is all that is left.

**Fig 7 — The pLDDT ranking is a degeneracy ranking.** pLDDT falls **9.7** points
per nat of sequence entropy over all 6,000 sequences (r = −0.362) and **10.2**
within one protein, budget and seed (r = −0.439): low-complexity sequences fold
"confidently". The eight partitions line up along that slope. `anticluster` has
the lowest entropy of the genuine partitions (2.164) and the highest pLDDT
(43.96); `contiguous` and `sse` have the highest entropy (2.516, 2.523) and
middling pLDDT; `autoregressive` is the extreme case, entropy 2.196 with a
longest homopolymer run of 140 residues and a duplicate rate of 0.827.

**Fig 8 — Removing the entropy gap removes the result.** Paired within protein,
budget and seed. As measured, `cluster − anticluster` is **−2.29**, which is the
sign her hypothesis predicts: the most-coupled partition folds worse than the
least-coupled one. But anticluster's sequences are 0.295 nats lower in entropy,
and that gap alone predicts −3.02. Subtracting it leaves **+0.73
[−0.10, +1.55]** — straddling zero, with the sign reversed. The same adjustment
turns `contiguous − anticluster` from −1.17 into **+2.44** and cuts
`autoregressive − contiguous` from +5.49 to +2.21. The contrast that
defines the hypothesis therefore does not survive. Two others do, and both put
`contiguous` first: adjusted, it is **+2.44** above `anticluster` and **+1.70**
above `cluster`, and `cluster − contiguous` is negative as measured as well
(−1.12).

*Correction, 2026-11-16 22:10.* This paragraph first ended "no pLDDT contrast
between partitions survives", which was too strong. The figure also bootstraps
the 750 protein × budget × seed cells, 25 per protein, so its intervals are too
narrow. Resampling the 30 proteins instead
(`blocks/results/slurm/check_exp2_protein_level.py`) gives
`cluster − anticluster` +0.73 [−0.50, +1.83], `contiguous − anticluster` +2.44
[+1.19, +3.44] (contiguous higher on 27 of 30 domains) and
`cluster − contiguous` −1.70 [−2.50, −0.98] (cluster higher on 7 of 30); no
conclusion changes at the protein level. The adjustment is a single linear
slope applied to every partition, so read these as "`contiguous` reaches about
the same pLDDT with more diverse sequences" (entropy 2.516 against 2.164 for
`anticluster`), not as a calibrated quality score.

**Fig 9 — Does a tighter bound give better samples? No.** Per protein, the
Spearman correlation across the eight strategies between the bound from
`bound_kuniform.jsonl` and sample quality. For pLDDT **ρ = −0.029
[−0.171, +0.105]**, for recovery **+0.053 [−0.082, +0.183]**. Both straddle
zero. The bound and the samples rank the partitions differently: Experiment 1
makes `contiguous` tightest and `anticluster` looser, Experiment 2 as measured
does the opposite. Once the entropy confound is removed the two agree on one
point only, that `contiguous` is ahead of both `cluster` and `anticluster`
(Fig 8), and the rank correlation over the strategies is still
indistinguishable from zero: with entropy-adjusted pLDDT ρ = −0.132
[−0.300, +0.040] over all eight, −0.073 [−0.239, +0.098] without
`autoregressive`. (This paragraph first said Experiment 2 then "ranks nothing
at all"; corrected 2026-11-16 22:10.)

### What Experiment 2 settles, and what it does not

It settles that **the bound does not predict sample quality here**, which is the
question the experiment exists to ask, and it is a clean negative.

It does not test the coupling hypothesis on generation, because the two metrics
cannot carry that test as built: recovery is pinned to its null by unconditional
generation, and pLDDT is both circular (ESMFold shares a trunk with ESM-2) and
dominated by sequence degeneracy. A version that could test it would need
conditional generation — inpainting from a native context, so recovery has
something to recover — and a complexity control, either entropy-matched
comparisons or a repeat-rejection step in the sampler.

## Reproducing

```bash
# Experiment 1, the bound
python blocks/results/slurm/mechanism_kuniform.py --pdb-dir blocks/data/cath \
    --limit 100 --n-draws 32 --max-blocks 32 --seeds 2
python blocks/results/slurm/figures_blocks.py

# Experiment 2, generation: ~1h35m sampling + ~2h folding on one A100
python -m blocks.cli sample --pdb-dir blocks/data/cath --device cuda \
    --limit 30 --max-length 200 --steps 8 16 32 64 128 --n-seeds 5 \
    --out blocks/results/samples.jsonl
python -m blocks.cli fold --samples blocks/results/samples.jsonl \
    --device cuda --out blocks/results/folded.jsonl
python blocks/results/slurm/check_samples.py blocks/results/samples.jsonl
python blocks/results/slurm/analyse_samples.py
python blocks/results/slurm/figures_samples.py
# protein-level intervals quoted under Fig 8 and Fig 9; prints, writes nothing
python blocks/results/slurm/check_exp2_protein_level.py
```

`check_samples.py` is a gate, not a report: it fails on mask leakage (an
unresolved position decodes to "X" and `Folder.fold` silently rewrites it to
alanine) and on NFE accounting, and prints the degeneracy and null references
that Figures 7 and 6B rest on.

`mechanism_kuniform.py` writes the same JSONL schema as `blocks.cli bound`, so
`python -m blocks.cli figures --bound <file>` reads it unchanged.

## Colour

Slots from a validated reference palette, **not** the Okabe-Ito set in
`blocks/plots.py` — that one fails on a white surface (`#F0E442` sits at 1.29:1
contrast; `#000000`/`#999999` read as achromatic). Sets used here each pass the
all-pairs colourblind and contrast gates: `#2a78d6 #eb6834 #1baf7a` for the three
estimator variants, `#2a78d6`/`#d03b3b` for the sign of a contrast. Strategy
identity is always carried by an axis label or a direct label, never by colour
alone.
