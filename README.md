# blocks: non-contiguous block diffusion for protein sequences

Everything for this idea lives in this folder. Run all commands from
`project2/` (the parent), so that `blocks` is importable.

## The claim

Block diffusion keeps the exact chain rule across blocks and uses masked
diffusion inside them. Its correctness argument is that the rate matrix acts
only on the current block while the prefix is a fixed condition, so every fact
of the masked diffusion framework holds block by block. **That argument never
uses adjacency.** Contiguous windows are a convention inherited from text,
where reading runs left to right.

Since the chain rule is exact for any ordered partition, `log p(x)` does not
depend on the partition at all, and every difference between partitions is
slack in the within-block diffusion bound. So the question is sharp: which
partitions minimise that slack, and does the tightest bound give the best
samples?

The mechanism we test is **within-block coupling mass**, the fraction of a
protein's true contacts that end up inside a block, where a factorized
posterior has to handle them, rather than across blocks, where the exact chain
rule does. Prediction: slack rises with coupling mass.

Nothing is trained. Everything runs on a frozen protein masked LM. And because
we evaluate on natural proteins with solved structures, the secondary
structure, burial and contact map are ground truth rather than predictions.

## Quickstart

```bash
cd project2

pip install -r blocks/requirements.txt

bash blocks/run/smoke.sh                 # self-test, no data, no downloads
python blocks/get_data.py                # ~40 small proteins from RCSB
bash blocks/run/bound.sh                 # the main experiment
bash blocks/run/figures.sh               # figures + tables
```

`bound.sh` downloads ESM-2 150M on first run (about 600 MB), then takes a few
minutes on a GPU. Set `DEVICE=cpu` if there is no GPU.

Optional and much more expensive:

```bash
FOLD=0 bash blocks/run/sample.sh    # generation + recovery, no folding
FOLD=1 bash blocks/run/sample.sh    # adds ESMFold pLDDT
bash blocks/run/order.sh            # ordering ablation
bash blocks/run/figures.sh
```

## Output

Written under this folder, and gitignored:

- `blocks/results/*.jsonl` raw per-protein records
- `blocks/figures/fig1_bound_by_strategy.pdf`
- `blocks/figures/fig2_mechanism.pdf` bound against coupling mass, the figure
  that carries the argument
- `blocks/figures/fig3_recovery_vs_nfe.pdf`, `fig4_plddt_vs_nfe.pdf`
- `blocks/figures/table_bound.md`, `table_quality.md` for pasting into the paper

## The eight partitions

| strategy | what it is | role |
|---|---|---|
| `contiguous` | adjacent windows | **BD3-LM exactly** |
| `single` | one block | **MDLM exactly** |
| `autoregressive` | blocks of size 1 | left-to-right AR |
| `random` | size-matched random partition | controls for "any non-adjacency" |
| `burial` | buried core first, then surface | structure-derived |
| `sse` | one block per secondary structure element | structure-derived |
| `anticluster` | greedy colouring, minimises within-block contacts | predicted best |
| `cluster` | spectral clustering, maximises within-block contacts | predicted worst |

Two baselines are exact special cases of the method itself, which is the
cleanest internal control available. `cluster` and `anticluster` bracket the
coupling range, so the main result is a regression rather than a bar chart. On
a synthetic 64-residue graph, coupling mass ran anticluster 0.00 < random 0.14
< contiguous 0.22 < burial 0.23 < sse 0.39 < cluster 0.46 < single 1.00. Real
contact maps are denser, so the low end will not be exactly zero.

## Knobs

| variable | default | notes |
|---|---|---|
| `MODEL` | `facebook/esm2_t30_150M_UR50D` | `esm2_t33_650M_UR50D` on a bigger GPU |
| `PDB_DIR` | `blocks/data/pdb` | any directory of `.pdb` or `.cif` |
| `DEVICE` | `cuda` | `cpu` works, slowly |
| `LIMIT` | 100 | number of proteins |
| `BLOCK` | 16 | target block size, fixed across strategies |
| `NT` | 8 | Monte Carlo time samples per block |
| `FOLD` | 1 | 0 skips ESMFold |

## Correctness check

`bound.sh` passes `--check`, which asserts the estimator reduces to the
autoregressive NLL at block size one. For a single-position block the loss
fires with probability `t`, the weight is `1/t`, and the state does not depend
on `t`, so the integral collapses to `-log d`. If that check fails there is a
bug in the estimator and the numbers mean nothing. It belongs in the appendix
as a rate-legality sanity test.

## Files

```
blocks/
  structure.py      PDB -> sequence, secondary structure, burial, contact map
  partitions.py     the eight strategies + coupling-mass diagnostic
  model.py          frozen denoiser wrapper, NFE counting
  bound.py          bound estimator + exactness check
  sample.py         block-wise sampler with matched budgets
  evaluate.py       recovery, diversity, ESMFold
  plots.py          figures
  cli.py            bound | sample | fold | figures | smoke
  get_data.py       structure downloader
  requirements.txt
  run/              smoke.sh bound.sh sample.sh figures.sh order.sh
```

## Known weaknesses, to state in the paper rather than discover in Q&A

- **MARD-Mol** (PMC13489711) raises granularity from atoms to motif units and
  couples AR planning across units with diffusion inside them, which is
  structurally a non-contiguous block decomposition for molecules. Needs a
  close read. If it covers the molecule case, what remains here is the protein
  case plus the bound-tightness analysis, which it does not do.
- **MolHIT** (2602.17602) embeds a hierarchy in the forward transition matrix.
  Different axis, over the alphabet rather than over positions, but cite and
  distinguish it.
- **The backbone is not a trained block model.** BD3-LM trains with
  block-causal attention; we apply a block decomposition at inference to a
  bidirectionally trained denoiser. The bound is still valid for that model,
  but it is not the bound BD3-LM reports, and the paper must say so.
- **ESMFold shares a trunk with ESM-2**, so pLDDT is partially circular when
  the denoiser is an ESM-2 derivative. Recovery against the native sequence is
  the primary metric for that reason.
- **Greedy colouring is a heuristic** for balanced k-way max-cut, which is
  NP-hard. No optimality is claimed.
