# blocks: non-contiguous block diffusion for protein sequences

Anush's idea for CIS 6270 Project 2. See the root README for quickstart.

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
coupling range, so the main result is a regression rather than a bar chart.

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
bug in the estimator and the numbers mean nothing.

## Files

```
structure.py      PDB -> sequence, secondary structure, burial, contact map
partitions.py     the eight strategies + coupling-mass diagnostic
model.py          frozen denoiser wrapper, NFE counting
bound.py          bound estimator + exactness check
sample.py         block-wise sampler with matched budgets
evaluate.py       recovery, diversity, ESMFold
plots.py          figures
cli.py            bound | sample | fold | figures | smoke
get_data.py       RCSB structure downloader
get_cath.py       CATH S35 domain downloader (matches Yan/Neha's data)
requirements.txt
run/              smoke.sh bound.sh sample.sh figures.sh order.sh
```

## Known weaknesses

- **MARD-Mol** (PMC13489711) raises granularity from atoms to motif units and
  couples AR planning across units with diffusion inside them, which is
  structurally a non-contiguous block decomposition for molecules.
- **MolHIT** (2602.17602) embeds a hierarchy in the forward transition matrix.
  Different axis, over the alphabet rather than over positions, but cite and
  distinguish it.
- **The backbone is not a trained block model.** BD3-LM trains with
  block-causal attention; we apply a block decomposition at inference to a
  bidirectionally trained denoiser. The bound is still valid for that model,
  but it is not the bound BD3-LM reports.
- **ESMFold shares a trunk with ESM-2**, so pLDDT is partially circular when
  the denoiser is an ESM-2 derivative. Recovery against the native sequence is
  the primary metric for that reason.
- **Greedy colouring is a heuristic** for balanced k-way max-cut, which is
  NP-hard. No optimality is claimed.
