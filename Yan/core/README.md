# core: what more than one line of work depends on

> **Copy for the shared repo.** This folder carries code, run scripts and
> figures only. The generated sequences, scored metrics and per-finding notes
> under `results/`, and the five write-ups under `reports/`, are not included;
> they live in the CIS6270 project repository under `project2/`. Names below
> beginning `results/` or `reports/` refer to files there — the run scripts
> write their own output into the empty `results/` directory here.

Nothing here belongs to a single idea. A file is in `core/` only because two
or three of `idea_a_contacts/`, `idea_b_two_level/` and `conditioning/` use
it, so putting it in any one of them would make the others import from it.

The largest case is `two_level/`. The two-level masked diffusion model is
Idea B's proposed mechanism *and* the interface the conditioning result runs
through — same architecture, same training run, same checkpoint
(`ckpt/dms6.pt`), differing only in whether the family pattern is generated
or supplied. Idea B is refuted and conditioning is the project's one positive
result, so neither can own the other's model. It lives here instead.

## Files

| file | what it is |
|---|---|
| `alphabet.py` | the 20 residues, the six chemical families, hydrophobicity scales. Every partition the project uses is defined here, including the bit-matched random controls |
| `data.py` | the SwissProt corpus and its length-filtered split, read from `data/swissprot_*.txt` |
| `metrics.py` | ESMFold pLDDT/pTM/pAE and ESM-2 pseudo-perplexity, plus the Idea-B periodicity metrics. **Holds the pTM fix**: HF's `ptm` is a batch maximum, so it is recomputed per sequence from `ptm_logits` |
| `run_score.py` | the scoring driver. Takes any generation file with a `cells` list, folds every cell, frees ESMFold, then loads ESM-2, so peak memory stays near 10 GB |
| `paths.py` | which folder owns a result or figure file, now that each idea has its own. Used by the scripts that span ideas |
| `two_level/model.py` | the flat and two-level masked diffusion models, one backbone and one parameter budget so only the state space differs |
| `two_level/train.py` | trains one arm. `--arm flat`, or `--arm hier --partition dms6`, or `--random-partition` for the control |
| `two_level/sample.py` | generation at matched forward-pass budget. `sample_hier_conditioned` is the conditioning entry point; sampling is done in float64 because float32 Gumbel silently lowers the effective temperature |
| `two_level/run_generate.py` | the generation driver for both lines: unconditional and temperature grids for Idea B, `--conditioned --pattern-file` for conditioning |
| `analysis/report.py` | paired contrasts with bootstrap intervals. The verdicts rest on these, not on raw means |
| `analysis/final.py` | assembles `reports/RESULTS.md` from every scored file in all three folders |
| `analysis/figures.py` | `fig1`–`fig6`, and the eight-slot palette that `conditioning/analysis/figures_cond.py` reuses so the two sets match |
| `tools/cache_swissprot.py` | builds the plain-text corpus from the HF parquet. Run once |
| `tools/verify_ptm.py` | proves per-sequence pTM is batch-invariant after the `metrics.py` fix |
| `run/smoke.py` | the end-to-end feasibility gate: does DPLM generate, does the contact head carry signal under masking, does ESMFold score |

## Run scripts

| script | what |
|---|---|
| `run/download.sh` | model weights. The environment-bootstrap scripts that built this project's venv are omitted from this copy; `../requirements.txt` replaces them |
| `run/smoke.sh` | the feasibility gate, one GPU |
| `run/train_all.sh` | the shared checkpoints, one arm per array task: `flat`, `dms6`, `rand6`, and the partition variants |
| `run/rescore_ptm.sh`, `run/rescore_ptm2.sh` | rescore after the pTM fix. Every pTM and foldability number predating this is invalid; pLDDT and pAE were never affected |

## Order

```bash
sbatch core/run/download.sh
sbatch core/run/smoke.sh
sbatch core/run/train_all.sh
```

`run/train_all.sh` produces the checkpoints both `idea_b_two_level/` and
`conditioning/` need. Two scripts run that same training again on
homology-purged corpora — `../conditioning/run/retrain_clustered.sh` and
`../conditioning/run/retrain_cath.sh` — and the corpus is the only
difference between the two sets of checkpoints.
