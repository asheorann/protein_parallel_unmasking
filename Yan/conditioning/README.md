# Conditioning on a real protein's chemical pattern

> **Copy for the shared repo.** This folder carries code, run scripts and
> figures only. The generated sequences, scored metrics and per-finding notes
> under `results/`, and the five write-ups under `reports/`, are not included;
> they live in the CIS6270 project repository under `project2/`. Names below
> beginning `results/` or `reports/` refer to files there — the run scripts
> write their own output into the empty `results/` directory here.

**The result.** Hand the two-level model the six-class chemical pattern read
off a real protein, instead of letting it generate one, and it produces
sequences that are not copies and that **fold like the protein whose pattern
they were given**. This is the one positive finding in project 2, and the only
one that survived its own control, a retrieval check and a homology purge.

It uses Idea B's model (`../core/two_level/`) and inverts the conclusion: the
family channel fails as a *generative mechanism* and works as a *control
interface*. The difference is where the pattern comes from.

## The numbers

Chemical minus a bit-matched random partition, paired per source, 128 sources:

| pattern source | original corpus | homologs purged | what memorisation was worth |
|---|---|---|---|
| SwissProt dev | +26.09 [+22.89, +29.28] | **+23.38 [+20.07, +26.68]** | +2.71 [+0.95, +4.55] |
| CATH S35 | +21.06 [+17.86, +24.24] | **+19.85 [+16.62, +23.05]** | +1.21 [−0.68, +3.06] |

The intervals overlap, so the effect is protein chemistry rather than a
property of the SwissProt split. Quote **+23.4** and **+19.9**, never the
+26.4 of the first run.

**The structural endpoint** (CATH S35, purged). Each generated sequence
folded and aligned to the crystal structure of the domain whose pattern it
was given:

| arm | Ref-TM | TM ≥ 0.5 | RMSD |
|---|---|---|---|
| the source's own slice (ceiling) | 0.861 | 95.3 % | 1.7 Å |
| **chemical, conditioned** | **0.596 [0.558, 0.635]** | **64.6 %** | 3.8 Å |
| random partition, conditioned | 0.299 | 0.8 % | 5.6 Å |
| either unconditional arm | 0.288–0.289 | 0.0 % | 5.6–5.7 Å |

Three controls hold: the bit-matched random partition sits at the
unrelated-pair background of 0.29; permuting which source each structure is
compared against collapses the chemical arm to 0.303, so it is the *source's*
fold and not a generic domain shape; and it is not sequence identity, because
the random arm recovers *more* native residues (NSR 0.393 against 0.348) and
still scores background.

**Measured by instruments that are not ESMFold.** FPD in ESM-2 embedding
space falls 27.7 → 12.0 against 26.1 for the random control, so the chemical
pattern closes 64 % (SwissProt) and 60 % (CATH) of the distance from the
unconditional model to natural proteins while its control closes 5 % and
0.3 %. Max-ID reproduces the purge by alignment: 32 % identity to the
original corpus, 11.5 % — the unconditional value — after it, while keeping
+24.6 pLDDT. Pseudo-perplexity is retired as a discriminator; it favours
chemical on one dataset and random on the other. scTM/scRMSD are **not run**:
no inverse-folding model is available, declared rather than approximated.

## The limit, which is the other half of the result

A pattern only helps while it is nearly exact. Conditioning on a real one is
worth +18.8 pLDDT; corrupting it costs **−1.15 pLDDT per 1 % of error**, so
7.8 % error halves the gain and 18.6 % erases it. No predictor comes close: a
family head trained for the job sits at 63.7 % error and at its
majority-class floor (0.3634 against 0.3633) no matter how much context it is
given, and two independent corrections to its loss improved validation by
1.0–1.25 nats while moving accuracy not at all. Feeding the model its *own*
generated pattern through the identical path is worth +0.29.

So coarse-to-fine cannot bootstrap itself here, and the pattern is useful
precisely because it is exact and supplied from outside. That is what makes
this a controllability result rather than a better generative model.

## Layout

| path | what |
|---|---|
| `patterns/clean_patterns.py` | selects pattern sources by 10-mer containment. Superseded as a homology filter — containment is a substring test — but kept because the published +20.22 came from it |
| `patterns/cath_patterns.py` | the CATH S35 source set: structural domains, non-redundant at 35 %, a genuinely different protein distribution |
| `patterns/fetch_cath_domains.py` | downloads each source's deposited structure, which is what makes Ref-TM possible |
| `purge/export_fasta.py` | writes the corpus and the dev candidates as FASTA — exactly the sequences the model trained on |
| `purge/build_clustered_split.py` | the homology purge: blastp every source against the corpus and delete every relative at E ≤ 1e-3. 19,645 sequences (11.89 %) on SwissProt, 6,640 (4.02 %) on CATH |
| `metrics.py` | the non-ESMFold metrics: FPD, MMD, NSR with its uniform and modal floors, composition floors |
| `experiments/run_pattern_curve.py` | the degradation curve and the family-head accuracy test — the budget above |
| `experiments/extra_metrics_run.py` | re-scores the **stored** sequences. Nothing is generated or re-folded, so it asks a new question of the same objects |
| `experiments/max_id.py` | highest identity to the corpus each arm actually trained on, by blastp, scored over the query |
| `experiments/ref_tm.py` | Ref-TM and Ref-RMSD against the source's crystal structure. Run with `-I`: the structure files are downloaded data and must never be on the import path |
| `analysis/clustered.py` | the 2×2 — chemical/random × original/purged — and the difference of differences |
| `analysis/cross_dataset.py` | SwissProt beside CATH |
| `analysis/pattern_curve.py` | required accuracy against achieved accuracy |
| `analysis/extra_metrics.py` | one table per dataset: what ESMFold says beside what the independent instruments say |
| `analysis/clean_cond.py` | the first, superseded containment-filtered comparison |
| `analysis/figures_cond.py` | `fig7`–`fig9` |

Result notes, in the order the work happened: `results/B_conditioning_note.md` (the
first run and its leakage caveat) → `results/G_clean_conditioning_note.md` (the
containment filter) → `results/J_clustered_split_note.md` (the purge, which showed
containment was measuring the wrong thing) → `results/K_cath_dataset_note.md` (the
replication) → `results/I_pattern_curve_note.md` (the budget) →
`results/L_extra_metrics_note.md` (the independent instruments) →
`results/M_reftm_note.md` (the fold).

## Order

Needs the shared checkpoints from `core/run/train_all.sh`.

```bash
sbatch conditioning/run/clustered_split.sh    # blastp purge, SwissProt
sbatch conditioning/run/retrain_clustered.sh  # retrain all three arms on the purged corpus
sbatch conditioning/run/cond_clustered.sh     # the 2x2
sbatch conditioning/run/cath_split.sh         # the same for CATH S35
sbatch conditioning/run/retrain_cath.sh
sbatch conditioning/run/cond_cath.sh
sbatch conditioning/run/pattern_curve.sh      # the degradation curve and the budget
sbatch conditioning/run/family_signal.sh      # can the model predict the pattern itself
sbatch conditioning/run/extra_metrics.sh      # FPD, MMD, NSR, Max-ID
sbatch conditioning/run/ref_tm.sh             # the fold, against deposited coordinates
sbatch conditioning/run/ref_tm_perm.sh        # the permuted-pairing control
python -m conditioning.analysis.cross_dataset
python -m conditioning.analysis.figures_cond
```

The retrain steps are the expensive part and are what make the purge a
measurement rather than an argument: identical hyperparameters, identical
seed, identical architecture, differing only in the corpus.

## What is still open

A hand-written or consensus pattern, rather than one read off a real protein,
and an independent structure predictor. The generated structures are ESMFold
predictions and ESMFold's own training was not purged — it has seen these
CATH domains — so only the reference is experimental. The random-partition
control bounds how much of the result that can explain, since ESMFold knows
the source for those sequences too and they have higher identity to it.
