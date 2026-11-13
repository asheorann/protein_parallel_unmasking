# Yan: masked diffusion for proteins — two ideas tested, one that works

Three lines of work on improving masked diffusion models for protein
sequence design. Two were proposals and both fail as written; the third grew
out of the second's machinery used a different way, and it is the one result
that survived every control.

| folder | the claim | verdict |
|---|---|---|
| [`idea_a_contacts/`](idea_a_contacts/) | choose the parallel unmasking set by contact couplings | **negative.** A deposited X-ray contact map does not beat free confidence ordering (+1.10 pLDDT, n.s.); inside the decoder DPLM ships the rule costs −10.73 |
| [`idea_b_two_level/`](idea_b_two_level/) | write six chemical families before exact identities | **negative.** 6–8 pLDDT below a flat baseline, and chemistry beats a bit-matched random partition by 1.3 with overlapping intervals |
| [`conditioning/`](conditioning/) | hand the model a real protein's chemical pattern instead of letting it invent one | **positive.** +23.4 pLDDT over the random control on a homology-purged corpus, and the sequences adopt the source domain's fold (Ref-TM 0.596, 64.6 % above the same-fold threshold) |

The third row uses the second row's model — same architecture, same training
run, same checkpoint — differing only in whether the family pattern is
generated or supplied. That is the finding: the family channel fails as a
generative mechanism and works as a control interface. Because the two share
it, the model and its training live in [`core/`](core/) rather than in either
folder.

## Quickstart

Run everything from the repository root. The scripts resolve their own
location, so they work wherever the repo is cloned.

```bash
# 1. setup
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r Yan/requirements.txt

# 2. model weights (DPLM, ESM-2, ESMFold; several GB, into $HF_HOME)
bash Yan/core/run/download.sh

# 3. smoke test on one GPU: does DPLM generate, does the contact head carry
#    signal under masking, does ESMFold score
bash Yan/core/run/smoke.sh

# 4. the corpus, cached as plain text (needs the SwissProt-Pfam HF dataset)
python Yan/core/tools/cache_swissprot.py

# 5. the shared checkpoints that Idea B and conditioning both use
bash Yan/core/run/train_all.sh
```

Then any folder, in the order its own README gives. The headline result is:

```bash
bash Yan/conditioning/run/clustered_split.sh    # blastp homology purge
bash Yan/conditioning/run/retrain_clustered.sh  # retrain all three arms purged
bash Yan/conditioning/run/cond_clustered.sh     # the 2x2
bash Yan/conditioning/run/ref_tm.sh             # the fold, vs deposited coordinates
python -m conditioning.analysis.cross_dataset   # run from inside Yan/
```

The analysis modules are imported as `core.*`, `conditioning.*` and
`idea_a_contacts.*`, so run them from **inside** `Yan/`:

```bash
cd Yan && python -m conditioning.analysis.figures_cond
```

| variable | default | what it does |
|---|---|---|
| `PY` | `python` | the interpreter the run scripts use |
| `HF_HOME` | `$HOME/.cache/huggingface` | where model weights and datasets are cached |

Scripts carry `#SBATCH` headers and were run on Slurm (A100, `gpuq`). Under
plain `bash` those lines are comments and the script runs in the foreground.
Submitting with `sbatch` writes job output to `logs/`, relative to the
directory you submit from.

## What is and is not in this copy

Included: all code, all run scripts, the per-folder READMEs, and the figures.

Not included: the generated sequences and scored metrics (32 MB of JSON), the
22 per-finding notes, the five write-ups, the trained checkpoints (1.8 GB),
and the input data (613 MB — SwissProt, CATH domains, deposited structures).
Those live in the CIS6270 project repository under `project2/`. The analysis
scripts read from `*/results/`, which is empty here, so regenerate with the
run scripts above or copy the result files across.

## Figures

| figure | folder | what it shows |
|---|---|---|
| `fig1_contact_quality` | Idea A | SUPERSEDED (atom14 indexing bug): contact precision against mask rate |
| `fig2_plddt_vs_nfe` | Idea A | pLDDT against forward passes, all selectors |
| `fig3_contrasts` | Idea A | the paired contrasts against confidence |
| `fig5_oracle` | Idea A | the oracle test, on deposited maps |
| `fig4_ideaB`, `fig6_ideaB` | Idea B | the unconditional two-level arms and the quality/diversity frontier |
| `fig7_conditioning` | conditioning | the contrast and its control, before and after the homology purge, on both datasets |
| `fig8_pattern_budget` | conditioning | the accuracy budget against what a family head achieves |
| `fig9_independent_metrics` | conditioning | FPD/MMD, Max-ID, NSR and Ref-TM |

`fig1` and `fig5` rest on numbers that two later correction rounds retired.

## Two things worth borrowing regardless of the ideas

**The firing-fraction diagnostic.** In a gated sampler, harm was linear in how
often the gate fires: `harm = −21.02 × fired_fraction + 1.99`, r² = 0.991,
extrapolating to zero when the rule never fires. That distinguishes "my
schedule fixed the mechanism" from "my schedule disabled it", and it is what
showed a proposed warm-up to be a dose control rather than a repair.

**The power floor.** Per-sequence pLDDT standard deviation is 17.8, so a
48-sequence cell resolves only effects above about 7 pLDDT, while the largest
published gain for unmasking-order selection on this model is 1.65 and needs
roughly 913 sequences per arm. A 12-cell sweep reports a +7 maximum under a
true null 27 % of the time — so never quote a sweep's best cell. Three of our
own findings died when held to this.
