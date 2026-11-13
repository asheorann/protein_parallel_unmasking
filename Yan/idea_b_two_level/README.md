# Idea B: the two-level chemical alphabet

> **Copy for the shared repo.** This folder carries code, run scripts and
> figures only. The generated sequences, scored metrics and per-finding notes
> under `results/`, and the five write-ups under `reports/`, are not included;
> they live in the CIS6270 project repository under `project2/`. Names below
> beginning `results/` or `reports/` refer to files there — the run scripts
> write their own output into the empty `results/` directory here.

**The proposal.** Generate a protein in two stages: first write one of six
chemical families at each position, then resolve exact identities inside the
family it was given. Coarse-to-fine should be easier than picking one of
twenty residues at once, and should let the model lay down a hydrophobic
pattern before committing to chemistry.

**Verdict: negative.** The two-level arms score **6–8 pLDDT below a flat
baseline** of the same size, data and budget (43.1 flat against 36.7
chemical and 35.4 random), and the chemical partition beats a bit-matched
*random* partition by only 1.3 with overlapping intervals. The flat
one-dimensional temperature curve dominates the two-level two-dimensional
frontier at every diversity level. Periodicity — the proposal's own decisive
test — moves the wrong way, and the premise is false: DPLM already carries
2.5× natural hydrophobic patterning, so there is no missing patterning to
install.

The mechanism was also published twice in 2025, and one of those papers'
own ablation found hard family binding to be its *worst* setting, which
contradicts the proposal's binding rule. The six families are, to within two
residues, the published Dokholyan–Mirny–Shakhnovich six-letter code of 2002.

**Why it fails is now known, and the explanation came later.** Binding makes
a family error unrecoverable, so a wrong pattern is actively harmful
(−6.38 pLDDT against generating freely). The unconditional two-level arm
supplies its own pattern at 64–78 % error, far outside the 8 % budget, so it
pays that penalty on most positions. That measurement lives in
`../conditioning/` because it was made with the conditioning apparatus; see
`../conditioning/results/I_pattern_curve_note.md`.

## This folder has no model of its own, on purpose

Idea B's model *is* [`../core/two_level/`](../core/two_level/) — the model,
its training, its samplers and its generation driver all live there. The
conditioning line of work uses the same architecture, the same training run
and the same checkpoint (`../ckpt/dms6.pt`), supplying the family pattern
instead of generating it.
Duplicating the model into both folders would leave two copies of the code
that produced the published checkpoints, so it sits in `core/` and both
folders use it. What is specific to Idea B is the *experiment*: generating
unconditionally and comparing against the flat arm.

## Layout

| path | what |
|---|---|
| `run/ideaB.sh` | the whole experiment: the main three-arm comparison, the quality/diversity temperature grids for the flat and two-level arms, the forward-pass sweep, and the two-phase sampler at three family/identity splits |
| `results/B_main*` | the three-arm comparison — flat, chemical six, random six |
| `results/B_qd_flat*`, `results/B_qd_dms6*` | the frontier: 1-D temperature against the 2-D family/identity grid |
| `results/B_nfe*` | the same arms at 25, 50 and 200 forward passes |
| `results/B_twophase*` | the family/identity budget split at ρ = 0.15, 0.3, 0.5 |
| `results/B_cleanrank*` | the commit-ranking ablation: what reporting pLDDT without diversity beside it would have let through |
| `results/B_main_note.md`, `results/B_premise_note.md`, `results/B_frontier_note.md` | the three findings, written up |
| `figures/fig4_ideaB`, `figures/fig6_ideaB` | the arms and the frontier |

The first conditioning run was numbered `B_cond*` before it became its own
line of work; those files are in `../conditioning/results/`.

## Order

Needs the checkpoints from `core/run/train_all.sh`:

```bash
sbatch core/run/train_all.sh          # if not already done
sbatch idea_b_two_level/run/ideaB.sh
python -m core.analysis.figures    # writes fig4 and fig6 here
```
