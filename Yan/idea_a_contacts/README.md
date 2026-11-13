# Idea A: choosing the parallel unmasking set by contact couplings

> **Copy for the shared repo.** This folder carries code, run scripts and
> figures only. The generated sequences, scored metrics and per-finding notes
> under `results/`, and the five write-ups under `reports/`, are not included;
> they live in the CIS6270 project repository under `project2/`. Names below
> beginning `results/` or `reports/` refer to files there — the run scripts
> write their own output into the empty `results/` directory here.

**The proposal.** A masked diffusion denoiser gives per-position marginals,
never the joint, so committing two *coupled* positions in the same step
samples them independently and loses their correlation. Run the ESM-2 contact
head on the partially masked sequence, sort masked positions by confidence,
and admit one only if its predicted coupling to every already-admitted
position stays below a threshold. Fewer, cleaner steps for the same quality.

**Verdict: negative, and it is the oracle that settles it.** Handed a
deposited X-ray contact map — the ceiling for any contact predictor — the
rule does not beat free confidence ordering: **+1.10 pLDDT [−2.67, +4.82]**
on CATH 4.4 S40, one chain per topology. Inside the remasking decoder DPLM
actually ships, the rule costs **−10.73 [−12.54, −8.91]**. No better contact
map can fix a mechanism whose oracle ceiling is already flat.

The idea is also not new. The admission rule is a method accepted at ICML
2026; a 2022 *Nature Communications* protein-design paper already colours a
contact graph into independent sets; and a 2023 workshop paper tried
coupling-biased unmasking order in a protein masked diffusion model, reported
that it failed, and listed this exact idea as an open problem.

## What survived as a finding

| finding | where |
|---|---|
| **External planners fail on masked input, and the repair is one line.** Fed the masked sequence, ESM-2 contact precision@L collapses from 0.747 to 0.067. Fill the masks with the denoiser's argmax first and it recovers to **0.737**, after which the external head beats the generator's own attention. This explains a published, unexplained eight-point failure in the closest prior work | `signals/argmax_contacts.py`, `analysis/argmax_gate.py`, `results/F_*`, `results/H_*` |
| **The firing-fraction diagnostic.** Harm is linear in how often the gate fires: `harm = −21.02 × fired_fraction + 1.99`, r² = 0.991, extrapolating to zero when the rule never fires. So the proposed warm-up does not repair the rule, it titrates the dose — a general way to tell "my schedule fixed the mechanism" from "my schedule disabled it" | `analysis/revised.py`, `results/C_*` |
| **The independence assumption is empirically safe.** The closest prior work states it and leaves it untested. Tested here with ground-truth maps, a pairwise rule does not beat a per-position one even with the true graph | `experiments/run_dependency.py`, `results/E_*` |
| **Contacts are the wrong graph.** The model's own interventional dependency is separated by true contact at AUROC 0.54 early / 0.73 late, while plain sequence distance predicts it better everywhere (0.70–0.85) | `signals/dependency.py`, `results/E_dependency.json` |
| **The commit-ranking trap.** Ranking commits by clean confidence and committing irrevocably looks 26 pLDDT better and is pure mode collapse: polyleucine, 131 of 200 identical residues | `run/scorecheck.sh` |
| **The power critique.** Per-sequence pLDDT sd is 17.8, so a 48-sequence cell resolves only effects above ~7 pLDDT while the largest published gain for unmasking-order selection on this model is 1.65 and needs ~913 per arm | `../core/analysis/report.py` |

## Withdrawn — do not carry forward

The **beta-fold effect** (`results/G_beta_effect_note.md`,
`analysis/beta_effect.py`). Reported as beta-minus-alpha +11.61 pLDDT and
withdrawn: the corroborating correlation used class-*mean* contact density,
which is a relabelling of the class. Against each chain's own density
r = +0.120, p = 0.44. Third winner's-curse artifact in this project, after
the sweep's best cell and the ad-hoc oracle chain set.

## Layout

| path | what |
|---|---|
| `model/sample.py` | DPLM loading and the two sampler families. **absorbing**: a written position is final — the family both proposals assume. **reparameterised**: re-predicts every position each step and returns the lowest-scoring to mask — what DPLM ships, and ~30 pLDDT better |
| `model/selectors.py` | the admission rules themselves: `confidence`, `contact`, `contact_shuffled`, `anti_contact`, `distance`, and the greedy decoupled set |
| `model/adaptive.py` | the revised method: warm-up, cached contact map, trigger refresh, group-size cap |
| `signals/contacts.py` | where the coupling comes from — a separate frozen ESM-2 pass, or the generator's own attention at zero extra cost |
| `signals/argmax_contacts.py` | the same, on the argmax completion instead of the masked sequence |
| `signals/dependency.py` | the model's own interventional dependency, measured rather than inferred from pLDDT |
| `oracles/oracle.py` | oracle maps from ESMFold coordinates, and the contact-quality diagnostics |
| `oracles/pdb_oracle.py` | oracle maps from **experimental** structures. The first oracle ran entirely inside the ESM-2 family; this breaks that circle |
| `oracles/cath_oracle.py` | the CATH 4.4 S40 replacement set, one chain per topology, class-stratified. The ad-hoc 48-chain set had 19 topologies and one fold in 11 entries |
| `oracles/self_oracle.py` | the sharpest oracle: fold the sequence the baseline itself produced and hand back its own true map |
| `experiments/run_*.py` | the drivers: the matched-budget grid, the revised method, the dependency study, the oracle redo |
| `experiments/gate_off_check.py` | gate-off ablation — proves turning the gate off recovers plain confidence ordering in this code |
| `analysis/` | the tables: paired contrasts, the argmax-gate configurations, the revised-method report, the control ladder with cost and diversity |

## Order

Needs `core/run/download.sh` first. Nothing here trains anything — DPLM is
pretrained, and that is why this folder has no training step.

```bash
sbatch idea_a_contacts/run/ideaA_main.sh      # the matched-forward-pass grid
sbatch idea_a_contacts/run/cath_oracle.sh     # build the CATH S40 oracle maps
sbatch idea_a_contacts/run/ideaA_oracle.sh    # the decisive oracle test
sbatch idea_a_contacts/run/ideaA_reparam.sh   # the same rule in DPLM's own decoder
sbatch idea_a_contacts/run/revised_all.sh     # the revised method
sbatch idea_a_contacts/run/argmax_gate.sh     # does a repaired map help the gate
sbatch idea_a_contacts/run/prof_asks.sh       # the control ladder
sbatch idea_a_contacts/run/confirm_power.sh   # replicate the sweep's best cell at 960/arm
python -m idea_a_contacts.analysis.revised
```

`run/confirm_power.sh` is worth keeping in the sequence: it is what showed 91 %
of the sweep's apparent +7.25 to be selection bias.
