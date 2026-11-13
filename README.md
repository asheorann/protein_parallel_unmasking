# CIS 6270 Project 2: Protein Parallel Unmasking

Shared repo for testing individual ideas on masked diffusion for proteins.

## Team

- **Anush** (`blocks/`) - Non-contiguous block diffusion partitions
- **Yan** (`Yan/`) - Contact-gated parallel unmasking, two-level chemical alphabet, and chemical-pattern conditioning
- **Neha** - Add your folder here (e.g., `parallel/`)

## Repository Structure

```
protein_parallel_unmasking/
├── README.md           # this file
├── blocks/             # Anush's block diffusion code
│   ├── run/            # shell scripts
│   ├── cli.py          # main entry point
│   ├── requirements.txt
│   └── ...
├── Yan/                # Yan's masked diffusion code
│   ├── core/           # shared model, training, alphabet, metrics
│   ├── idea_a_contacts/    # contact-gated parallel unmasking
│   ├── idea_b_two_level/   # the two-level chemical alphabet
│   ├── conditioning/       # chemical-pattern conditioning
│   ├── requirements.txt
│   └── README.md
└── parallel/           # Neha: add your contact-guided unmasking here
```

Each person's code lives in its own subfolder. Run commands from this root directory.

---

## For Yan: How to Run Anush's Code

### 1. Setup (one time)

```bash
cd protein_parallel_unmasking
pip install -r blocks/requirements.txt
```

### 2. Quick smoke test (no GPU needed)

```bash
bash blocks/run/smoke.sh
```

### 3. Download data

**Option A: Small RCSB proteins (quick test)**
```bash
python blocks/get_data.py
```

**Option B: CATH S35 domains (same as your experiments)**
```bash
python blocks/get_cath.py --n 100
```

### 4. Run the main experiment

```bash
# Using RCSB data (default)
bash blocks/run/bound.sh

# Using CATH data (for comparison with your results)
PDB_DIR=blocks/data/cath bash blocks/run/bound.sh
```

This estimates the NLL bound for each partition strategy. Takes a few minutes on GPU.

### 5. Generate sequences and score (optional, slower)

```bash
bash blocks/run/sample.sh          # generation + ESMFold scoring
bash blocks/run/figures.sh         # create figures
```

### 6. Results

Results appear in:
- `blocks/results/*.jsonl` - raw data
- `blocks/figures/*.pdf` - plots

---

## Environment Variables

| Variable | Default | What it does |
|----------|---------|--------------|
| `DEVICE` | `cuda` | Set to `cpu` if no GPU |
| `PDB_DIR` | `blocks/data/pdb` | Input structure directory |
| `LIMIT` | `100` | Number of proteins to process |
| `MODEL` | `facebook/esm2_t30_150M_UR50D` | ESM-2 model size |

Example with custom settings:
```bash
DEVICE=cpu LIMIT=10 bash blocks/run/bound.sh
```

---

## For the Team: How to Run Yan's Code

Three lines of work: two proposals, both negative, and one positive result.
Details and the per-folder order are in `Yan/README.md`.

| folder | the claim | verdict |
|---|---|---|
| `Yan/idea_a_contacts/` | choose the parallel unmasking set by contact couplings | **negative.** A deposited X-ray contact map does not beat free confidence ordering (+1.10 pLDDT, n.s.); inside the decoder DPLM ships the rule costs −10.73 |
| `Yan/idea_b_two_level/` | write six chemical families before exact identities | **negative.** 6–8 pLDDT below a flat baseline; chemistry beats a bit-matched random partition by 1.3, intervals overlapping |
| `Yan/conditioning/` | hand the model a real protein's chemical pattern instead of letting it invent one | **positive.** +23.4 pLDDT over the random control on a homology-purged corpus, and the sequences adopt the source domain's fold (Ref-TM 0.596, 64.6 % above the same-fold threshold) |

The third uses the second's model, differing only in whether the family
pattern is generated or supplied — which is the finding. The shared model
therefore lives in `Yan/core/`, not in either folder.

### 1. Setup (one time)

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r Yan/requirements.txt
```

torch must be 2.6 or newer: transformers refuses `.bin` checkpoints below
that, and DPLM ships only `.bin`. numpy stays below 2.

### 2. Weights and a smoke test

```bash
bash Yan/core/run/download.sh     # DPLM, ESM-2, ESMFold into $HF_HOME
bash Yan/core/run/smoke.sh        # one GPU: does it generate, score, fold
```

### 3. Corpus and the shared checkpoints

```bash
python Yan/core/tools/cache_swissprot.py   # needs the SwissProt-Pfam HF dataset
bash Yan/core/run/train_all.sh             # used by Idea B and conditioning
```

### 4. The headline result

```bash
bash Yan/conditioning/run/clustered_split.sh    # blastp homology purge
bash Yan/conditioning/run/retrain_clustered.sh  # retrain all arms, purged
bash Yan/conditioning/run/cond_clustered.sh     # the 2x2
bash Yan/conditioning/run/ref_tm.sh             # the fold, vs deposited coordinates
```

The analysis modules import as `core.*`, `conditioning.*` and
`idea_a_contacts.*`, so run those from inside the folder:

```bash
cd Yan && python -m conditioning.analysis.figures_cond
```

### 5. Environment variables

| Variable | Default | What it does |
|----------|---------|--------------|
| `PY` | `python` | Interpreter the run scripts use |
| `HF_HOME` | `$HOME/.cache/huggingface` | Where weights and datasets are cached |

Scripts carry `#SBATCH` headers and were run on Slurm (A100). Under plain
`bash` those lines are comments and the script runs in the foreground.

### 6. Results

Generated sequences, scored metrics, the per-finding notes and the five
write-ups are **not** in this repo — they are in `CIS6270/project2/`. The
analysis scripts read from `Yan/*/results/`, which is empty here, so either
regenerate with the scripts above or copy the result files across. Figures
are committed under `Yan/*/figures/`.

---

## Adding Your Code

1. Create your folder: `mkdir hier` (or `parallel`)
2. Add your Python files and a `requirements.txt`
3. Add a `run/` subfolder for your shell scripts
4. Update this README with your quickstart instructions

The structure mirrors `CIS6270/project2/` so we can easily compare results.
