# CIS 6270 Project 2: Protein Parallel Unmasking

Shared repo for testing individual ideas on masked diffusion for proteins.

## Team

- **Anush** (`blocks/`) - Non-contiguous block diffusion partitions
- **Yan** - Add your folder here (e.g., `hier/`)
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
├── hier/               # Yan: add your hierarchical alphabet code here
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

## Adding Your Code

1. Create your folder: `mkdir hier` (or `parallel`)
2. Add your Python files and a `requirements.txt`
3. Add a `run/` subfolder for your shell scripts
4. Update this README with your quickstart instructions

The structure mirrors `CIS6270/project2/` so we can easily compare results.
