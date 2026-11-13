"""Download CATH S35 domain structures for experiments.

This fetches the same CATH domains that Yan and Neha use in CIS6270/project2,
ensuring your block-diffusion experiments are directly comparable.

Run from repo root:
    python blocks/get_cath.py                                  # default: 100 domains
    python blocks/get_cath.py --n 200 --out blocks/data/cath   # more domains
    python blocks/get_cath.py --ids 1a8dA01 1aozA02            # specific domains

The CATH S35 set is non-redundant at 35% sequence identity and contains
structural DOMAINS (not full-length proteins), making it ideal for testing
structure-based partition strategies.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import requests

# CATH v4.4 API endpoint for domain PDB files
CATH_API = "http://www.cathdb.info/version/v4_4_0/api/rest/id/{domain_id}.pdb"
# CATH S35 representative domains FASTA
CATH_S35_URL = "http://download.cathdb.info/cath/releases/all-releases/v4_4_0/sequence-data/cath-domain-seqs-S35.fa"

# Regex to extract domain ID from CATH FASTA header
# e.g., "cath|4_4_0|1a8dA01/1-246" -> "1a8dA01"
DOMAIN_RE = re.compile(r"cath\|[\d_]+\|(\w+)/")


def fetch_cath_s35_ids(min_length: int = 40, max_length: int = 256) -> list[tuple[str, str]]:
    """Download CATH S35 FASTA and extract domain IDs with sequences.

    Returns list of (domain_id, sequence) tuples filtered by length.
    """
    print(f"Fetching CATH S35 sequence list...")
    r = requests.get(CATH_S35_URL, timeout=120)
    r.raise_for_status()

    domains = []
    current_header, current_seq = None, []

    for line in r.text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            if current_header and current_seq:
                seq = "".join(current_seq)
                if min_length <= len(seq) <= max_length:
                    m = DOMAIN_RE.search(current_header)
                    if m:
                        domains.append((m.group(1), seq))
            current_header = line
            current_seq = []
        elif line:
            current_seq.append(line)

    # Handle last entry
    if current_header and current_seq:
        seq = "".join(current_seq)
        if min_length <= len(seq) <= max_length:
            m = DOMAIN_RE.search(current_header)
            if m:
                domains.append((m.group(1), seq))

    print(f"  Found {len(domains)} domains in length range [{min_length}, {max_length}]")
    return domains


def download_domain(domain_id: str, out_dir: Path, timeout: int = 60) -> bool:
    """Download a single CATH domain PDB file."""
    dest = out_dir / f"{domain_id}.pdb"
    if dest.exists() and dest.stat().st_size > 500:
        return True

    try:
        r = requests.get(CATH_API.format(domain_id=domain_id), timeout=timeout)
        if r.status_code != 200:
            print(f"  skip {domain_id}: HTTP {r.status_code}")
            return False
        if b"ATOM" not in r.content:
            print(f"  skip {domain_id}: no ATOM records")
            return False
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"  skip {domain_id}: {e}")
        return False


def main():
    p = argparse.ArgumentParser(
        description="Download CATH S35 domain structures for block diffusion experiments"
    )
    p.add_argument("--out", default="blocks/data/cath", help="Output directory for PDB files")
    p.add_argument("--n", type=int, default=100, help="Number of domains to download")
    p.add_argument("--ids", nargs="*", help="Specific domain IDs to download")
    p.add_argument("--min-length", type=int, default=40, help="Minimum sequence length")
    p.add_argument("--max-length", type=int, default=256, help="Maximum sequence length")
    p.add_argument("--sleep", type=float, default=0.2, help="Delay between requests")
    p.add_argument("--seed", type=int, default=0, help="Random seed for domain selection")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ids:
        # Use specific domain IDs
        targets = [(d, None) for d in args.ids]
    else:
        # Fetch from CATH S35
        all_domains = fetch_cath_s35_ids(args.min_length, args.max_length)

        # Sample domains reproducibly
        import random
        random.seed(args.seed)
        if len(all_domains) > args.n:
            targets = random.sample(all_domains, args.n)
        else:
            targets = all_domains
        print(f"Selected {len(targets)} domains (seed={args.seed})")

    # Download structures
    print(f"Downloading {len(targets)} domain structures to {out_dir}...")
    ok = 0
    for i, (domain_id, seq) in enumerate(targets):
        if download_domain(domain_id, out_dir):
            ok += 1
        time.sleep(args.sleep)
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i + 1}/{len(targets)} ({ok} successful)")

    # Write sequence index for reference
    index_file = out_dir / "index.txt"
    with open(index_file, "w") as f:
        for domain_id, seq in targets:
            if seq:
                f.write(f"{domain_id}\t{len(seq)}\t{seq}\n")
            else:
                f.write(f"{domain_id}\n")

    print(f"\nDownloaded {ok}/{len(targets)} structures to {out_dir}")
    print(f"Index written to {index_file}")

    if ok == 0:
        raise SystemExit("No structures downloaded; check network or try different domains")


if __name__ == "__main__":
    main()
