"""Download protein structures from RCSB so the experiments have input.

Run from repo root:
    python blocks/get_data.py                            # default set
    python blocks/get_data.py --out blocks/data/pdb -n 20  # fewer
    python blocks/get_data.py --ids 1UBQ 2GB1 1SHG       # specific ones

Any directory of .pdb or .cif files works, so if you already have CATH domain
files, point the experiments at that directory instead and skip this.

Failures are skipped with a warning rather than raised, so one dead id does not
kill the run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

# Small, well-resolved, structurally diverse single-domain proteins. Chosen for
# length (most under 150 residues) so the bound experiment runs quickly.
DEFAULT_IDS = [
    "1UBQ", "1CRN", "2GB1", "1PGB", "1SHG", "1VII", "1BPI", "3CHY",
    "1ENH", "1FKB", "1TIT", "2CI2", "1LYZ", "2LZM", "1CSP", "1SRL",
    "2PTL", "1PIN", "1BTA", "1AKE", "1MBN", "1HRC", "1UBI", "1TEN",
    "1RIS", "1FNF", "1POH", "1OPC", "5PTI", "1A2P", "1BDD", "1LMB",
    "2TRX", "1DIV", "1HDN", "1PHT", "1QYS", "1ANK", "1ARB", "1BEB",
]

URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def download(pdb_id: str, out_dir: Path, timeout: int = 30) -> bool:
    pdb_id = pdb_id.strip().upper()
    if not pdb_id:
        return False
    dest = out_dir / f"{pdb_id}.pdb"
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        r = requests.get(URL.format(pdb_id=pdb_id), timeout=timeout)
        if r.status_code != 200 or len(r.content) < 500:
            print(f"  skip {pdb_id}: http {r.status_code}")
            return False
        dest.write_bytes(r.content)
        return True
    except Exception as exc:
        print(f"  skip {pdb_id}: {exc}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="blocks/data/pdb")
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--id-file", default=None)
    p.add_argument("-n", "--limit", type=int, default=None)
    args = p.parse_args()

    if args.id_file:
        ids = [l.strip() for l in Path(args.id_file).read_text().splitlines() if l.strip()]
    elif args.ids:
        ids = args.ids
    else:
        ids = DEFAULT_IDS
    if args.limit:
        ids = ids[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {len(ids)} structures to {out_dir}")

    ok = sum(download(i, out_dir) for i in ids)
    print(f"\n{ok}/{len(ids)} structures available in {out_dir}")
    if ok == 0:
        raise SystemExit("nothing downloaded; check the network or supply files yourself")


if __name__ == "__main__":
    main()
