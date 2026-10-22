"""Write the kept-frame lists of every evaluation cohort (results/manifests/<cohort>.txt, one frame id per line).

The ids are those of the per-image records in AdaPolyp's evaluation logs; the deduplication itself is done by
dang/bench/data.py, which rebuilds the same lists from the public datasets and logs their SHA-256 prefix
(BENCH_MANIFEST_SHA256_16).

    uv run --locked python scripts/export_manifests.py
"""

import glob
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = {"Kvasir": 100, "CVC-ClinicDB": 62, "CVC-ColonDB": 380, "ETIS-LaribPolypDB": 196, "CVC-300": 60,
            "PolypGen": 1399, "PolypDB": 1376}


def main():
    kept = {}
    for f in sorted(glob.glob(str(ROOT / "results" / "logs" / "DANG__*.log.gz"))):
        for line in gzip.open(f, "rt", errors="replace"):
            if line.startswith("BENCH_PER_IMAGE: "):
                d = json.loads(line[len("BENCH_PER_IMAGE: "):])
                kept.setdefault(d["cohort"], set()).update(r["u"] for r in d["recs"])
    out = ROOT / "results" / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    for cohort, n in EXPECTED.items():
        ids = sorted(kept[cohort])
        assert len(ids) == n, (cohort, len(ids), n)
        (out / f"{cohort}.txt").write_text("\n".join(ids) + "\n")
        print(f"{cohort}: {len(ids)} frames")


if __name__ == "__main__":
    main()
