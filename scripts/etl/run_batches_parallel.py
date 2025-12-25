"""
Launch multiple batch50_full_ingest jobs in parallel with explicit symbol lists.

Groups are hard-coded for current backlog; adjust as needed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "python"
COMMON_ARGS = [
    "scripts/etl/run_batch50_full_ingest.py",
    "--source-db",
    "services/data_ingestion/data/tse_data.db",
    "--target-db",
    "postgresql://gravity:gravity_db_pass@127.0.0.1:5545/tech_analysis",
    "--batch-size",
    "10",
    "--min-candles",
    "120",
    "--limit",
    "0",
    "--ingest-limit",
    "0",
]

# Explicit groups (10 symbols each) from the current backlog
GROUPS = [
    "باران,بازرگام,بالاس,بایکا,بپاس,بپیوند,بتک,بتهران,بجهرم,بخاور",
    "بزاگرس,بزندگی,بسام,بکابل,بکهنوج,بگیلان,بمپنا,بمولد,بمیلا,بنو",
    "بهامرز,بهیر,بیوتیک,پارتا,پارس,پارسان,پارسیان,پاسا,پاکشو,پتایر",
    "پترول,پخش,پدرخش,پرداخت,پردیس,پسهند,پکرمان,پکویر,پلاست,پلاسک",
]


def main() -> None:
    procs: list[subprocess.Popen] = []
    for idx, symbols in enumerate(GROUPS, start=1):
        args = [PYTHON, *COMMON_ARGS, "--symbols", symbols]
        print(f"[launcher] starting batch{idx}: {symbols}")
        procs.append(
            subprocess.Popen(
                args,
                cwd=REPO_ROOT,
            )
        )
    # Wait for all
    for idx, p in enumerate(procs, start=1):
        rc = p.wait()
        print(f"[launcher] batch{idx} exited with code {rc}")


if __name__ == "__main__":
    main()
