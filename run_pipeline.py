"""Run the whole pipeline, steps 1 through 6, in one command.

    python run_pipeline.py                     # train, predict, hash, seal, export
    python run_pipeline.py --mode mock         # never touch the network
    python run_pipeline.py --mode real --limit 5
    python run_pipeline.py --tune-weights      # search voting weights first
    python run_pipeline.py --skip-train        # reuse the saved model

Each stage prints what it did and where it wrote it, so a demo can be narrated
step by step. Nothing here is new logic -- it just calls the modules in order:

    step 1-2  src.train_model   preprocess, fit five models, soft-vote them
    step 3-4  src.predict       one record per cow, keccak256 over the canonical JSON
    step 5-6  src.oracle        consensus gate, write on-chain, confirm, store
    step 7    src.database      export frontend/data.json + data.js for the dashboard
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Load .env before importing config, since config reads the environment at import.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    # python-dotenv is optional; the pipeline runs fine in mock mode without it.
    pass

from src import config  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 66}\n  {title}\n{'=' * 66}")


def main() -> int:
    ap = argparse.ArgumentParser(description="End-to-end mastitis + blockchain pipeline.")
    ap.add_argument("--mode", choices=["auto", "real", "mock"], default=None,
                    help="chain backend (default: CHAIN_MODE from .env, else auto)")
    ap.add_argument("--limit", type=int, default=None,
                    help="how many cows to write on-chain (default: ONCHAIN_BATCH_SIZE)")
    ap.add_argument("--batch", action="store_true",
                    help="write all eligible cows in one transaction")
    ap.add_argument("--tune-weights", action="store_true",
                    help="search soft-voting weights instead of using equal weights")
    ap.add_argument("--no-noise-test", action="store_true",
                    help="skip the noise-robustness sweep (faster)")
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse artifacts/ensemble_model.joblib")
    ap.add_argument("--predict-limit", type=int, default=None,
                    help="only score the first N cows")
    ap.add_argument("--csv", default=None, help="alternative dataset CSV")
    a = ap.parse_args()

    config.ensure_dirs()
    started = time.time()

    csv = Path(a.csv) if a.csv else config.DATA_CSV
    if not csv.exists():
        print(f"Dataset not found: {csv}")
        print("Put the CSV beside this script, or pass --csv path/to/file.csv")
        return 1

    # ---- steps 1 and 2 ---------------------------------------------------
    rule("STEPS 1-2  preprocess the dataset and train the soft-voting ensemble")
    if a.skip_train:
        if not config.MODEL_PATH.exists():
            print(f"--skip-train was given but {config.MODEL_PATH} does not exist.")
            return 1
        print(f"  reusing the saved model at {config.MODEL_PATH}")
    else:
        from src.train_model import train

        train(tune=a.tune_weights, noise_test=not a.no_noise_test, csv_path=a.csv)

    # ---- steps 3 and 4 ---------------------------------------------------
    rule("STEPS 3-4  score every cow and hash each result with keccak256")
    from src.predict import run as run_predictions

    records = run_predictions(limit=a.predict_limit, csv_path=a.csv)

    # ---- steps 5 and 6 ---------------------------------------------------
    rule("STEPS 5-6  consensus gate, write to the registry, confirm, store off-chain")
    import pandas as pd

    from src.oracle import submit_records

    df = pd.read_csv(csv)
    raw_rows = {str(r[config.ID_COLUMN]): r for _, r in df.iterrows()}

    stored, skipped = submit_records(
        records, raw_rows=raw_rows, mode=a.mode, limit=a.limit, use_batch=a.batch
    )

    # ---- step 7 feed -----------------------------------------------------
    rule("STEP 7  export the dashboard feed")
    from src.database import export_frontend, init_db

    conn = init_db()
    try:
        payload = export_frontend(conn)
    finally:
        conn.close()

    print(f"  {len(payload['records'])} record(s) -> {config.FRONTEND_DATA}")
    print(f"  script-loadable copy   -> {config.FRONTEND_DATA.with_suffix('.js')}")

    # ---- summary ---------------------------------------------------------
    modes = payload["stats"].get("chain_modes", {})
    rule("done")
    print(f"  elapsed            : {time.time() - started:.1f}s")
    print(f"  cows scored        : {len(records)}")
    print(f"  written to chain   : {len(stored)}")
    print(f"  withheld for review: {len(skipped)}")
    print(f"  chain modes in db  : {json.dumps(modes)}")

    if modes.get("mock"):
        print("\n  NOTE: mock-mode records are a local simulation. They prove the")
        print("  pipeline works, not that anything was published to a public ledger.")
        print("  For a real seal: deploy the contract, fill in .env, re-run with")
        print("  --mode real.")

    print(f"\n  Open the dashboard:")
    print(f"    file://{config.FRONTEND / 'index.html'}")
    print(f"  or serve it live with the API:")
    print(f"    uvicorn src.api:app --reload   ->  http://127.0.0.1:8000/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
