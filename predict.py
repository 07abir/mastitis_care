"""Step 3 - produce one prediction record per cow.

Each cow gets:
    {"cow_id": "C0001", "prediction": 1, "confidence": 0.87, ...}

`prediction`  0 = healthy, 1 = mastitic
`confidence`  probability of the predicted class, i.e. max(p, 1-p) -- the standard
              ML reading of confidence, so a healthy cow at 0.97 means "97% sure
              she is healthy", not "97% risk"
`risk_score`  P(mastitis) on its own, which is the number a farmer wants to sort by

No information is lost by hashing only the four spec'd fields: risk_score is
recoverable as `confidence if prediction == 1 else 1 - confidence`.

Usage
    python -m src.predict                  # score every cow in the dataset
    python -m src.predict --limit 25       # just the first 25
    python -m src.predict --only-test-set  # only cows the model never trained on
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import config
from .hashing import hash_record, utc_timestamp
from .preprocess import prepare


def load_model():
    """Load the trained pipeline, with a clear error if training hasn't run."""
    import joblib

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {config.MODEL_PATH}.\n"
            f"Run:  python -m src.train_model"
        )
    return joblib.load(config.MODEL_PATH)


def predict_frame(model_bundle, X: pd.DataFrame, ids, timestamp=None) -> list:
    """Score a frame of cow readings and return hashed prediction records."""
    pipeline = model_bundle["pipeline"]
    ts = timestamp if timestamp is not None else utc_timestamp()

    proba_mastitic = pipeline.predict_proba(X)[:, 1]
    records = []

    for cow_id, risk in zip(ids, proba_mastitic):
        prediction = int(risk >= config.DECISION_THRESHOLD)
        confidence = float(risk if prediction == 1 else 1.0 - risk)

        rec = hash_record(cow_id, prediction, confidence, ts)
        rec["risk_score"] = round(float(risk), 4)
        rec["status"] = "mastitic" if prediction == 1 else "healthy"
        # Low-confidence calls should go to a vet, not straight into a buyer's
        # dashboard as fact.
        rec["needs_review"] = bool(confidence < config.REVIEW_CONFIDENCE)
        records.append(rec)

    return records


def run(limit=None, only_test_set=False, csv_path=None, verbose=True) -> list:
    config.ensure_dirs()
    bundle = load_model()
    data = prepare(csv_path, verbose=False)

    if only_test_set:
        X, ids, truth = data.X_test, data.ids_test, data.y_test
        scope = "held-out test set (never seen during training)"
    else:
        X, ids, truth = data.X_all, data.ids_all, data.y_all
        scope = "all cows in the dataset"

    if limit:
        X, ids, truth = X.iloc[:limit], ids.iloc[:limit], truth.iloc[:limit]

    records = predict_frame(bundle, X, ids)

    # The CSV is labelled, so report real accuracy rather than asking the user
    # to trust the output blindly. Live sensor data would have no truth column.
    correct = sum(int(r["prediction"] == t) for r, t in zip(records, truth))

    if verbose:
        n_mast = sum(r["prediction"] for r in records)
        n_review = sum(r["needs_review"] for r in records)
        print(f"STEP 3 - predictions ({scope})")
        print(f"  scored          : {len(records)} cows")
        print(f"  flagged mastitic: {n_mast}   healthy: {len(records) - n_mast}")
        print(f"  low confidence  : {n_review} (below {config.REVIEW_CONFIDENCE:.2f}, "
              f"routed to vet review)")
        print(f"  accuracy vs CSV : {correct}/{len(records)} = {correct/len(records):.4f}")
        print("\nSTEP 4 - keccak256 hashes")
        for r in records[:3]:
            print(f"  {r['cow_id']}  pred={r['prediction']} "
                  f"conf={r['confidence']:.4f}  {r['result_hash']}")
        if len(records) > 3:
            print(f"  ... and {len(records) - 3} more")

    config.PREDICTIONS_PATH.write_text(json.dumps(records, indent=2))
    if verbose:
        print(f"\n  saved -> {config.PREDICTIONS_PATH}")

    return records


def main():
    ap = argparse.ArgumentParser(description="Generate per-cow mastitis predictions.")
    ap.add_argument("--limit", type=int, default=None, help="only score the first N cows")
    ap.add_argument("--only-test-set", action="store_true",
                    help="score only held-out cows (honest accuracy)")
    ap.add_argument("--csv", default=None, help="alternative dataset CSV")
    a = ap.parse_args()
    run(limit=a.limit, only_test_set=a.only_test_set, csv_path=a.csv)


if __name__ == "__main__":
    main()
