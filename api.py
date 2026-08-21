"""FastAPI layer between the pipeline and the dashboard.

Endpoints
    GET  /                      the dashboard itself
    GET  /api/health            service + chain status
    GET  /api/stats             herd summary counts
    GET  /api/model             training metrics for the model panel
    GET  /api/records           paged records, filterable
    GET  /api/records/{cow_id}  full history for one cow
    GET  /api/verify/{cow_id}   re-check a cow against the contract, live
    POST /api/predict           score a new reading, hash it, optionally publish

/api/verify is the endpoint that matters for the trust story. Everything else
reads the local cache; this one recomputes the hash from the stored values and
asks the contract whether that exact digest was ever recorded. It is what a
sceptical buyer would press.

Run:
    uvicorn src.api:app --reload
    python -m src.api          (equivalent)
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import config, consensus, database
from .hashing import KECCAK_BACKEND, hash_record, utc_timestamp, verify_record

app = FastAPI(
    title="Blockchain-Verified Cattle Health API",
    description="Mastitis screening with keccak256 results anchored on Polygon.",
    version="1.0.0",
)

# The dashboard may be opened straight from the filesystem, which counts as a
# null origin, so keep CORS permissive. Tighten this before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def db():
    return database.init_db()


# --------------------------------------------------------------------------
# request models
# --------------------------------------------------------------------------
class Reading(BaseModel):
    """One cow's sensor readings, as a farmer would enter them."""

    cow_id: str = Field(..., examples=["C0801"])
    Milk_Temperature: float = Field(..., examples=[38.4])
    Milk_pH: float = Field(..., examples=[6.95])
    Milk_Conductivity: float = Field(..., examples=[6.2])
    Somatic_Cell_Count: float = Field(..., examples=[420])
    Milk_Yield: float = Field(..., examples=[11.2])
    Clotting: int = Field(0, ge=0, le=1)
    publish: bool = Field(False, description="also write the hash on-chain")


# --------------------------------------------------------------------------
# read endpoints
# --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    conn = db()
    try:
        s = database.stats(conn)
    finally:
        conn.close()
    return {
        "status": "ok",
        "keccak_backend": KECCAK_BACKEND,
        "chain_mode_configured": config.CHAIN_MODE,
        "contract_address": config.CONTRACT_ADDRESS or None,
        "model_trained": config.MODEL_PATH.exists(),
        "records": s["total_records"],
    }


@app.get("/api/stats")
def get_stats():
    conn = db()
    try:
        return database.stats(conn)
    finally:
        conn.close()


@app.get("/api/model")
def get_model_metrics():
    import json

    if not config.METRICS_PATH.exists():
        raise HTTPException(404, "No metrics yet. Run: python -m src.train_model")
    return json.loads(config.METRICS_PATH.read_text())


@app.get("/api/records")
def list_records(
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None, pattern="^(healthy|mastitic)$"),
    verified_only: bool = False,
):
    conn = db()
    try:
        rows = database.get_records(
            conn, limit=limit, offset=offset, status=status, verified_only=verified_only
        )
        return {"count": len(rows), "records": rows}
    finally:
        conn.close()


@app.get("/api/records/{cow_id}")
def cow_history(cow_id: str):
    conn = db()
    try:
        rows = database.get_by_cow(conn, cow_id)
    finally:
        conn.close()
    if not rows:
        raise HTTPException(404, f"No records for cow {cow_id}")
    return {"cow_id": cow_id, "count": len(rows), "records": rows}


# --------------------------------------------------------------------------
# the verification endpoint
# --------------------------------------------------------------------------
@app.get("/api/verify/{cow_id}")
def verify(cow_id: str):
    """Two independent checks, reported separately.

    hash_matches_values -- recompute keccak256 from the stored values; catches any
                           edit to the record itself
    found_on_chain      -- ask the contract whether that digest is in this cow's
                           history; catches a record that was never published
    Both must be true for the record to mean anything to a buyer.
    """
    conn = db()
    try:
        rec = database.get_latest_by_cow(conn, cow_id)
    finally:
        conn.close()

    if not rec:
        raise HTTPException(404, f"No records for cow {cow_id}")

    local_ok = verify_record(rec)

    chain_ok, onchain_ts, idx, err = False, None, None, None
    try:
        from .oracle import get_chain

        chain = get_chain()
        chain_ok, onchain_ts, idx = chain.verify_record(cow_id, rec["result_hash"])
        mode = chain.mode
    except Exception as e:
        mode, err = "unavailable", str(e)

    return {
        "cow_id": cow_id,
        "prediction": rec["prediction"],
        "status": rec["status"],
        "confidence": rec["confidence"],
        "result_hash": rec["result_hash"],
        "tx_hash": rec.get("tx_hash"),
        "explorer_url": rec.get("explorer_url"),
        "hash_matches_values": local_ok,
        "found_on_chain": bool(chain_ok),
        "fully_verified": bool(local_ok and chain_ok and mode == "real"),
        "chain_mode": mode,
        "onchain_timestamp": onchain_ts,
        "record_index": idx,
        "error": err,
        "note": (
            "fully_verified requires the real chain; in mock mode the hash check "
            "still runs but there is no independent ledger behind it."
        ),
    }


# --------------------------------------------------------------------------
# live scoring
# --------------------------------------------------------------------------
@app.post("/api/predict")
def predict_one(reading: Reading):
    """Score a hand-entered reading. Powers the 'add a reading' demo form."""
    import pandas as pd

    from .predict import load_model

    try:
        bundle = load_model()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))

    schema = bundle["schema"]
    features = schema["numeric"] + schema["binary"] + schema["categorical"]
    payload = reading.model_dump()

    missing = [f for f in features if f not in payload]
    if missing:
        raise HTTPException(422, f"missing feature(s): {missing}")

    X = pd.DataFrame([{f: payload[f] for f in features}])
    risk = float(bundle["pipeline"].predict_proba(X)[:, 1][0])
    prediction = int(risk >= config.DECISION_THRESHOLD)
    confidence = risk if prediction == 1 else 1.0 - risk

    rec = hash_record(reading.cow_id, prediction, confidence, utc_timestamp())
    rec["risk_score"] = round(risk, 4)
    rec["status"] = "mastitic" if prediction else "healthy"
    rec["needs_review"] = confidence < config.REVIEW_CONFIDENCE

    c = consensus.evaluate(prediction, payload, confidence)
    rec.update(c.as_dict())

    if reading.publish:
        if not c.agreed:
            return JSONResponse(
                status_code=409,
                content={
                    **rec,
                    "published": False,
                    "reason": "sources disagree; held for vet review rather than published",
                },
            )
        from .oracle import get_chain

        chain = get_chain()
        receipt = chain.store_health_record(rec["cow_id"], rec["result_hash"], rec["timestamp"])
        rec.update({k: receipt[k] for k in
                    ("tx_hash", "block_number", "gas_used", "mode", "explorer_url")})
        ok, _, _ = chain.verify_record(rec["cow_id"], rec["result_hash"])
        rec["onchain_verified"] = bool(ok)

        conn = db()
        try:
            database.insert_record(conn, rec)
        finally:
            conn.close()
        rec["published"] = True

    return rec


# --------------------------------------------------------------------------
# dashboard
# --------------------------------------------------------------------------
@app.get("/")
def dashboard():
    index = config.FRONTEND / "index.html"
    if not index.exists():
        raise HTTPException(404, "frontend/index.html is missing")
    return FileResponse(index)


def main():
    import uvicorn

    print(f"  dashboard: http://{config.API_HOST}:{config.API_PORT}/")
    print(f"  api docs : http://{config.API_HOST}:{config.API_PORT}/docs")
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)


if __name__ == "__main__":
    main()
