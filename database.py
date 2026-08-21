"""Step 6 - the off-chain store.

Reading from a public RPC on every page load is slow and rate-limited, so the
dashboard reads from here instead. The chain remains the source of truth for
*integrity*; this table is a cache for *speed*.

The design keeps that distinction honest. Alongside each result it stores the
result_hash, the tx_hash and the chain mode, so any row can be re-checked against
the contract at any time (see `api.py /verify/{cow_id}` and
`oracle.py --verify`). A row is never treated as trustworthy just because it is
in the database.

SQLite is used because it needs no server and no credentials. The schema is plain
SQL, so pointing this at Postgres later is mostly a matter of swapping the
connection and the AUTOINCREMENT keyword.

Usage
    python -m src.database --stats
    python -m src.database --export     # write frontend/data.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS health_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cow_id          TEXT    NOT NULL,
    prediction      INTEGER NOT NULL,
    confidence      REAL    NOT NULL,
    risk_score      REAL,
    status          TEXT,
    timestamp       INTEGER NOT NULL,
    payload         TEXT    NOT NULL,
    result_hash     TEXT    NOT NULL,
    tx_hash         TEXT,
    block_number    INTEGER,
    gas_used        INTEGER,
    chain_mode      TEXT,
    explorer_url    TEXT,
    onchain_verified INTEGER DEFAULT 0,
    needs_review    INTEGER DEFAULT 0,
    consensus_agreed INTEGER,
    consensus_votes TEXT,
    consensus_reasons TEXT,
    created_at      TEXT    NOT NULL
);

-- One row per (cow, hash): re-running the pipeline updates rather than duplicates.
CREATE UNIQUE INDEX IF NOT EXISTS idx_cow_hash
    ON health_records (cow_id, result_hash);
CREATE INDEX IF NOT EXISTS idx_cow      ON health_records (cow_id);
CREATE INDEX IF NOT EXISTS idx_verified ON health_records (onchain_verified);
CREATE INDEX IF NOT EXISTS idx_ts       ON health_records (timestamp);
"""


def init_db(path=None) -> sqlite3.Connection:
    path = path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def insert_record(conn: sqlite3.Connection, rec: dict) -> int:
    """Upsert one record. Returns the row id."""
    row = (
        str(rec["cow_id"]),
        int(rec["prediction"]),
        float(rec["confidence"]),
        float(rec.get("risk_score") or 0.0),
        rec.get("status") or ("mastitic" if rec["prediction"] else "healthy"),
        int(rec["timestamp"]),
        rec["payload"],
        rec["result_hash"],
        rec.get("tx_hash"),
        rec.get("block_number"),
        rec.get("gas_used"),
        rec.get("mode") or rec.get("chain_mode"),
        rec.get("explorer_url"),
        int(bool(rec.get("onchain_verified"))),
        int(bool(rec.get("needs_review"))),
        None if rec.get("consensus_agreed") is None else int(bool(rec["consensus_agreed"])),
        json.dumps(rec.get("consensus_votes")) if rec.get("consensus_votes") else None,
        json.dumps(rec.get("consensus_reasons")) if rec.get("consensus_reasons") else None,
        datetime.now(timezone.utc).isoformat(),
    )
    cur = conn.execute(
        """
        INSERT INTO health_records (
            cow_id, prediction, confidence, risk_score, status, timestamp,
            payload, result_hash, tx_hash, block_number, gas_used, chain_mode,
            explorer_url, onchain_verified, needs_review, consensus_agreed,
            consensus_votes, consensus_reasons, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT (cow_id, result_hash) DO UPDATE SET
            tx_hash          = excluded.tx_hash,
            block_number     = excluded.block_number,
            gas_used         = excluded.gas_used,
            chain_mode       = excluded.chain_mode,
            explorer_url     = excluded.explorer_url,
            onchain_verified = excluded.onchain_verified,
            consensus_agreed = excluded.consensus_agreed,
            consensus_votes  = excluded.consensus_votes,
            consensus_reasons= excluded.consensus_reasons
        """,
        row,
    )
    conn.commit()
    return cur.lastrowid


def _shape(r: sqlite3.Row) -> dict:
    d = dict(r)
    for key in ("consensus_votes", "consensus_reasons"):
        if d.get(key):
            try:
                d[key] = json.loads(d[key])
            except json.JSONDecodeError:
                pass
    d["onchain_verified"] = bool(d["onchain_verified"])
    d["needs_review"] = bool(d["needs_review"])
    if d.get("consensus_agreed") is not None:
        d["consensus_agreed"] = bool(d["consensus_agreed"])
    return d


def get_records(conn, limit=500, offset=0, status=None, verified_only=False) -> list:
    sql = "SELECT * FROM health_records WHERE 1=1"
    args = []
    if status in ("healthy", "mastitic"):
        sql += " AND status = ?"
        args.append(status)
    if verified_only:
        sql += " AND onchain_verified = 1"
    sql += " ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?"
    args += [int(limit), int(offset)]
    return [_shape(r) for r in conn.execute(sql, args).fetchall()]


def get_by_cow(conn, cow_id: str) -> list:
    return [
        _shape(r)
        for r in conn.execute(
            "SELECT * FROM health_records WHERE cow_id = ? ORDER BY timestamp DESC",
            (str(cow_id),),
        ).fetchall()
    ]


def get_latest_by_cow(conn, cow_id: str):
    rows = get_by_cow(conn, cow_id)
    return rows[0] if rows else None


def stats(conn) -> dict:
    q = lambda s, a=(): conn.execute(s, a).fetchone()[0]
    total = q("SELECT COUNT(*) FROM health_records")
    return {
        "total_records": total,
        "distinct_cows": q("SELECT COUNT(DISTINCT cow_id) FROM health_records"),
        "mastitic": q("SELECT COUNT(*) FROM health_records WHERE prediction = 1"),
        "healthy": q("SELECT COUNT(*) FROM health_records WHERE prediction = 0"),
        "onchain_verified": q("SELECT COUNT(*) FROM health_records WHERE onchain_verified = 1"),
        "needs_review": q("SELECT COUNT(*) FROM health_records WHERE needs_review = 1"),
        "consensus_disputed": q(
            "SELECT COUNT(*) FROM health_records WHERE consensus_agreed = 0"
        ),
        "avg_confidence": round(
            conn.execute("SELECT COALESCE(AVG(confidence),0) FROM health_records").fetchone()[0], 4
        ),
        "chain_modes": {
            r[0] or "unknown": r[1]
            for r in conn.execute(
                "SELECT chain_mode, COUNT(*) FROM health_records GROUP BY chain_mode"
            ).fetchall()
        },
    }


def export_frontend(conn, path=None) -> dict:
    """Dump the table to frontend/data.json.

    This is what lets the dashboard open as a plain file with real data, no API
    server required -- useful when demoing on a machine that is not running uvicorn.
    """
    path = path or config.FRONTEND_DATA
    path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {}
    if config.METRICS_PATH.exists():
        try:
            metrics = json.loads(config.METRICS_PATH.read_text())
        except json.JSONDecodeError:
            pass

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "explorer_tx_url": config.EXPLORER_TX_URL,
        "contract_address": config.CONTRACT_ADDRESS or None,
        "stats": stats(conn),
        "model": {
            "trained_at": metrics.get("trained_at"),
            "n_train": metrics.get("n_train"),
            "n_test": metrics.get("n_test"),
            "class_balance": metrics.get("class_balance"),
            "features": metrics.get("features"),
            "ensemble": metrics.get("ensemble"),
            "per_model": metrics.get("per_model"),
            "confusion_matrix": metrics.get("confusion_matrix"),
            "cross_validation": metrics.get("cross_validation"),
            "voting": metrics.get("voting"),
            "noise_robustness": metrics.get("noise_robustness"),
        },
        "records": get_records(conn, limit=5000),
    }
    path.write_text(json.dumps(payload, indent=2))

    # Also emit a <script>-loadable copy. Browsers block fetch() of a file:// URL
    # (null-origin CORS), so a plain double-clicked index.html cannot read
    # data.json directly -- but it CAN load data.js via a <script> tag. This is
    # what makes the dashboard genuinely open as a file with real data, no server.
    js_path = path.with_suffix(".js")
    js_path.write_text("window.__DATA__ = " + json.dumps(payload) + ";\n")

    return payload


def main():
    ap = argparse.ArgumentParser(description="Off-chain record store.")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--export", action="store_true", help="write frontend/data.json")
    ap.add_argument("--cow", help="show every record for one cow")
    a = ap.parse_args()

    conn = init_db()
    if a.cow:
        for r in get_by_cow(conn, a.cow):
            print(json.dumps(r, indent=2))
    elif a.export:
        p = export_frontend(conn)
        print(f"  exported {len(p['records'])} record(s) -> {config.FRONTEND_DATA}")
    else:
        print(json.dumps(stats(conn), indent=2))
    conn.close()


if __name__ == "__main__":
    main()
