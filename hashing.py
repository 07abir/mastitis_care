"""Step 4 - hash each result so the chain never stores raw farm data.

Only a 32-byte keccak256 digest goes on-chain. That keeps the cow's readings
private and transactions cheap, while still proving the record was not altered:
change one digit of confidence and the recomputed hash no longer matches what
the contract stored.

CANONICAL SERIALIZATION MATTERS
-------------------------------
A hash is only useful if a third party can recompute it and get the same bytes.
Python's ``str(0.87)`` and JavaScript's ``String(0.87)`` do not always agree, and
dict ordering is not guaranteed across languages, so the payload is pinned down:

  * keys sorted alphabetically
  * no whitespace between tokens
  * confidence written as a fixed 4-decimal STRING, never a raw float
  * timestamp as an integer of whole unix seconds
  * UTF-8 encoded

Result: ``{"confidence":"0.8700","cow_id":"C0001","prediction":1,"timestamp":1755...}``

The matching JavaScript is in frontend/index.html (``canonicalPayload``) so a
buyer can verify in their own browser. Change the rules here and you must change
them there too, or every previously stored hash stops verifying.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

CONFIDENCE_DECIMALS = 4

# Prefer the real Ethereum implementations; fall back so the pipeline still runs
# on a machine that has not installed web3.
try:  # pragma: no cover
    from web3 import Web3

    def _keccak(data: bytes) -> bytes:
        return Web3.keccak(data)

    KECCAK_BACKEND = "web3.Web3.keccak"
except ImportError:  # pragma: no cover
    try:
        from eth_hash.auto import keccak as _eth_keccak

        def _keccak(data: bytes) -> bytes:
            return _eth_keccak(data)

        KECCAK_BACKEND = "eth_hash.auto.keccak"
    except ImportError:
        from .keccak_fallback import keccak256 as _keccak

        KECCAK_BACKEND = "pure-python fallback (src/keccak_fallback.py)"


def utc_timestamp() -> int:
    """Whole unix seconds. Sub-second precision would break reproducibility."""
    return int(datetime.now(timezone.utc).timestamp())


def canonical_payload(cow_id: str, prediction: int, confidence: float, timestamp: int) -> str:
    """Build the exact string that gets hashed. Deterministic across languages."""
    conf = float(confidence)
    # A NaN or infinite probability means something upstream broke -- a model
    # divided by zero, a feature arrived empty. Left alone it would serialize as
    # "nan" and hash perfectly happily, putting a meaningless digest on an
    # immutable ledger. Refuse instead.
    if conf != conf or conf in (float("inf"), float("-inf")):
        raise ValueError(
            f"confidence for cow {cow_id!r} is {confidence!r}, which cannot be "
            f"hashed meaningfully. Check the model's predict_proba output."
        )
    if not 0.0 <= conf <= 1.0:
        raise ValueError(
            f"confidence for cow {cow_id!r} is {conf}, outside [0, 1]. "
            f"It should be a probability."
        )
    if int(prediction) not in (0, 1):
        raise ValueError(f"prediction for cow {cow_id!r} must be 0 or 1, got {prediction!r}")

    record = {
        "cow_id": str(cow_id),
        "prediction": int(prediction),
        # string, not float: pins the precision so no language rounds differently
        "confidence": f"{conf:.{CONFIDENCE_DECIMALS}f}",
        "timestamp": int(timestamp),
    }
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def hash_record(cow_id: str, prediction: int, confidence: float, timestamp: int) -> dict:
    """Return the canonical payload plus its 0x-prefixed keccak256 digest."""
    payload = canonical_payload(cow_id, prediction, confidence, timestamp)
    digest = _keccak(payload.encode("utf-8"))
    return {
        "cow_id": str(cow_id),
        "prediction": int(prediction),
        "confidence": round(float(confidence), CONFIDENCE_DECIMALS),
        "timestamp": int(timestamp),
        "payload": payload,
        "result_hash": "0x" + digest.hex(),
    }


def verify_record(record: dict) -> bool:
    """Recompute the hash from the record's own fields and compare.

    This is the tamper check. It answers "do these values still hash to the
    digest that was stored?" -- it does NOT by itself prove the digest is the one
    on-chain; pair it with oracle.fetch_onchain_hash for the full guarantee.

    Any record that cannot even be serialized -- a confidence of 1.01, a
    prediction of 2 -- is a failed verification, not an exception to propagate.
    """
    try:
        expected = hash_record(
            record["cow_id"], record["prediction"], record["confidence"], record["timestamp"]
        )["result_hash"]
    except (ValueError, TypeError, KeyError):
        return False
    return expected.lower() == str(record.get("result_hash", "")).lower()


if __name__ == "__main__":
    print(f"keccak256 backend: {KECCAK_BACKEND}\n")

    rec = hash_record("C0001", 1, 0.87, 1755000000)
    print("  payload:", rec["payload"])
    print("  hash   :", rec["result_hash"])
    print("  verify :", verify_record(rec))

    print("\n  tamper test - flip prediction 1 -> 0, keep the old hash:")
    tampered = dict(rec, prediction=0)
    print("  verify :", verify_record(tampered), "(False means tampering was caught)")

    print("\n  sensitivity - a 0.0001 confidence change must change the hash:")
    a = hash_record("C0001", 1, 0.8700, 1755000000)["result_hash"]
    b = hash_record("C0001", 1, 0.8701, 1755000000)["result_hash"]
    print(f"  0.8700 -> {a[:18]}...")
    print(f"  0.8701 -> {b[:18]}...")
    print("  different:", a != b)
