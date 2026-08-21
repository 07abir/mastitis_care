"""Local mock chain - a stand-in for the deployed contract.

Why this exists: a hackathon demo should not depend on conference wifi, faucet
availability or testnet uptime. This module mirrors CattleHealthRegistry's
behaviour exactly (append-only history, keccak256-keyed lookups, the same
verifyRecord semantics) and persists to a JSON file, so `python run_pipeline.py`
produces a complete, working end-to-end demo with zero setup.

What it is NOT: a real blockchain. There is no consensus, no immutability
guarantee and no independent party. Anyone with write access to
artifacts/mock_chain.json can rewrite it. Records produced in mock mode are
tagged `"mode": "mock"` everywhere they surface, and the dashboard shows them as
"Local (unverified)" rather than "Verified on blockchain" -- do not present mock
records to judges as on-chain proof.

Switch to the real thing by filling in .env and running with CHAIN_MODE=real.
"""
from __future__ import annotations

import json
import time

from . import config
from .hashing import _keccak


class MockChain:
    """File-backed simulation of the registry contract."""

    mode = "mock"

    def __init__(self, state_path=None):
        self.path = state_path or config.MOCK_CHAIN_STATE
        self.state = self._load()

    # -- persistence -------------------------------------------------------
    def _load(self):
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except json.JSONDecodeError:
                print(f"  warning: {self.path} was corrupt, starting a fresh mock chain")
        return {
            "block_number": 1_000_000,
            "nonce": 0,
            "total_records": 0,
            "history": {},   # cow_id -> [record, ...]
            "txs": {},       # tx_hash -> record
        }

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2))

    def reset(self):
        """Wipe the mock chain back to genesis.

        Overwrites rather than deletes: on some synced folders (OneDrive, Dropbox)
        deletion is blocked while writes are fine, and this needs to work there.
        """
        self.state = {
            "block_number": 1_000_000,
            "nonce": 0,
            "total_records": 0,
            "history": {},
            "txs": {},
        }
        self._save()

    # -- writes ------------------------------------------------------------
    def store_health_record(self, cow_id: str, result_hash: str, timestamp: int) -> dict:
        """Mirror of storeHealthRecord. Returns a receipt-shaped dict."""
        if not cow_id:
            raise ValueError("cowId must not be empty (contract reverts with EmptyCowId)")
        if not result_hash or int(result_hash, 16) == 0:
            raise ValueError("resultHash must not be zero (contract reverts with ZeroHash)")

        self.state["nonce"] += 1
        self.state["block_number"] += 1

        # Derive a realistic-looking 32-byte tx hash from the call data, so the
        # same input always yields the same tx id and the demo is reproducible.
        seed = f"{self.state['nonce']}|{cow_id}|{result_hash}|{timestamp}".encode()
        tx_hash = "0x" + _keccak(seed).hex()

        entry = {
            "cow_id": cow_id,
            "result_hash": result_hash,
            "timestamp": int(timestamp),
            "stored_at": int(time.time()),
            "tx_hash": tx_hash,
            "block_number": self.state["block_number"],
            "index": len(self.state["history"].get(cow_id, [])),
        }

        self.state["history"].setdefault(cow_id, []).append(entry)
        self.state["txs"][tx_hash] = entry
        self.state["total_records"] += 1
        self._save()

        return {
            "tx_hash": tx_hash,
            "block_number": entry["block_number"],
            "status": 1,
            "gas_used": 0,
            "mode": "mock",
            "explorer_url": None,  # nothing to link to; there is no real explorer
        }

    # -- reads -------------------------------------------------------------
    def latest_record(self, cow_id: str):
        h = self.state["history"].get(cow_id, [])
        return h[-1] if h else None

    def record_count(self, cow_id: str) -> int:
        return len(self.state["history"].get(cow_id, []))

    def verify_record(self, cow_id: str, result_hash: str):
        """Mirror of verifyRecord: newest-first scan of the cow's history."""
        for entry in reversed(self.state["history"].get(cow_id, [])):
            if entry["result_hash"].lower() == str(result_hash).lower():
                return True, entry["timestamp"], entry["index"]
        return False, 0, 0

    @property
    def total_records(self) -> int:
        return self.state["total_records"]


if __name__ == "__main__":
    from .hashing import hash_record, utc_timestamp

    chain = MockChain()
    rec = hash_record("C0001", 1, 0.9123, utc_timestamp())
    receipt = chain.store_health_record(rec["cow_id"], rec["result_hash"], rec["timestamp"])
    print("  stored :", receipt["tx_hash"], "block", receipt["block_number"])
    print("  verify (correct hash) :", chain.verify_record("C0001", rec["result_hash"])[0])
    print("  verify (tampered hash):", chain.verify_record("C0001", "0x" + "de" * 32)[0])
    print("  total records on mock chain:", chain.total_records)
