"""Steps 5 and 6 - the oracle: bridge ML output onto the chain, then confirm it.

This is the piece that makes the project more than a classifier. For each cow it:

  1. runs the 2-of-3 consensus check (src/consensus.py) and holds back disputes
  2. calls storeHealthRecord(cowId, resultHash, timestamp) on the registry
  3. waits for the transaction receipt
  4. writes the tx hash and result into the off-chain SQLite store, so the
     dashboard can render "verified" instantly without re-querying the chain

Two backends behind one interface. CHAIN_MODE=real talks to Polygon Amoy through
web3.py; CHAIN_MODE=mock uses the local simulation; CHAIN_MODE=auto (the default)
tries real, explains precisely why it could not connect, and falls back to mock so
a demo still runs end to end.

Usage
    python -m src.oracle                    # honour CHAIN_MODE from .env
    python -m src.oracle --mode mock        # force the local simulation
    python -m src.oracle --mode real --limit 5
    python -m src.oracle --verify C0001     # re-check one cow against the chain
"""
from __future__ import annotations

import argparse
import json
import time

from . import config, consensus
from .contract_abi import CONTRACT_ABI
from .mock_chain import MockChain

GWEI = 10**9
# Polygon's validators enforce a priority-fee floor; bidding under it means the
# transaction simply never gets picked up.
MIN_PRIORITY_FEE = 30 * GWEI


def _hash_to_bytes32(result_hash: str) -> bytes:
    h = str(result_hash)
    if h.startswith("0x"):
        h = h[2:]
    b = bytes.fromhex(h)
    if len(b) != 32:
        raise ValueError(f"result_hash must be 32 bytes, got {len(b)}")
    return b


class ChainUnavailable(RuntimeError):
    """Raised when the real chain cannot be used, with the reason attached."""


# --------------------------------------------------------------------------
# real backend
# --------------------------------------------------------------------------
class RealChain:
    """Polygon Amoy backend via web3.py."""

    mode = "real"

    def __init__(self):
        try:
            from web3 import Web3
        except ImportError as e:
            raise ChainUnavailable(
                "web3 is not installed. Run: pip install 'web3>=6.11'"
            ) from e

        self.Web3 = Web3

        if not config.PRIVATE_KEY:
            raise ChainUnavailable(
                "ORACLE_PRIVATE_KEY is not set. Copy .env.example to .env and fill it in."
            )
        if not config.CONTRACT_ADDRESS:
            raise ChainUnavailable(
                "CONTRACT_ADDRESS is not set. Deploy first: python scripts/deploy_contract.py"
            )

        self.w3 = Web3(Web3.HTTPProvider(config.RPC_URL, request_kwargs={"timeout": 30}))
        if not self.w3.is_connected():
            raise ChainUnavailable(f"Could not reach the RPC endpoint at {config.RPC_URL}")

        self.account = self.w3.eth.account.from_key(config.PRIVATE_KEY)
        self.address = self.account.address

        abi = CONTRACT_ABI
        if config.CONTRACT_INFO.exists():
            # Prefer the ABI solc actually produced at deploy time.
            try:
                abi = json.loads(config.CONTRACT_INFO.read_text())["abi"]
            except (KeyError, json.JSONDecodeError):
                pass

        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(config.CONTRACT_ADDRESS), abi=abi
        )

        self.balance = self.w3.eth.get_balance(self.address)
        if self.balance == 0:
            raise ChainUnavailable(
                f"Oracle wallet {self.address} holds 0 POL. Get free testnet funds "
                f"from https://faucet.polygon.technology (select Amoy)."
            )

        # Track the nonce locally so a batch of sends does not collide.
        self._nonce = self.w3.eth.get_transaction_count(self.address, "pending")

        print(f"  connected to chain id {self.w3.eth.chain_id} at {config.RPC_URL}")
        print(f"  oracle wallet  : {self.address}")
        print(f"  balance        : {self.w3.from_wei(self.balance, 'ether'):.4f} POL")
        print(f"  contract       : {config.CONTRACT_ADDRESS}")

        if not self.contract.functions.authorizedOracles(self.address).call():
            raise ChainUnavailable(
                f"{self.address} is not an authorized oracle on this contract. "
                f"The contract owner must call authorizeOracle({self.address})."
            )

    # -- fees --------------------------------------------------------------
    def _fee_fields(self) -> dict:
        """EIP-1559 fees, respecting Polygon's priority-fee floor."""
        try:
            base = self.w3.eth.get_block("latest").get("baseFeePerGas")
            if base is None:
                raise KeyError("no baseFeePerGas")
            try:
                priority = max(int(self.w3.eth.max_priority_fee), MIN_PRIORITY_FEE)
            except Exception:
                priority = MIN_PRIORITY_FEE
            # 2x base leaves headroom for a fee bump between build and inclusion
            return {"maxFeePerGas": base * 2 + priority, "maxPriorityFeePerGas": priority}
        except Exception:
            # Pre-1559 or an RPC that will not report the base fee
            return {"gasPrice": max(int(self.w3.eth.gas_price), MIN_PRIORITY_FEE)}

    def _send(self, fn):
        tx = fn.build_transaction(
            {
                "from": self.address,
                "nonce": self._nonce,
                "chainId": config.CHAIN_ID,
                **self._fee_fields(),
            }
        )
        try:
            # 25% headroom over the estimate; storage writes vary with history depth
            tx["gas"] = int(self.w3.eth.estimate_gas(tx) * 1.25)
        except Exception:
            tx["gas"] = 300_000

        signed = self.account.sign_transaction(tx)
        # web3 v7 renamed rawTransaction -> raw_transaction
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        tx_hash = self.w3.eth.send_raw_transaction(raw)
        self._nonce += 1

        # ---- step 6: wait for confirmation --------------------------------
        receipt = self.w3.eth.wait_for_transaction_receipt(
            tx_hash, timeout=config.TX_TIMEOUT_SECONDS
        )
        if receipt["status"] != 1:
            raise RuntimeError(f"transaction reverted on-chain: {tx_hash.hex()}")

        h = receipt["transactionHash"].hex()
        if not h.startswith("0x"):
            h = "0x" + h
        return {
            "tx_hash": h,
            "block_number": int(receipt["blockNumber"]),
            "status": int(receipt["status"]),
            "gas_used": int(receipt["gasUsed"]),
            "mode": "real",
            "explorer_url": config.EXPLORER_TX_URL + h,
        }

    # -- writes ------------------------------------------------------------
    def store_health_record(self, cow_id, result_hash, timestamp):
        return self._send(
            self.contract.functions.storeHealthRecord(
                str(cow_id), _hash_to_bytes32(result_hash), int(timestamp)
            )
        )

    def store_batch(self, records):
        """One transaction for many cows - far cheaper than one tx each."""
        return self._send(
            self.contract.functions.storeHealthRecordBatch(
                [str(r["cow_id"]) for r in records],
                [_hash_to_bytes32(r["result_hash"]) for r in records],
                [int(r["timestamp"]) for r in records],
            )
        )

    # -- reads -------------------------------------------------------------
    def verify_record(self, cow_id, result_hash):
        found, ts, idx = self.contract.functions.verifyRecord(
            str(cow_id), _hash_to_bytes32(result_hash)
        ).call()
        return bool(found), int(ts), int(idx)

    def record_count(self, cow_id):
        """How many screenings this cow has on the registry.

        MockChain has always had this; RealChain did not, which meant any future
        caller would work offline and break the moment the demo went on-chain.
        The two backends have to stay interchangeable -- submit_records() holds
        one of them without knowing which.
        """
        return int(self.contract.functions.recordCount(str(cow_id)).call())

    def latest_record(self, cow_id):
        try:
            h, ts, stored, oracle = self.contract.functions.latestRecord(str(cow_id)).call()
        except Exception:
            return None  # NoRecord()
        return {
            "cow_id": cow_id,
            "result_hash": "0x" + h.hex() if isinstance(h, bytes) else str(h),
            "timestamp": int(ts),
            "stored_at": int(stored),
            "oracle": oracle,
        }

    @property
    def total_records(self):
        return int(self.contract.functions.totalRecords().call())


# --------------------------------------------------------------------------
# backend selection
# --------------------------------------------------------------------------
def get_chain(mode=None):
    """Return a chain backend. In 'auto' mode, explain any fallback out loud."""
    mode = (mode or config.CHAIN_MODE).lower()

    if mode == "mock":
        print("  chain mode: MOCK (local simulation, not a real blockchain)")
        return MockChain()

    try:
        return RealChain()
    except ChainUnavailable as e:
        if mode == "real":
            raise
        print(f"  real chain unavailable: {e}")
        print("  falling back to MOCK mode - records will NOT be independently verifiable")
        return MockChain()


# --------------------------------------------------------------------------
# the oracle run
# --------------------------------------------------------------------------
def submit_records(records, raw_rows=None, mode=None, limit=None, use_batch=False):
    """Push prediction records on-chain and persist the receipts off-chain.

    ``raw_rows`` maps cow_id -> raw sensor reading, enabling the consensus check.
    Cows whose sources disagree are skipped rather than published.
    """
    from .database import init_db, insert_record

    config.ensure_dirs()
    chain = get_chain(mode)
    conn = init_db()

    todo = records[: (limit or config.ONCHAIN_BATCH_SIZE)]
    stored, skipped = [], []

    print(f"\nSTEP 5 - writing {len(todo)} record(s) to the registry")

    # ---- consensus gate --------------------------------------------------
    eligible = []
    for rec in todo:
        row = (raw_rows or {}).get(rec["cow_id"])
        if row is None:
            rec["consensus_agreed"] = None  # cannot check without raw readings
            eligible.append(rec)
            continue

        c = consensus.evaluate(rec["prediction"], row, rec.get("confidence"))
        rec.update(c.as_dict())
        if c.agreed:
            eligible.append(rec)
        else:
            skipped.append(rec)
            why = ("model outvoted by the sensor sources"
                   if c.ml_outvoted else "no clear majority")
            print(f"  {rec['cow_id']}: {why} "
                  f"({c.votes_for}/{len(c.votes)} voted mastitic) - held for vet review")

    if skipped:
        print(f"  {len(skipped)} record(s) withheld from the chain pending review")

    # ---- write -----------------------------------------------------------
    if use_batch and hasattr(chain, "store_batch") and len(eligible) > 1:
        receipt = chain.store_batch(eligible)
        print(f"  batched {len(eligible)} records in one tx {receipt['tx_hash']}")
        for rec in eligible:
            rec.update({k: receipt[k] for k in
                        ("tx_hash", "block_number", "gas_used", "mode", "explorer_url")})
            insert_record(conn, rec)
            stored.append(rec)
    else:
        for i, rec in enumerate(eligible, 1):
            try:
                receipt = chain.store_health_record(
                    rec["cow_id"], rec["result_hash"], rec["timestamp"]
                )
            except Exception as e:
                print(f"  {rec['cow_id']}: FAILED - {e}")
                continue

            rec.update({k: receipt[k] for k in
                        ("tx_hash", "block_number", "gas_used", "mode", "explorer_url")})

            # ---- step 6: confirm, then persist off-chain -------------------
            ok, onchain_ts, _ = chain.verify_record(rec["cow_id"], rec["result_hash"])
            rec["onchain_verified"] = bool(ok)

            insert_record(conn, rec)
            stored.append(rec)

            flag = "verified" if ok else "NOT FOUND ON CHAIN"
            print(f"  [{i}/{len(eligible)}] {rec['cow_id']} "
                  f"pred={rec['prediction']} conf={rec['confidence']:.3f} "
                  f"block={rec['block_number']} {flag}")
            if chain.mode == "real":
                time.sleep(0.4)  # be gentle with public RPC rate limits

    print(f"\nSTEP 6 - confirmed and stored off-chain")
    print(f"  written to chain    : {len(stored)}")
    print(f"  withheld (no consensus): {len(skipped)}")
    print(f"  database            : {config.DB_PATH}")
    print(f"  total on registry   : {chain.total_records}")

    config.RECORDS_PATH.write_text(json.dumps(stored, indent=2))
    conn.close()
    return stored, skipped


def main():
    ap = argparse.ArgumentParser(description="Oracle: write ML results on-chain.")
    ap.add_argument("--mode", choices=["auto", "real", "mock"], default=None)
    ap.add_argument("--limit", type=int, default=None, help="how many cows to submit")
    ap.add_argument("--batch", action="store_true", help="use one batched transaction")
    ap.add_argument("--verify", metavar="COW_ID", help="re-check one cow against the chain")
    a = ap.parse_args()

    if a.verify:
        chain = get_chain(a.mode)
        rec = chain.latest_record(a.verify)
        if not rec:
            print(f"  no on-chain record for {a.verify}")
            return
        print(f"  cow            : {a.verify}")
        print(f"  on-chain hash  : {rec['result_hash']}")
        print(f"  screened at    : {rec['timestamp']}")
        ok, _, idx = chain.verify_record(a.verify, rec["result_hash"])
        print(f"  hash confirmed : {ok} (record index {idx})")
        return

    if not config.PREDICTIONS_PATH.exists():
        raise SystemExit("No predictions found. Run: python -m src.predict")

    records = json.loads(config.PREDICTIONS_PATH.read_text())

    # Raw readings power the consensus check.
    import pandas as pd

    df = pd.read_csv(config.DATA_CSV)
    raw_rows = {str(r[config.ID_COLUMN]): r for _, r in df.iterrows()}

    submit_records(records, raw_rows=raw_rows, mode=a.mode, limit=a.limit, use_batch=a.batch)


if __name__ == "__main__":
    main()
