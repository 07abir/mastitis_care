"""Compile and deploy CattleHealthRegistry to Polygon Amoy.

Uses py-solc-x, which downloads a standalone solc binary, so there is no
Node/Hardhat/Foundry toolchain to install.

    python scripts/deploy_contract.py                 # compile + deploy
    python scripts/deploy_contract.py --compile-only  # just check it compiles
    python scripts/deploy_contract.py --authorize 0x… # add another oracle later

On success it writes artifacts/contract.json (address + ABI) and prints the
CONTRACT_ADDRESS line to paste into .env.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src import config  # noqa: E402

SOLC_VERSION = "0.8.24"
SOURCE = ROOT / "contracts" / "CattleHealthRegistry.sol"


def compile_contract() -> tuple[list, str]:
    """Return (abi, bytecode) for CattleHealthRegistry."""
    try:
        import solcx
    except ImportError:
        raise SystemExit(
            "py-solc-x is not installed. Run:  pip install py-solc-x\n"
            "(It fetches a standalone solc binary; no Node.js required.)"
        )

    installed = [str(v) for v in solcx.get_installed_solc_versions()]
    if SOLC_VERSION not in installed:
        print(f"  downloading solc {SOLC_VERSION} (one time)…")
        solcx.install_solc(SOLC_VERSION)

    print(f"  compiling {SOURCE.name} with solc {SOLC_VERSION}")
    compiled = solcx.compile_standard(
        {
            "language": "Solidity",
            "sources": {SOURCE.name: {"content": SOURCE.read_text()}},
            "settings": {
                # 200 runs: the registry is written far more often than deployed,
                # so optimise for call cost rather than deploy cost.
                "optimizer": {"enabled": True, "runs": 200},
                "outputSelection": {"*": {"*": ["abi", "evm.bytecode.object"]}},
            },
        },
        solc_version=SOLC_VERSION,
    )

    contract = compiled["contracts"][SOURCE.name]["CattleHealthRegistry"]
    abi = contract["abi"]
    bytecode = contract["evm"]["bytecode"]["object"]
    print(f"  compiled: {len(abi)} ABI entries, {len(bytecode) // 2} bytes of bytecode")
    return abi, bytecode


def connect():
    """Return (w3, account) or exit with a readable reason."""
    try:
        from web3 import Web3
    except ImportError:
        raise SystemExit("web3 is not installed. Run:  pip install 'web3>=6.11'")

    if not config.PRIVATE_KEY:
        raise SystemExit(
            "ORACLE_PRIVATE_KEY is not set.\n"
            "  1. cp .env.example .env\n"
            "  2. paste a throwaway wallet's private key into ORACLE_PRIVATE_KEY\n"
            "  3. fund it at https://faucet.polygon.technology (select Amoy)"
        )

    w3 = Web3(Web3.HTTPProvider(config.RPC_URL, request_kwargs={"timeout": 60}))
    if not w3.is_connected():
        raise SystemExit(f"Could not reach the RPC endpoint at {config.RPC_URL}")

    acct = w3.eth.account.from_key(config.PRIVATE_KEY)
    balance = w3.eth.get_balance(acct.address)
    print(f"  chain id : {w3.eth.chain_id}")
    print(f"  deployer : {acct.address}")
    print(f"  balance  : {w3.from_wei(balance, 'ether'):.4f} POL")

    if balance == 0:
        raise SystemExit(
            f"{acct.address} holds 0 POL, so it cannot pay for deployment.\n"
            f"Get free test funds at https://faucet.polygon.technology (select Amoy)."
        )
    return w3, acct


def _fees(w3):
    """EIP-1559 fees respecting Polygon's 30 gwei priority-fee floor."""
    gwei = 10**9
    floor = 30 * gwei
    try:
        base = w3.eth.get_block("latest")["baseFeePerGas"]
        try:
            priority = max(int(w3.eth.max_priority_fee), floor)
        except Exception:
            priority = floor
        return {"maxFeePerGas": base * 2 + priority, "maxPriorityFeePerGas": priority}
    except Exception:
        return {"gasPrice": max(int(w3.eth.gas_price), floor)}


def _send(w3, acct, tx):
    signed = acct.sign_transaction(tx)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    tx_hash = w3.eth.send_raw_transaction(raw)
    print(f"  tx sent  : {tx_hash.hex()}")
    print("  waiting for confirmation…")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
    if receipt["status"] != 1:
        raise SystemExit(f"transaction reverted: {tx_hash.hex()}")
    return receipt


def deploy() -> None:
    abi, bytecode = compile_contract()
    w3, acct = connect()

    factory = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx = factory.constructor().build_transaction(
        {
            "from": acct.address,
            "nonce": w3.eth.get_transaction_count(acct.address, "pending"),
            "chainId": config.CHAIN_ID,
            **_fees(w3),
        }
    )
    try:
        tx["gas"] = int(w3.eth.estimate_gas(tx) * 1.2)
    except Exception:
        tx["gas"] = 3_000_000

    receipt = _send(w3, acct, tx)
    address = receipt["contractAddress"]

    config.ensure_dirs()
    config.CONTRACT_INFO.write_text(
        json.dumps(
            {
                "address": address,
                "chain_id": config.CHAIN_ID,
                "deployer": acct.address,
                "block_number": int(receipt["blockNumber"]),
                "tx_hash": receipt["transactionHash"].hex(),
                "solc_version": SOLC_VERSION,
                "abi": abi,
            },
            indent=2,
        )
    )

    print(f"\n  deployed at    : {address}")
    print(f"  block          : {receipt['blockNumber']}")
    print(f"  gas used       : {receipt['gasUsed']:,}")
    print(f"  explorer       : https://amoy.polygonscan.com/address/{address}")
    print(f"  abi + address  : {config.CONTRACT_INFO}")
    print("\n  Add this line to your .env, then run: python run_pipeline.py --mode real")
    print(f"\n    CONTRACT_ADDRESS={address}\n")
    print("  The deployer is already an authorized oracle, so no extra step is")
    print("  needed if the same key signs the health records.")


def authorize(new_oracle: str) -> None:
    """Let another wallet write records - e.g. a second machine at the demo."""
    if not config.CONTRACT_ADDRESS:
        raise SystemExit("CONTRACT_ADDRESS is not set in .env. Deploy first.")

    abi, _ = compile_contract()
    w3, acct = connect()
    from web3 import Web3

    contract = w3.eth.contract(
        address=Web3.to_checksum_address(config.CONTRACT_ADDRESS), abi=abi
    )
    target = Web3.to_checksum_address(new_oracle)

    if contract.functions.authorizedOracles(target).call():
        print(f"  {target} is already authorized. Nothing to do.")
        return

    tx = contract.functions.authorizeOracle(target).build_transaction(
        {
            "from": acct.address,
            "nonce": w3.eth.get_transaction_count(acct.address, "pending"),
            "chainId": config.CHAIN_ID,
            **_fees(w3),
        }
    )
    try:
        tx["gas"] = int(w3.eth.estimate_gas(tx) * 1.2)
    except Exception:
        tx["gas"] = 120_000

    _send(w3, acct, tx)
    print(f"  {target} can now write health records.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Deploy CattleHealthRegistry to Polygon Amoy.")
    ap.add_argument("--compile-only", action="store_true",
                    help="compile and stop; no wallet or network needed")
    ap.add_argument("--authorize", metavar="ADDRESS",
                    help="authorize another wallet as an oracle")
    a = ap.parse_args()

    if not SOURCE.exists():
        raise SystemExit(f"Contract source not found: {SOURCE}")

    if a.compile_only:
        abi, bytecode = compile_contract()
        config.ensure_dirs()
        (config.ARTIFACTS / "contract_abi.json").write_text(json.dumps(abi, indent=2))
        print(f"  ABI written to {config.ARTIFACTS / 'contract_abi.json'}")
        print("  compile-only: nothing was deployed.")
    elif a.authorize:
        authorize(a.authorize)
    else:
        deploy()


if __name__ == "__main__":
    main()
