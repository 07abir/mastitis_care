"""ABI for CattleHealthRegistry.

This is the hand-maintained fallback. When you deploy with
`python scripts/deploy_contract.py`, solc emits the authoritative ABI into
artifacts/contract.json and the oracle prefers that. Keep this in sync if you
change the contract's external interface.
"""

CONTRACT_ABI = [
    {
        "inputs": [],
        "stateMutability": "nonpayable",
        "type": "constructor",
    },
    # ---- writes ----------------------------------------------------------
    {
        "inputs": [
            {"internalType": "string", "name": "cowId", "type": "string"},
            {"internalType": "bytes32", "name": "resultHash", "type": "bytes32"},
            {"internalType": "uint64", "name": "timestamp", "type": "uint64"},
        ],
        "name": "storeHealthRecord",
        "outputs": [{"internalType": "uint256", "name": "index", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "string[]", "name": "cowIds", "type": "string[]"},
            {"internalType": "bytes32[]", "name": "resultHashes", "type": "bytes32[]"},
            {"internalType": "uint64[]", "name": "timestamps", "type": "uint64[]"},
        ],
        "name": "storeHealthRecordBatch",
        "outputs": [{"internalType": "uint256", "name": "stored", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "oracle", "type": "address"}],
        "name": "authorizeOracle",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "oracle", "type": "address"}],
        "name": "revokeOracle",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "newOwner", "type": "address"}],
        "name": "transferOwnership",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    # ---- reads -----------------------------------------------------------
    {
        "inputs": [{"internalType": "string", "name": "cowId", "type": "string"}],
        "name": "latestRecord",
        "outputs": [
            {"internalType": "bytes32", "name": "resultHash", "type": "bytes32"},
            {"internalType": "uint64", "name": "timestamp", "type": "uint64"},
            {"internalType": "uint64", "name": "storedAt", "type": "uint64"},
            {"internalType": "address", "name": "oracle", "type": "address"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "string", "name": "cowId", "type": "string"},
            {"internalType": "uint256", "name": "index", "type": "uint256"},
        ],
        "name": "recordAt",
        "outputs": [
            {"internalType": "bytes32", "name": "resultHash", "type": "bytes32"},
            {"internalType": "uint64", "name": "timestamp", "type": "uint64"},
            {"internalType": "uint64", "name": "storedAt", "type": "uint64"},
            {"internalType": "address", "name": "oracle", "type": "address"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "string", "name": "cowId", "type": "string"},
            {"internalType": "bytes32", "name": "resultHash", "type": "bytes32"},
        ],
        "name": "verifyRecord",
        "outputs": [
            {"internalType": "bool", "name": "found", "type": "bool"},
            {"internalType": "uint64", "name": "timestamp", "type": "uint64"},
            {"internalType": "uint256", "name": "index", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "string", "name": "cowId", "type": "string"}],
        "name": "recordCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "string", "name": "cowId", "type": "string"}],
        "name": "cowKey",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "pure",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "", "type": "address"}],
        "name": "authorizedOracles",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "owner",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "totalRecords",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    # ---- events ----------------------------------------------------------
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "cowKey", "type": "bytes32"},
            {"indexed": False, "internalType": "string", "name": "cowId", "type": "string"},
            {"indexed": False, "internalType": "bytes32", "name": "resultHash", "type": "bytes32"},
            {"indexed": False, "internalType": "uint64", "name": "timestamp", "type": "uint64"},
            {"indexed": False, "internalType": "uint64", "name": "storedAt", "type": "uint64"},
            {"indexed": True, "internalType": "address", "name": "oracle", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "index", "type": "uint256"},
        ],
        "name": "HealthRecordStored",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "oracle", "type": "address"}
        ],
        "name": "OracleAuthorized",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "oracle", "type": "address"}
        ],
        "name": "OracleRevoked",
        "type": "event",
    },
]
