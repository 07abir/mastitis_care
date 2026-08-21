"""Blockchain-verified cattle mastitis detection.

Pipeline stages, each in its own module:
    config      central paths, schema and chain settings
    preprocess  step 1 - load, encode, scale, split
    train_model step 2 - five-model soft-voting ensemble
    predict     step 3 - one prediction record per cow
    hashing     step 4 - canonical JSON + keccak256
    oracle      step 5/6 - write hashes on-chain, confirm receipts
    database    step 6 - off-chain SQLite store
    api         step 7 - FastAPI layer for the dashboard
"""

__version__ = "1.0.0"
