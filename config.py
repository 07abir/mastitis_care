"""Central configuration: paths, column roles, model + chain settings.

Everything tunable lives here so the pipeline modules stay free of magic values.
"""
from pathlib import Path
import os

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "cow_milk_mastitis_dataset.csv"
ARTIFACTS = ROOT / "artifacts"
FRONTEND = ROOT / "frontend"

MODEL_PATH = ARTIFACTS / "ensemble_model.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"
PREDICTIONS_PATH = ARTIFACTS / "predictions.json"
RECORDS_PATH = ARTIFACTS / "onchain_records.json"
DB_PATH = ARTIFACTS / "health_records.db"
CONTRACT_INFO = ARTIFACTS / "contract.json"
MOCK_CHAIN_STATE = ARTIFACTS / "mock_chain.json"
# The frontend reads this when no API server is running, so the dashboard
# still opens as a plain file with real data in it.
FRONTEND_DATA = FRONTEND / "data.json"

# --------------------------------------------------------------------------
# dataset schema
# --------------------------------------------------------------------------
ID_COLUMN = "Cow_ID"
TARGET_COLUMN = "class1"

# Continuous sensor readings -> StandardScaler.
NUMERIC_FEATURES = [
    "Milk_Temperature",
    "Milk_pH",
    "Milk_Conductivity",
    "Somatic_Cell_Count",
    "Milk_Yield",
]

# Already 0/1, so scaling or one-hot encoding them adds nothing. Passed through.
BINARY_FEATURES = ["Clotting"]

# True categoricals -> OneHotEncoder. Empty for the current CSV, but the
# original project plan mentioned farm_id and season: if you ever add those
# columns they are picked up automatically (see preprocess.resolve_schema),
# so no code change is needed.
CATEGORICAL_FEATURES = []

# Candidate categorical names to auto-detect if present in the CSV.
AUTO_CATEGORICAL_CANDIDATES = ["farm_id", "Farm_ID", "season", "Season", "breed", "Breed"]

# 'Day' is the day-of-lactation index. Its correlation with the target is
# -0.002, and the best single threshold anywhere on it reaches only 0.774 --
# below the 0.789 majority-class base rate, so it is beaten by always guessing
# "healthy". It carries no signal, and keeping it only injects distance noise
# into KNN, so it is dropped by default.
# Set KEEP_DAY=1 in the environment to include it anyway.
DROP_FEATURES = [] if os.getenv("KEEP_DAY") == "1" else ["Day"]

# --------------------------------------------------------------------------
# model settings
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Equal weights reproduce a textbook soft-voting VotingClassifier.
# `python -m src.train_model --tune-weights` searches for better ones and
# writes them into the saved model. See README "Naive Bayes caveat".
VOTING_WEIGHTS = None  # None => equal weight for all five models

# Probability at or above which a cow is flagged mastitic.
DECISION_THRESHOLD = 0.5
# Predictions whose confidence falls below this are marked for manual vet review
# rather than trusted outright.
REVIEW_CONFIDENCE = 0.70

# --------------------------------------------------------------------------
# blockchain settings (read from .env - see .env.example)
# --------------------------------------------------------------------------
# "auto" tries the real chain and falls back to the mock chain if anything is
# missing or unreachable. Force with "real" or "mock".
CHAIN_MODE = os.getenv("CHAIN_MODE", "auto")

RPC_URL = os.getenv("POLYGON_AMOY_RPC", "https://rpc-amoy.polygon.technology")
CHAIN_ID = int(os.getenv("CHAIN_ID", "80002"))  # Polygon Amoy testnet
PRIVATE_KEY = os.getenv("ORACLE_PRIVATE_KEY", "")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS", "")
EXPLORER_TX_URL = os.getenv("EXPLORER_TX_URL", "https://amoy.polygonscan.com/tx/")

# How many cow records to push on-chain in a demo run. Each real transaction
# costs gas and takes a few seconds, so keep this small when CHAIN_MODE=real.
ONCHAIN_BATCH_SIZE = int(os.getenv("ONCHAIN_BATCH_SIZE", "10"))
TX_TIMEOUT_SECONDS = 180

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))


def ensure_dirs():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FRONTEND.mkdir(parents=True, exist_ok=True)
