# Mastitis screening, on the record

Mastitis is the most expensive disease in dairy farming, and the milk of an infected
cow looks exactly like the milk of a healthy one. A farm can test for it; the problem
is that a buyer has no way to know whether the result they were shown is the result
the test produced. A number in a spreadsheet can be edited after the fact, and there
is nothing in the number itself that says whether it was.

This project closes that gap. A five-model ensemble screens each cow from her milk
sensor readings, the result is reduced to a 32-byte keccak256 digest, and that digest
is written to an append-only Solidity contract on Polygon Amoy. The readings never
leave the farm's own database — only the hash goes on-chain. A buyer who is shown a
record can recompute its hash in their own browser and compare it against what the
contract recorded. If the two differ, the record was altered. If they match, it was
not.

The dashboard makes that check a button rather than a claim.

## The seven steps, and where each one lives

The pipeline follows the plan it was built from, one module per step.

Loading and preprocessing is `src/preprocess.py`. It reads
`cow_milk_mastitis_dataset.csv` with pandas, puts the five continuous sensor columns
through `StandardScaler`, one-hot encodes any true categoricals, passes the binary
`Clotting` flag through untouched, and makes a stratified 80/20 split. The scaler
lives inside a `Pipeline` so it is fitted on training folds only — fitting it on all
800 rows first is the most common way to leak test data into a model and get an
accuracy you cannot reproduce.

Training is `src/train_model.py`. KNN, logistic regression, an MLP, an SVM with
`probability=True` and Gaussian naive Bayes are combined by
`VotingClassifier(voting="soft")`, which averages the five predicted probabilities
rather than counting votes. Soft voting is the right choice here because the output
has to be a risk score, not just a label.

Prediction is `src/predict.py`, which emits one record per cow in the shape the plan
asked for — `{"cow_id": "C0001", "prediction": 1, "confidence": 0.87}` — and flags
anything below 0.70 confidence for manual vet review instead of trusting it.

Hashing is `src/hashing.py`. This is the step where being careless would quietly
destroy the whole guarantee, so the serialization is pinned: keys sorted, no
whitespace, confidence as a fixed four-decimal *string* rather than a float,
timestamp as whole unix seconds, UTF-8. Python's `str(0.87)` and JavaScript's
`String(0.87)` do not always agree, and a hash that only one language can reproduce
proves nothing to anyone else.

Writing to the chain is `src/oracle.py`, the bridge between the model and the ledger.
It signs an EIP-1559 transaction with web3.py, respects Polygon's 30 gwei priority-fee
floor, and can batch many cows into a single transaction because fifty separate
writes cost far more gas than one write of fifty records.

Confirming and storing is `src/oracle.py` plus `src/database.py`. The oracle waits
for the receipt, checks `status == 1`, and hands the transaction hash to SQLite so the
dashboard can answer "is this verified?" without re-querying the chain for every row.

The dashboard is `frontend/index.html` — a single file, no build step, no external
requests. It shows each cow's status with a verification seal and, for real on-chain
records, a link to Polygonscan.

Two components sit outside the original seven steps. `src/consensus.py` adds a 2-of-3
agreement gate between the ML ensemble, a clinical threshold rule and a sensor
plausibility check, so a single model failure cannot publish a clean bill of health on
its own; a cow whose three sources disagree is held back rather than sealed.
`src/mock_chain.py` is a local file-backed simulation of the contract, so a demo works
with no wifi, no faucet and no testnet — see the honesty note below about what it does
and does not prove.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env          # Windows: copy .env.example .env
python run_pipeline.py
```

That trains the ensemble, screens every cow, hashes the results, writes them to the
chain (mock mode by default), confirms them, stores them in SQLite and exports the
dashboard's data. It ends by printing the two ways to open the dashboard.

Useful flags: `--limit N` caps how many records are sent to the chain, `--batch` uses
the batch contract call, `--tune-weights` searches for better ensemble weights than
equal ones, `--skip-train` reuses the saved model, and `--mode real|mock|auto` forces
a chain backend.

To serve the dashboard with a live API instead of a static export:

```bash
uvicorn src.api:app --reload
```

Then open http://127.0.0.1:8000 — the page detects the API and prefers it. The
endpoints are `/api/health`, `/api/stats`, `/api/model`, `/api/records`,
`/api/records/{cow_id}`, `/api/verify/{cow_id}` and `POST /api/predict`, the last of
which scores a single new sensor reading.

### Going on-chain for real

```bash
python scripts/deploy_contract.py --compile-only    # do this first, see caveat below
python scripts/deploy_contract.py                   # deploys, writes artifacts/contract.json
# paste the printed CONTRACT_ADDRESS= line into .env
python run_pipeline.py --mode real --limit 10
```

You need a funded Amoy wallet; the faucet is at https://faucet.polygon.technology.
Keep `--limit` small on the real chain — each transaction costs gas and takes a few
seconds. The private key is read from `.env` and from nowhere else, is never written to
`artifacts/` or printed to the console, and should never be a key that has touched
mainnet funds.

### If scikit-learn cannot be installed

`verify/seed_demo.py` runs the same end-to-end flow with a from-scratch NumPy
implementation of all five models substituted for sklearn. It exists because this
project was developed in an environment with no PyPI access, and shipping untested
code was not an option. It produced the data currently in `frontend/data.json`, tagged
`"engine": "numpy-reference"` in `artifacts/metrics.json` so the numbers are never
mistaken for the real model's.

```bash
python verify/seed_demo.py --limit 120 --fresh
```

## Verifying it

```bash
python verify/test_pipeline.py             # 23 checks: preprocessing, hashing, consensus, db
python verify/test_contract_interface.py   # 34 checks: .sol vs ABI vs oracle.py
node   verify/test_keccak_js.js            # the JS keccak against known vectors
node   verify/test_frontend_hash.js        # the shipped page's hasher vs every real record
node   verify/test_frontend_render.js      # the page's own render functions, DOM stubbed
```

The third and fourth are the ones that matter most. `test_frontend_hash.js` extracts
the hashing code *out of the shipped `index.html`* — not from a copy kept in sync by
hand — and confirms it reproduces the exact payload string and the exact digest for
every record in the database. That is what makes "recompute it yourself in your
browser" a true statement rather than a decorative one. `test_frontend_render.js`
stubs a minimal DOM and runs the page's real render functions against real data,
because there is no browser in the development environment; it asserts, among other
things, that a mock record can never render as verified.

## What the numbers actually mean

Held-out accuracy is 1.0000. Five-fold cross-validation is 1.0000 ± 0.0000. Every
individual model also scores 1.0000. **Do not read that as a strong result.** This
dataset is nearly linearly separable: a single threshold on somatic cell count alone
already reaches 99.88%, and one on milk conductivity reaches the same, so a perfect
score says more about the data than about the ensemble. Reporting it without that
sentence attached would be misleading.

The honest measurement is what happens when the sensors are noisy, which on a real
farm they are. Gaussian noise at multiples of each feature's training standard
deviation was added to the test set:

| noise (× train σ) | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|
| KNN | 1.0000 | 1.0000 | 0.9925 | 0.9637 | 0.9200 | 0.8369 |
| Logistic regression | 1.0000 | 1.0000 | 0.9863 | 0.9444 | 0.8944 | 0.8187 |
| MLP | 1.0000 | 1.0000 | 0.9756 | 0.9275 | 0.8887 | 0.8137 |
| SVM | 1.0000 | 1.0000 | 0.9931 | 0.9713 | 0.9300 | 0.8469 |
| Naive Bayes | 1.0000 | 1.0000 | 0.9725 | 0.8631 | 0.7412 | 0.5825 |
| **Ensemble** | 1.0000 | 1.0000 | 0.9888 | 0.9481 | 0.9056 | 0.8213 |

Two things follow. Naive Bayes is the weak link — it collapses to 0.5825 at 3σ while
SVM still holds 0.8469 — because its independence assumption is badly wrong for these
features, where conductivity, somatic cell count and pH all move together during an
infection. And because the ensemble weights all five models equally, naive Bayes drags
the ensemble slightly *below* its best members under noise. `--tune-weights` searches
for weights that fix this; the default is left at equal weights so the reported numbers
match a textbook `VotingClassifier`.

The dataset has 800 cows, 169 of them mastitic (21.1% positive), with six features
after `Day` is dropped. `Day` is discarded because it carries no signal: its
correlation with the target is −0.002, and the best single threshold anywhere on it
reaches 0.774, which is *below* the 0.789 majority-class base rate — you do better by
ignoring it and guessing "healthy" every time. Keeping it only adds distance noise to
KNN. Set `KEEP_DAY=1` to include it anyway.

The original plan named `farm_id` and `season` as categorical columns to encode. The
real CSV contains neither. Rather than fake them, `src/config.py` auto-detects those
names if they are ever added to the data, so the preprocessing step needs no edit when
a multi-farm dataset arrives.

## Things this project does not prove

**Mock-mode records are not blockchain proof.** In mock mode there is no consensus, no
independent party and no immutability — anyone who can write to
`artifacts/mock_chain.json` can rewrite history. Those records are tagged `"mock"`
everywhere they surface, carry no explorer link, and the dashboard labels them
"Simulated locally" rather than verified. The render test enforces this. Do not present
them to judges as on-chain proof; run `--mode real` for that.

**The contract has never been compiled here.** `pip install py-solc-x` and `npm view
solc` are both blocked in the development sandbox, so `contracts/CattleHealthRegistry.sol`
has not been through solc. `verify/test_contract_interface.py` narrows the gap by
checking the contract, the ABI in `src/contract_abi.py` and the calls in
`src/oracle.py` all still describe the same interface — including that the write
functions are gated by `onlyOracle`, that the history is only ever appended to, and
that the pinned solc version satisfies the pragma. But a consistency check is not a
compiler. Run `python scripts/deploy_contract.py --compile-only` before you deploy.

**The consensus layer has never fired on this data.** Across all 800 cows there were
zero disputes, because the clinical thresholds and the model agree everywhere on a
dataset this clean. It is exercised only by a deliberately contradictory synthetic cow
in `verify/test_pipeline.py`. The layer is real and it works; it just has nothing to
catch here, and it would be dishonest to present it as visibly doing work.

**Hash verification is not the same as chain verification.** Recomputing a hash proves
the record's fields still match its stored digest. Proving the digest is the one the
contract holds requires reading the contract, which is what `/api/verify/{cow_id}` and
`oracle.fetch_onchain_hash` do. The dashboard only shows the full "verified on chain"
seal when both checks pass and the chain mode is real.

## Layout

```
cow_milk_mastitis_dataset.csv     800 cows, 6 usable features
run_pipeline.py                   steps 1-7 in one command
requirements.txt  .env.example
contracts/CattleHealthRegistry.sol   append-only registry, custom errors, oracle ACL
scripts/deploy_contract.py           compile, deploy, authorize
src/  preprocess  train_model  predict  hashing  consensus
      oracle  mock_chain  database  api  config  contract_abi  keccak_fallback
frontend/index.html              the dashboard; data.json / data.js are generated
verify/                          five test suites + the NumPy reference ensemble
artifacts/                       generated: model, metrics, sqlite db, mock chain
```

`src/keccak_fallback.py` is a pure-Python keccak256 used only if neither web3 nor
eth-hash is installed. It is the Ethereum variant — padding byte `0x01`, not SHA3's
`0x06`, a distinction that silently produces wrong-but-plausible digests if you get it
backwards.
