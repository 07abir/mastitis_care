"""Step 2 - train the soft-voting ensemble.

Five base learners (KNN, Logistic Regression, MLP, SVM, Gaussian Naive Bayes)
are combined with a VotingClassifier in soft-voting mode, which averages their
predicted probabilities to produce one mastitis risk score per cow.

Layout is a single Pipeline:  ColumnTransformer -> VotingClassifier
so the scaler is fitted on training folds only, with no leakage into validation.

Usage
    python -m src.train_model                 # textbook equal-weight soft voting
    python -m src.train_model --tune-weights  # search voting weights (see README)
    python -m src.train_model --no-noise-test # skip the robustness sweep
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from . import config
from .preprocess import build_preprocessor, prepare

RS = config.RANDOM_STATE


def base_estimators():
    """The five models. Every one exposes predict_proba, required by soft voting."""
    return [
        # distance weighting lets nearer cows dominate the vote
        ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance")),
        ("logreg", LogisticRegression(max_iter=2000, random_state=RS)),
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                max_iter=2000,
                early_stopping=True,
                n_iter_no_change=25,
                random_state=RS,
            ),
        ),
        # probability=True fits Platt scaling over the margins; without it an SVM
        # has no calibrated probability for soft voting to average.
        ("svm", SVC(kernel="rbf", C=1.0, probability=True, random_state=RS)),
        ("nb", GaussianNB()),
    ]


def build_model(schema, weights=None) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(schema)),
            (
                "ensemble",
                VotingClassifier(
                    estimators=base_estimators(),
                    voting="soft",  # average probabilities, not hard votes
                    weights=weights,
                ),
            ),
        ]
    )


def score_all(y_true, proba, threshold=config.DECISION_THRESHOLD) -> dict:
    y_pred = (proba >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    # roc_auc_score raises if the split happens to contain a single class
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    except ValueError:
        out["roc_auc"] = None
    return out


def noise_robustness(model, data, levels=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0), repeats=10):
    """Re-score the test set with gaussian noise added to the sensor readings.

    On clean synthetic data every model scores ~100%, which tells you very little.
    Real milk sensors drift and misread, so this sweep is the more informative
    number: it shows which learners degrade gracefully and which fall apart.

    Noise is expressed in units of each feature's training standard deviation, so
    sd=1.0 means "as much jitter as the natural spread of that reading".
    """
    rng = np.random.default_rng(RS)
    numeric = data.schema.numeric
    if not numeric:
        return {}

    stds = data.X_train[numeric].std().replace(0, 1.0)
    voter = model.named_steps["ensemble"]
    names = [n for n, _ in voter.estimators] + ["ensemble"]
    results = {n: [] for n in names}

    for sd in levels:
        acc = {n: [] for n in names}
        for _ in range(repeats if sd > 0 else 1):
            Xn = data.X_test.copy()
            if sd > 0:
                jitter = rng.normal(0, sd, size=(len(Xn), len(numeric))) * stds.values
                Xn[numeric] = Xn[numeric].values + jitter

            Xt = model.named_steps["preprocess"].transform(Xn)
            probs = []
            for name, est in zip([n for n, _ in voter.estimators], voter.estimators_):
                p = est.predict_proba(Xt)[:, 1]
                probs.append(p)
                acc[name].append(accuracy_score(data.y_test, (p >= 0.5).astype(int)))
            w = voter.weights or [1] * len(probs)
            ens = np.average(probs, axis=0, weights=w)
            acc["ensemble"].append(accuracy_score(data.y_test, (ens >= 0.5).astype(int)))

        for n in names:
            results[n].append(round(float(np.mean(acc[n])), 4))

    return {"noise_levels": list(levels), "accuracy_by_model": results}


def tune_weights(data, verbose=True):
    """Small search over per-model voting weights using cross-validated accuracy.

    Kept deliberately coarse: it tries a handful of sensible weightings rather
    than a full grid, because with 800 near-separable rows an exhaustive search
    just overfits the validation folds.
    """
    candidates = {
        "equal (textbook soft voting)": [1, 1, 1, 1, 1],
        "downweight naive bayes": [2, 2, 2, 2, 1],
        "exclude naive bayes": [1, 1, 1, 1, 0],
        "favour svm and knn": [2, 1, 1, 3, 1],
        "favour margin models": [1, 2, 2, 3, 0],
    }
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=RS)
    best_name, best_w, best_score = None, None, -1.0

    if verbose:
        print("\n  weight search (5-fold CV accuracy):")
    for name, w in candidates.items():
        m = build_model(data.schema, weights=w)
        s = cross_val_score(m, data.X_all, data.y_all, cv=cv, scoring="accuracy").mean()
        if verbose:
            print(f"    {name:<30} {w}  ->  {s:.4f}")
        if s > best_score:
            best_name, best_w, best_score = name, w, s

    if verbose:
        print(f"  best: {best_name} {best_w} ({best_score:.4f})")
    return best_w, best_name, float(best_score)


def train(tune=False, noise_test=True, csv_path=None):
    config.ensure_dirs()

    print("STEP 1 - load and preprocess")
    data = prepare(csv_path)

    weights = config.VOTING_WEIGHTS
    weight_note = "equal weights (standard soft voting)"
    if tune:
        weights, weight_note, _ = tune_weights(data)

    print("\nSTEP 2 - train the ensemble")
    model = build_model(data.schema, weights=weights)
    model.fit(data.X_train, data.y_train)
    print(f"  fitted 5 base models + soft-voting ensemble ({weight_note})")

    # ---- per-model and ensemble scores on the held-out test set -------------
    Xt = model.named_steps["preprocess"].transform(data.X_test)
    voter = model.named_steps["ensemble"]

    per_model = {}
    print(f"\n  {'model':<14}{'acc':>8}{'prec':>8}{'recall':>8}{'f1':>8}{'auc':>8}")
    print("  " + "-" * 54)
    for name, est in zip([n for n, _ in voter.estimators], voter.estimators_):
        p = est.predict_proba(Xt)[:, 1]
        m = score_all(data.y_test, p)
        per_model[name] = m
        auc = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "  n/a"
        print(f"  {name:<14}{m['accuracy']:>8.4f}{m['precision']:>8.4f}"
              f"{m['recall']:>8.4f}{m['f1']:>8.4f}{auc:>8}")

    ens_proba = model.predict_proba(data.X_test)[:, 1]
    ens = score_all(data.y_test, ens_proba)
    print("  " + "-" * 54)
    auc = f"{ens['roc_auc']:.4f}" if ens["roc_auc"] is not None else "  n/a"
    print(f"  {'ENSEMBLE':<14}{ens['accuracy']:>8.4f}{ens['precision']:>8.4f}"
          f"{ens['recall']:>8.4f}{ens['f1']:>8.4f}{auc:>8}")

    cm = confusion_matrix(data.y_test, (ens_proba >= config.DECISION_THRESHOLD).astype(int))
    print(f"\n  confusion matrix (rows=actual, cols=predicted):\n{cm}")
    print("\n" + classification_report(
        data.y_test,
        (ens_proba >= config.DECISION_THRESHOLD).astype(int),
        target_names=["healthy", "mastitic"],
        zero_division=0,
    ))

    # ---- cross-validation --------------------------------------------------
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=RS)
    cv_scores = cross_val_score(
        build_model(data.schema, weights=weights), data.X_all, data.y_all,
        cv=cv, scoring="accuracy",
    )
    print(f"  {config.CV_FOLDS}-fold CV accuracy: {cv_scores.mean():.4f} "
          f"+/- {cv_scores.std():.4f}   folds={np.round(cv_scores, 4).tolist()}")

    # ---- honesty check on separability -------------------------------------
    if ens["accuracy"] > 0.99 and cv_scores.std() < 0.01:
        print(
            "\n  NOTE: near-perfect scores mean this dataset is almost linearly\n"
            "  separable, not that the model would perform this well on real farm\n"
            "  data. Quote the noise-robustness numbers below instead."
        )

    noise = {}
    if noise_test:
        print("\n  noise robustness (accuracy vs sensor jitter, in training std units)")
        noise = noise_robustness(model, data)
        if noise:
            hdr = "".join(f"{sd:>9.1f}" for sd in noise["noise_levels"])
            print(f"    {'model':<12}{hdr}")
            for name, row in noise["accuracy_by_model"].items():
                marker = " <-- combined" if name == "ensemble" else ""
                print(f"    {name:<12}" + "".join(f"{v:>9.4f}" for v in row) + marker)

    # ---- persist -----------------------------------------------------------
    import joblib

    joblib.dump(
        {
            "pipeline": model,
            "schema": {
                "numeric": data.schema.numeric,
                "binary": data.schema.binary,
                "categorical": data.schema.categorical,
            },
            "weights": weights,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        config.MODEL_PATH,
    )

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(data.y_all)),
        "n_train": int(len(data.y_train)),
        "n_test": int(len(data.y_test)),
        "class_balance": {
            "healthy": int((data.y_all == 0).sum()),
            "mastitic": int((data.y_all == 1).sum()),
        },
        "features": {
            "numeric": data.schema.numeric,
            "binary": data.schema.binary,
            "categorical": data.schema.categorical,
            "dropped": config.DROP_FEATURES,
        },
        "voting": {"mode": "soft", "weights": weights, "note": weight_note},
        "per_model": per_model,
        "ensemble": ens,
        "confusion_matrix": cm.tolist(),
        "cross_validation": {
            "folds": config.CV_FOLDS,
            "scores": [round(float(s), 4) for s in cv_scores],
            "mean": float(cv_scores.mean()),
            "std": float(cv_scores.std()),
        },
        "noise_robustness": noise,
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\n  saved model   -> {config.MODEL_PATH}")
    print(f"  saved metrics -> {config.METRICS_PATH}")
    return model, metrics


def main():
    ap = argparse.ArgumentParser(description="Train the mastitis soft-voting ensemble.")
    ap.add_argument("--tune-weights", action="store_true",
                    help="search voting weights with cross-validation")
    ap.add_argument("--no-noise-test", action="store_true",
                    help="skip the noise robustness sweep (faster)")
    ap.add_argument("--csv", default=None, help="path to an alternative dataset CSV")
    a = ap.parse_args()
    train(tune=a.tune_weights, noise_test=not a.no_noise_test, csv_path=a.csv)


if __name__ == "__main__":
    main()
