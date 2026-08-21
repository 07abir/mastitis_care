"""Step 1 - load and preprocess the dataset.

Reads the CSV with pandas, one-hot encodes categoricals, standard-scales the
numeric sensor features, and produces a stratified train/test split.

The scaler is wrapped in a scikit-learn ColumnTransformer that is later fitted
*inside* a Pipeline. That matters: it means test data is transformed using the
training set's mean and variance, which is what prevents the subtle data leak
you get from calling StandardScaler().fit_transform() on the whole frame first.

Run standalone to inspect what the preprocessing produces:
    python -m src.preprocess
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


@dataclass
class Schema:
    numeric: list = field(default_factory=list)
    binary: list = field(default_factory=list)
    categorical: list = field(default_factory=list)

    @property
    def all_features(self):
        return self.numeric + self.binary + self.categorical


@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    ids_train: pd.Series
    ids_test: pd.Series
    X_all: pd.DataFrame
    y_all: pd.Series
    ids_all: pd.Series
    schema: Schema


def load_raw(csv_path=None) -> pd.DataFrame:
    """Read the CSV and fail loudly if the expected columns are missing."""
    path = csv_path or config.DATA_CSV
    df = pd.read_csv(path)

    missing = [c for c in (config.ID_COLUMN, config.TARGET_COLUMN) if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required column(s) {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    n_before = len(df)
    df = df.drop_duplicates()
    if len(df) < n_before:
        print(f"  dropped {n_before - len(df)} exact duplicate rows")

    # Median-impute any numeric gaps rather than silently dropping cows.
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                fill = df[col].median()
            else:
                fill = df[col].mode().iloc[0]
            print(f"  filled {int(df[col].isna().sum())} missing value(s) in {col} -> {fill}")
            df[col] = df[col].fillna(fill)

    return df


def resolve_schema(df: pd.DataFrame) -> Schema:
    """Work out which columns are numeric / binary / categorical.

    Configured lists win; anything else in the frame is classified by dtype.
    This is what lets farm_id and season slot in later without a code change.
    """
    reserved = {config.ID_COLUMN, config.TARGET_COLUMN, *config.DROP_FEATURES}

    numeric = [c for c in config.NUMERIC_FEATURES if c in df.columns]
    binary = [c for c in config.BINARY_FEATURES if c in df.columns]
    categorical = [c for c in config.CATEGORICAL_FEATURES if c in df.columns]

    known = set(numeric) | set(binary) | set(categorical) | reserved

    for col in df.columns:
        if col in known:
            continue
        if col in config.AUTO_CATEGORICAL_CANDIDATES or not pd.api.types.is_numeric_dtype(df[col]):
            categorical.append(col)
            print(f"  auto-detected categorical column: {col}")
        elif set(df[col].dropna().unique()) <= {0, 1}:
            binary.append(col)
            print(f"  auto-detected binary column: {col}")
        else:
            numeric.append(col)
            print(f"  auto-detected numeric column: {col}")

    if not numeric and not binary and not categorical:
        raise ValueError("No usable feature columns were found.")
    return Schema(numeric=numeric, binary=binary, categorical=categorical)


def build_preprocessor(schema: Schema) -> ColumnTransformer:
    """StandardScaler on numerics, OneHotEncoder on categoricals, binaries as-is."""
    transformers = []
    if schema.numeric:
        transformers.append(("num", StandardScaler(), schema.numeric))
    if schema.categorical:
        # handle_unknown='ignore' keeps inference from crashing on a farm_id or
        # season value that never appeared during training.
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # scikit-learn < 1.2 spelled it differently
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", ohe, schema.categorical))
    if schema.binary:
        transformers.append(("bin", "passthrough", schema.binary))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare(csv_path=None, verbose: bool = True) -> Dataset:
    """Load, validate, and split. Returns everything downstream steps need."""
    df = load_raw(csv_path)
    schema = resolve_schema(df)

    ids = df[config.ID_COLUMN].astype(str)
    y = df[config.TARGET_COLUMN].astype(int)
    X = df[schema.all_features]

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X,
        y,
        ids,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=stratify,
    )

    if verbose:
        print(f"  rows            : {len(df)}")
        print(f"  numeric         : {schema.numeric}")
        print(f"  binary          : {schema.binary}")
        print(f"  categorical     : {schema.categorical or '(none in this CSV)'}")
        if config.DROP_FEATURES:
            print(f"  dropped         : {config.DROP_FEATURES} (no predictive signal)")
        healthy, mastitic = int((y == 0).sum()), int((y == 1).sum())
        print(f"  class balance   : healthy={healthy}  mastitic={mastitic} ({y.mean():.1%} positive)")
        print(f"  train / test    : {len(X_train)} / {len(X_test)} (stratified)")

    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        ids_train=ids_train,
        ids_test=ids_test,
        X_all=X,
        y_all=y,
        ids_all=ids,
        schema=schema,
    )


if __name__ == "__main__":
    print("STEP 1 - load and preprocess")
    data = prepare()
    pre = build_preprocessor(data.schema).fit(data.X_train)
    out = pre.transform(data.X_train)
    print(f"\n  transformed training matrix: {out.shape}")
    print(f"  scaled numeric means (~0)  : {out[:, :len(data.schema.numeric)].mean(axis=0).round(3)}")
    print(f"  scaled numeric stds  (~1)  : {out[:, :len(data.schema.numeric)].std(axis=0).round(3)}")
