"""Train the content-decline classifier and save the fitted pipeline."""
import argparse
import pickle
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    engineer_features,
    get_feature_matrix,
    load_and_clean,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/content_refresh_anonymized.csv")
    parser.add_argument("--output", default="models/model.pkl")
    args = parser.parse_args()

    df = load_and_clean(args.input)
    df = engineer_features(df)
    X, y, groups = get_feature_matrix(df)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, _ = next(gss.split(X, y, groups))
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), MODEL_NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), MODEL_CATEGORICAL_FEATURES),
    ])
    pipe = Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(X_train, y_train)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(pipe, f)
    print(f"Trained on {len(X_train):,} rows. Model saved to {args.output}")


if __name__ == "__main__":
    main()
