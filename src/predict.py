"""Score new content rows with the trained pipeline."""
import argparse
import pickle

import pandas as pd

from preprocessing import engineer_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.pkl")
    parser.add_argument("--input", required=True, help="CSV of raw content rows to score")
    parser.add_argument("--output", default="outputs/predictions.csv")
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        pipe = pickle.load(f)

    df = pd.read_csv(args.input)
    df = engineer_features(df)

    proba = pipe.predict_proba(df)[:, 1]
    df["decline_probability"] = proba
    df["predicted_declining"] = (proba >= 0.5).astype(int)
    df.sort_values("decline_probability", ascending=False).to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} scored rows to {args.output}")


if __name__ == "__main__":
    main()
