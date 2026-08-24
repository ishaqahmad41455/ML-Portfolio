"""Cleaning + feature engineering for the content-decline model."""
import numpy as np
import pandas as pd

NUMERIC_FILL_ZERO = [
    "search_volume", "competition", "cpc", "word_count", "char_count",
    "impressions_90d", "clicks_90d", "pageviews_90d", "sessions_90d", "users_90d",
    "engaged_sessions_90d", "ai_sessions_90d", "scroll_events_90d",
    "days_with_impressions", "days_with_sessions",
    "impressions_last_30d", "clicks_last_30d", "sessions_last_30d",
    "impressions_prev_30d", "clicks_prev_30d", "sessions_prev_30d",
    "content_age_days", "age_tier_order", "days_since_last_update",
    "ctr", "avg_position", "engagement_rate", "scroll_rate", "ai_traffic_pct", "trend_pct",
]

CATEGORICAL_FILL = [
    "competition_level", "content_type", "main_intent", "provider_used", "model_used",
    "age_tier", "freshness_tier", "word_count_tier", "char_count_tier",
    "impression_tier", "position_tier", "trend_direction",
]

MODEL_NUMERIC_FEATURES = [
    "search_volume", "competition", "cpc", "word_count", "char_count",
    "log_impressions_90d", "log_clicks_90d", "log_sessions_90d", "log_ai_sessions_90d",
    "days_with_impressions", "days_with_sessions", "content_age_days",
    "days_since_last_update", "ctr", "avg_position", "engagement_rate",
    "scroll_rate", "ai_traffic_pct",
]

MODEL_CATEGORICAL_FEATURES = [
    "competition_level", "content_type", "main_intent",
    "age_tier", "freshness_tier", "word_count_tier",
    "impression_tier", "position_tier",
]


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load the raw anonymized export and apply the cleaning + target definition."""
    df = pd.read_csv(csv_path)

    for c in NUMERIC_FILL_ZERO:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    for c in CATEGORICAL_FILL:
        df[c] = df[c].fillna("unknown").astype(str).replace({"": "unknown", "nan": "unknown"})

    df = df[(df["impressions_90d"] > 0) & (df["content_age_days"] >= 90)].copy()
    df = df.drop_duplicates(subset=["content_id"]).reset_index(drop=True)

    df["is_declining_label"] = df["trend_direction"].str.lower().eq("down").astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe engineered features. `trend_direction`/`trend_pct` stay label-only."""
    df = df.copy()
    df["log_impressions_90d"] = np.log1p(df["impressions_90d"])
    df["log_clicks_90d"] = np.log1p(df["clicks_90d"])
    df["log_sessions_90d"] = np.log1p(df["sessions_90d"])
    df["log_ai_sessions_90d"] = np.log1p(df["ai_sessions_90d"])
    df["has_clicks"] = (df["clicks_90d"] > 0).astype(int)
    df["has_ai_sessions"] = (df["ai_sessions_90d"] > 0).astype(int)
    df["measurable_opportunity"] = ((df["impressions_90d"] >= 100) & (df["sessions_90d"] > 0)).astype(int)
    return df


def get_feature_matrix(df: pd.DataFrame):
    X = df[MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES]
    y = df["is_declining_label"]
    groups = df["client_id"]
    return X, y, groups
