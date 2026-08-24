"""Streamlit demo: score a single content page for decline risk."""
import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from preprocessing import engineer_features  # noqa: E402

st.set_page_config(page_title="Content Decline Risk", page_icon="📉", layout="centered")

st.title("Content Decline Risk Estimator")
st.caption(
    "Estimates the probability that a content page is currently declining in organic search "
    "performance, based on the FlyRank content-refresh model. Decision-support only — "
    "not a guarantee of future performance."
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"

if not MODEL_PATH.exists():
    st.warning("No trained model found. Run `python src/train.py` first to generate `models/model.pkl`.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    pipe = pickle.load(f)

st.subheader("Content signals (90-day window)")

col1, col2 = st.columns(2)
with col1:
    impressions_90d = st.number_input("Impressions (90d)", min_value=0, value=1500)
    clicks_90d = st.number_input("Clicks (90d)", min_value=0, value=25)
    sessions_90d = st.number_input("Sessions (90d)", min_value=0, value=20)
    ai_sessions_90d = st.number_input("AI-referred sessions (90d)", min_value=0, value=0)
    days_with_impressions = st.slider("Days with ≥1 impression (of 90)", 0, 90, 40)
    days_with_sessions = st.slider("Days with ≥1 session (of 90)", 0, 90, 15)
    content_age_days = st.number_input("Content age (days)", min_value=90, value=200)
    days_since_last_update = st.number_input("Days since last update", min_value=0, value=60)

with col2:
    word_count = st.number_input("Word count", min_value=0, value=1800)
    char_count = st.number_input("Character count", min_value=0, value=11000)
    avg_position = st.number_input("Average GSC position", min_value=0.0, value=18.5)
    ctr = st.number_input("CTR (%)", min_value=0.0, value=1.2)
    engagement_rate = st.number_input("Engagement rate (%)", min_value=0.0, value=45.0)
    scroll_rate = st.number_input("Scroll rate (%)", min_value=0.0, value=60.0)
    content_type = st.selectbox("Content type", ["keyword article", "feedly article", "comparison article"])
    main_intent = st.selectbox("Main intent", ["informational", "transactional", "commercial", "navigational", "unknown"])

age_tier = st.selectbox("Age tier", ["31-90", "91-180", "181-365", "365+"])
freshness_tier = st.selectbox("Freshness tier", ["0-30", "31-90", "91-180", "181+"])
word_count_tier = st.selectbox("Word count tier", ["<1000", "1000-2000", "2000-3500", "3500+"])
impression_tier = st.selectbox("Impression tier", ["none", "low", "moderate", "good", "excellent"])
position_tier = st.selectbox("Position tier", ["top_3", "page_1", "striking", "page_3_5", "deep"])
competition_level = st.selectbox("Competition level", ["LOW", "MEDIUM", "HIGH", "unknown"])

if st.button("Predict decline risk", type="primary"):
    row = pd.DataFrame([{
        "impressions_90d": impressions_90d, "clicks_90d": clicks_90d, "sessions_90d": sessions_90d,
        "ai_sessions_90d": ai_sessions_90d, "days_with_impressions": days_with_impressions,
        "days_with_sessions": days_with_sessions, "content_age_days": content_age_days,
        "days_since_last_update": days_since_last_update, "word_count": word_count,
        "char_count": char_count, "avg_position": avg_position, "ctr": ctr,
        "engagement_rate": engagement_rate, "scroll_rate": scroll_rate,
        "search_volume": 0, "competition": 0, "cpc": 0, "ai_traffic_pct": 0,
        "content_type": content_type, "main_intent": main_intent, "age_tier": age_tier,
        "freshness_tier": freshness_tier, "word_count_tier": word_count_tier,
        "impression_tier": impression_tier, "position_tier": position_tier,
        "competition_level": competition_level,
    }])
    row = engineer_features(row)
    proba = pipe.predict_proba(row)[0, 1]

    st.metric("Decline probability", f"{proba*100:.1f}%")
    if proba >= 0.5:
        st.error("Flagged as likely declining — prioritize for review.")
    else:
        st.success("Not currently flagged as declining.")
    st.caption("Model: Logistic Regression · Test AUC ≈ 0.62 on the starter slice. Use as a ranking signal, not a verdict.")
