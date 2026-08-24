"""
Content Decline Prediction — Full Analysis Pipeline
Dataset: FlyRank content_refresh_anonymized.csv (30,000 rows x 44 cols)
Target: is_declining_label (trend_direction == 'down')
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)

sns.set_style("darkgrid")
plt.rcParams.update({
    "figure.facecolor": "#0a0e17",
    "axes.facecolor": "#0a0e17",
    "savefig.facecolor": "#0a0e17",
    "axes.edgecolor": "#2a3244",
    "axes.labelcolor": "#e5e9f0",
    "xtick.color": "#8b93a7",
    "ytick.color": "#8b93a7",
    "text.color": "#e5e9f0",
    "grid.color": "#1a2030",
    "font.size": 11,
})

CYAN = "#22d3ee"
BLUE = "#3b82f6"
RED = "#f87171"
GREEN = "#34d399"

OUT = "/home/claude/project/outputs"
CHARTS = f"{OUT}/charts"

results = {}

# ============================================================
# 1. LOAD + INSPECT
# ============================================================
df_raw = pd.read_csv("/home/claude/project/data/raw/content_refresh_anonymized.csv")
results["dataset"] = {
    "name": "FlyRank Content Refresh (anonymized starter slice)",
    "source_repo": "flyrank-bih/flyrank-ml-internship-starter",
    "n_rows_raw": int(len(df_raw)),
    "n_cols_raw": int(df_raw.shape[1]),
    "n_clients": int(df_raw["client_id"].nunique()),
    "columns": list(df_raw.columns),
}

# Missingness
missing = df_raw.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
results["missing_values"] = {k: int(v) for k, v in missing.items()}

# Dtypes
dtypes = df_raw.dtypes.astype(str).to_dict()
results["dtypes"] = dtypes

# categorical vs numeric
cat_cols = df_raw.select_dtypes(include="object").columns.tolist()
num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
results["categorical_features"] = cat_cols
results["numerical_features"] = num_cols

print("RAW SHAPE:", df_raw.shape)
print("MISSING (top):\n", missing.head(10))
print("TREND DIRECTION VALUE COUNTS:\n", df_raw["trend_direction"].value_counts())

# ============================================================
# 2. CLEAN + FEATURE ENGINEERING (mirrors the real prep pipeline)
# ============================================================
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

df = df_raw.copy()
for c in NUMERIC_FILL_ZERO:
    df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
for c in CATEGORICAL_FILL:
    df[c] = df[c].fillna("unknown").astype(str).replace({"": "unknown", "nan": "unknown"})

initial = len(df)
# quality filter: rows must have real impression history and be at least 90 days old
df = df[(df["impressions_90d"] > 0) & (df["content_age_days"] >= 90)].copy()
df = df.drop_duplicates(subset=["content_id"]).reset_index(drop=True)
n_dupes_and_filtered = initial - len(df)

# TARGET: is_declining_label
df["is_declining_label"] = df["trend_direction"].str.lower().eq("down").astype(int)

# leakage-safe engineered features
df["log_impressions_90d"] = np.log1p(df["impressions_90d"])
df["log_clicks_90d"] = np.log1p(df["clicks_90d"])
df["log_sessions_90d"] = np.log1p(df["sessions_90d"])
df["log_ai_sessions_90d"] = np.log1p(df["ai_sessions_90d"])
df["has_clicks"] = (df["clicks_90d"] > 0).astype(int)
df["has_ai_sessions"] = (df["ai_sessions_90d"] > 0).astype(int)
df["measurable_opportunity"] = ((df["impressions_90d"] >= 100) & (df["sessions_90d"] > 0)).astype(int)

results["cleaning"] = {
    "rows_before": int(initial),
    "rows_removed_low_quality_or_dupe": int(n_dupes_and_filtered),
    "rows_after": int(len(df)),
    "target_definition": "is_declining_label = 1 if trend_direction == 'down' else 0",
    "declining_rows": int(df["is_declining_label"].sum()),
    "declining_rate_pct": round(float(df["is_declining_label"].mean()) * 100, 1),
}
print(results["cleaning"])

# ============================================================
# 3. EDA / VISUALIZATIONS
# ============================================================

# 3a. Target distribution
fig, ax = plt.subplots(figsize=(6, 5))
counts = df["trend_direction"].value_counts()
colors = [CYAN if x == "down" else "#3a4358" for x in counts.index]
ax.bar(counts.index, counts.values, color=colors, edgecolor="none")
ax.set_title("Trend Direction Distribution", color="#e5e9f0", fontsize=13)
ax.set_ylabel("Content Pages")
plt.tight_layout()
plt.savefig(f"{CHARTS}/trend_direction_distribution.png", dpi=150)
plt.close()

# 3b. Correlation heatmap of numeric model features
feature_num = [
    "search_volume", "competition", "cpc", "word_count", "char_count",
    "log_impressions_90d", "log_clicks_90d", "log_sessions_90d", "log_ai_sessions_90d",
    "days_with_impressions", "days_with_sessions", "content_age_days",
    "days_since_last_update", "ctr", "avg_position", "engagement_rate",
    "scroll_rate", "ai_traffic_pct",
]
corr = df[feature_num + ["is_declining_label"]].corr()
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, cmap="mako", center=0, annot=False, ax=ax, cbar_kws={"label": "correlation"})
ax.set_title("Feature Correlation Matrix", color="#e5e9f0", fontsize=13)
plt.tight_layout()
plt.savefig(f"{CHARTS}/correlation_heatmap.png", dpi=150)
plt.close()

# 3c. Declining rate by content_type
rate_by_type = df.groupby("content_type")["is_declining_label"].mean().sort_values(ascending=False) * 100
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(rate_by_type.index, rate_by_type.values, color=CYAN)
ax.set_xlabel("Declining Rate (%)")
ax.set_title("Decline Rate by Content Type", color="#e5e9f0", fontsize=13)
plt.tight_layout()
plt.savefig(f"{CHARTS}/decline_rate_by_content_type.png", dpi=150)
plt.close()
results["decline_rate_by_content_type"] = rate_by_type.round(1).to_dict()

# 3d. Declining rate by freshness tier
rate_by_fresh = df.groupby("freshness_tier")["is_declining_label"].mean().sort_values(ascending=False) * 100
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(rate_by_fresh.index, rate_by_fresh.values, color=BLUE)
ax.set_xlabel("Declining Rate (%)")
ax.set_title("Decline Rate by Freshness Tier (days since update)", color="#e5e9f0", fontsize=13)
plt.tight_layout()
plt.savefig(f"{CHARTS}/decline_rate_by_freshness.png", dpi=150)
plt.close()
results["decline_rate_by_freshness_tier"] = rate_by_fresh.round(1).to_dict()

# ============================================================
# 4. TRAIN / TEST SPLIT (grouped by client — prevents leakage)
# ============================================================
MODEL_NUMERIC_FEATURES = feature_num
MODEL_CATEGORICAL_FEATURES = [
    "competition_level", "content_type", "main_intent",
    "age_tier", "freshness_tier", "word_count_tier",
    "impression_tier", "position_tier",
]

X = df[MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES]
y = df["is_declining_label"]
groups = df["client_id"]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

results["split"] = {
    "method": "GroupShuffleSplit by client_id (client-holdout, prevents leakage across train/test)",
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "train_clients": int(df.iloc[train_idx]["client_id"].nunique()),
    "test_clients": int(df.iloc[test_idx]["client_id"].nunique()),
}
print(results["split"])

preprocess = ColumnTransformer([
    ("num", StandardScaler(), MODEL_NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), MODEL_CATEGORICAL_FEATURES),
])

# ============================================================
# 5. TRAIN MODELS
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=30, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1),
}

model_results = {}
best_name, best_f1, best_pipe, best_pred, best_proba = None, -1, None, None, None

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)

    model_results[name] = {
        "accuracy": round(acc * 100, 1),
        "precision": round(prec * 100, 1),
        "recall": round(rec * 100, 1),
        "f1": round(f1 * 100, 1),
        "roc_auc": round(auc_score, 3),
    }
    print(name, model_results[name])

    if f1 > best_f1:
        best_f1 = f1
        best_name = name
        best_pipe = pipe
        best_pred = y_pred
        best_proba = y_proba

results["model_results"] = model_results
results["best_model"] = best_name
print("BEST MODEL:", best_name)

# ============================================================
# 6. EVALUATION CHARTS (for best model)
# ============================================================

# 6a. Confusion matrix
cm = confusion_matrix(y_test, best_pred)
fig, ax = plt.subplots(figsize=(6, 5.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="mako", cbar=False, ax=ax,
            xticklabels=["Stable/Up", "Declining"], yticklabels=["Stable/Up", "Declining"],
            annot_kws={"size": 16, "color": "white"})
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best_name}", color="#e5e9f0", fontsize=13)
plt.tight_layout()
plt.savefig(f"{CHARTS}/confusion_matrix.png", dpi=150)
plt.close()
results["confusion_matrix"] = cm.tolist()

# 6b. ROC curve — all 3 models
fig, ax = plt.subplots(figsize=(6.5, 5.5))
roc_colors = {"Logistic Regression": "#8b93a7", "Decision Tree": BLUE, "Random Forest": CYAN}
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})", color=roc_colors[name], linewidth=2)
ax.plot([0, 1], [0, 1], linestyle="--", color="#3a4358")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Model Comparison", color="#e5e9f0", fontsize=13)
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{CHARTS}/roc_curve.png", dpi=150)
plt.close()

# 6c. Feature importance (Random Forest)
rf_pipe = Pipeline([("prep", preprocess), ("clf", RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1))])
rf_pipe.fit(X_train, y_train)
feature_names = (
    MODEL_NUMERIC_FEATURES +
    list(rf_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(MODEL_CATEGORICAL_FEATURES))
)
importances = rf_pipe.named_steps["clf"].feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color=CYAN)
ax.set_xlabel("Importance")
ax.set_title("Top 15 Feature Importances — Random Forest", color="#e5e9f0", fontsize=13)
plt.tight_layout()
plt.savefig(f"{CHARTS}/feature_importance.png", dpi=150)
plt.close()
results["top_features"] = imp_df.round(4).to_dict(orient="records")

# 6d. Prediction distribution
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(best_proba[y_test == 0], bins=30, alpha=0.7, label="Actually Stable/Up", color=BLUE)
ax.hist(best_proba[y_test == 1], bins=30, alpha=0.7, label="Actually Declining", color=CYAN)
ax.set_xlabel("Predicted Probability of Decline")
ax.set_ylabel("Count")
ax.set_title(f"Prediction Distribution — {best_name}", color="#e5e9f0", fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig(f"{CHARTS}/prediction_distribution.png", dpi=150)
plt.close()

# classification report
report = classification_report(y_test, best_pred, target_names=["Stable/Up", "Declining"])
results["classification_report"] = report
print(report)

# ============================================================
# SAVE ALL RESULTS
# ============================================================
with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n\nDONE. Results saved to outputs/results.json")
print("Charts saved to outputs/charts/")
