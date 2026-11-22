import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

seed_value = 42
np.random.seed(seed_value)

os.makedirs("model", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Load bank telemarketing CSV where "unknown" stands for missing entries
bank_records = pd.read_csv(
    "data/bank-additional-full.csv", sep=";", na_values=["unknown"]
)
print("Initial shape:", bank_records.shape)

empty_per_column = bank_records.isnull().sum()
print("Missing values:\n", empty_per_column[empty_per_column > 0])

# Drop duplicate client contact rows
rows_before_dedup = len(bank_records)
bank_records = bank_records.drop_duplicates()
rows_after_dedup = len(bank_records)
print(
    f"\nDuplicate rows removed: {rows_before_dedup - rows_after_dedup} "
    f"(Remaining rows: {rows_after_dedup})"
)

# Split predictors and binary subscription label ("yes" -> 1, "no" -> 0)
client_predictors = bank_records.drop(columns=["y"])
subscribed_label = bank_records["y"].map({"no": 0, "yes": 1})
print("\nClass distribution:\n", subscribed_label.value_counts(normalize=True))

# Stratified 80:20 split so class proportions stay balanced
(
    train_predictors,
    holdout_predictors,
    train_subscribed,
    holdout_subscribed,
) = train_test_split(
    client_predictors,
    subscribed_label,
    test_size=0.20,
    stratify=subscribed_label,
    random_state=seed_value,
)

# Save holdout test slice with target column for evaluation
holdout_snapshot = holdout_predictors.copy()
holdout_snapshot["target"] = holdout_subscribed
holdout_snapshot.to_csv("test_data.csv", index=False)

# Identify numeric vs. text/category columns from training predictors only
continuous_fields = train_predictors.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()
discrete_fields = train_predictors.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

continuous_pipeline = Pipeline([
    ("median_fill", SimpleImputer(strategy="median")),
    ("zscore_scale", StandardScaler()),
])

discrete_pipeline = Pipeline([
    ("mode_fill", SimpleImputer(strategy="most_frequent")),
    ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# Combine preprocessing steps inside ColumnTransformer (fitted per fold/run on train only)
bank_transformer = ColumnTransformer([
    ("continuous", continuous_pipeline, continuous_fields),
    ("discrete", discrete_pipeline, discrete_fields),
])

# Five required classification algorithms
alg_suite = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed_value),
    "Decision Tree": DecisionTreeClassifier(random_state=seed_value),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=100, random_state=seed_value
    ),
}

score_records = []

for alg_title, alg_instance in alg_suite.items():
    training_flow = Pipeline([
        ("transform", bank_transformer),
        ("estimator", alg_instance),
    ])
    training_flow.fit(train_predictors, train_subscribed)

    predicted_outcomes = training_flow.predict(holdout_predictors)
    subscription_odds = training_flow.predict_proba(holdout_predictors)[:, 1]

    score_records.append({
        "ML Model Name": alg_title,
        "Accuracy": round(accuracy_score(holdout_subscribed, predicted_outcomes), 4),
        "AUC": round(roc_auc_score(holdout_subscribed, subscription_odds), 4),
        "Precision": round(precision_score(holdout_subscribed, predicted_outcomes), 4),
        "Recall": round(recall_score(holdout_subscribed, predicted_outcomes), 4),
        "F1": round(f1_score(holdout_subscribed, predicted_outcomes), 4),
        "MCC": round(matthews_corrcoef(holdout_subscribed, predicted_outcomes), 4),
    })

    artifact_file = (
        alg_title.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
    )
    joblib.dump(training_flow, os.path.join("model", artifact_file))

score_board = pd.DataFrame(score_records)
print("\n=== Evaluation Metrics Comparison ===")
print(score_board.to_string(index=False))

score_board.set_index("ML Model Name").plot(kind="bar", figsize=(11, 5))
plt.title("Classification Model Comparison Across 6 Metrics")
plt.ylabel("Score")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("plots/model_comparison.png", dpi=150)
plt.close()
print("\nSaved chart to plots/model_comparison.png")
