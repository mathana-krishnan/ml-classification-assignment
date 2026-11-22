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
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

df = pd.read_csv("data/bank-additional-full.csv", sep=";", na_values=["unknown"])
print("Initial shape:", df.shape)
print("Missing values:\n", df.isnull().sum()[df.isnull().sum() > 0])

df = df.drop_duplicates()
X = df.drop(columns=["y"])
y = df["y"].map({"no": 0, "yes": 1})
print("\nClass distribution:\n", y.value_counts(normalize=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

test_export = X_test.copy()
test_export["target"] = y_test
test_export.to_csv("test_data.csv", index=False)

num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ]
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    results.append({
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    })

    filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
    joblib.dump(pipe, os.path.join("model", filename))

results_df = pd.DataFrame(results)
print("\n=== Evaluation Metrics Comparison ===")
print(results_df.to_string(index=False))

ax = results_df.set_index("ML Model Name").plot(kind="bar", figsize=(11, 5))
plt.title("Classification Model Comparison Across 6 Metrics")
plt.ylabel("Score")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("plots/model_comparison.png", dpi=150)
print("\nSaved chart to plots/model_comparison.png")
