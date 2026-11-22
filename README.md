# Bank Term Deposit Subscription Classification

## 1. Problem Statement
Predict whether a bank client will subscribe to a term deposit (`y`: `yes` or `no`) using client profile attributes, campaign contact history, and macroeconomic indicators from `bank-additional-full.csv`.

## 2. Dataset Description
* **Source:** UCI Machine Learning Repository — Bank Marketing Dataset (ID: 222), `bank-additional-full.csv` (Sérgio Moro, P. Cortez, and P. Rita, 2014).
* **Size & Target:** `41,188` initial instances and `21` columns (20 input features + target `y` mapped to `0` for `no` and `1` for `yes`).
* **Missing Values:** `"unknown"` entries treated as missing (`NaN`): `job` (330), `marital` (80), `education` (1731), `default` (8597), `housing` (990), and `loan` (990).
* **Class Distribution:** Imbalanced target with `0` (`no`) at `0.887337` (~88.73%) and `1` (`yes`) at `0.112663` (~11.27%).
* **Preprocessing & Execution:**
  * Remove exact duplicate rows, then split `80:20` using stratification (`random_state=42`) and export `test_data.csv` with `target`.
  * Fit `ColumnTransformer` strictly on the training set inside each model `Pipeline` (`SimpleImputer(strategy="median")` + `StandardScaler()` for numeric features; `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")` for categorical features).
  * Run from the repository root:
    ```bash
    pip install -r requirements.txt
    python model/train_evaluate_2026ct05056.py
    ```

## 3. GitHub Repository Link
* **Repository:** `https://github.com/mathana-krishnan/ml-classification-assignment`

## 4. Models Used & Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9097 | 0.9389 | 0.6538 | 0.4213 | 0.5125 | 0.4789 |
| Decision Tree | 0.8876 | 0.7302 | 0.5010 | 0.5269 | 0.5137 | 0.4503 |
| kNN | 0.9027 | 0.8718 | 0.5924 | 0.4386 | 0.5040 | 0.4577 |
| Naive Bayes | 0.7347 | 0.8474 | 0.2764 | 0.8373 | 0.4156 | 0.3728 |
| Random Forest (Ensemble) | 0.9148 | 0.9447 | 0.6687 | 0.4828 | 0.5607 | 0.5232 |

![Classification Model Comparison Across 6 Metrics](plots/model_comparison.png)

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Achieves strong ranking and precision (AUC `0.9389`, Precision `0.6538`, MCC `0.4789`), though conservative recall (`0.4213`) limits F1 (`0.5125`) on the minority `yes` class. |
| Decision Tree | Slightly edges out Logistic Regression on F1 (`0.5137`) with higher recall (`0.5269`), but drops sharply on AUC (`0.7302`) and precision (`0.5010`) due to unpruned tree splits. |
| kNN | Delivers solid overall accuracy (`0.9027`) and AUC (`0.8718`), while moderate recall (`0.4386`) and MCC (`0.4577`) reflect distance sensitivity across scaled numeric and one-hot features. |
| Naive Bayes | Captures the highest minority-class recall (`0.8373`) and decent AUC (`0.8474`), but low precision (`0.2764`) and accuracy (`0.7347`) produce many false positives under conditional independence. |
| Random Forest (Ensemble) | Leads across Accuracy (`0.9148`), AUC (`0.9447`), Precision (`0.6687`), F1 (`0.5607`), and MCC (`0.5232`), averaging multiple trees to handle non-linear patterns and class imbalance better than single models. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** is the overall winner for this run because it achieves the highest scores on Accuracy, AUC, Precision, F1, and MCC. |
