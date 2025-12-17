import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    chi2,
    mutual_info_classif,
    RFE
)

from sklearn.ensemble import RandomForestClassifier

# =========================================================
# 1. Data
# =========================================================
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================================================
# 2. Helper
# =========================================================
def evaluate_model(Xtr, Xte, ytr, yte):
    model = LogisticRegression(max_iter=500)
    model.fit(Xtr, ytr)
    y_pred = model.predict(Xte)
    return (
        accuracy_score(yte, y_pred),
        f1_score(yte, y_pred)
    )

results = []

# =========================================================
# 3. No Feature Selection
# =========================================================
start = time.time()
acc, f1 = evaluate_model(X_train, X_test, y_train, y_test)
results.append(["No FS", acc, f1, X_train.shape[1], time.time() - start])

# =========================================================
# 4. Filter Methods
# =========================================================

# Variance Threshold
start = time.time()
vt = VarianceThreshold(threshold=0.01)
Xtr_vt = vt.fit_transform(X_train)
Xte_vt = vt.transform(X_test)
acc, f1 = evaluate_model(Xtr_vt, Xte_vt, y_train, y_test)
results.append(["Variance Threshold", acc, f1, Xtr_vt.shape[1], time.time() - start])

# Correlation (Pearson)
start = time.time()
corr = np.abs(pd.DataFrame(X_train).corrwith(pd.Series(y_train)))
idx = corr.sort_values(ascending=False).head(10).index
Xtr_corr = X_train[:, idx]
Xte_corr = X_test[:, idx]
acc, f1 = evaluate_model(Xtr_corr, Xte_corr, y_train, y_test)
results.append(["Correlation", acc, f1, 10, time.time() - start])

# ANOVA
start = time.time()
anova = SelectKBest(score_func=f_classif, k=10)
Xtr_anova = anova.fit_transform(X_train, y_train)
Xte_anova = anova.transform(X_test)
acc, f1 = evaluate_model(Xtr_anova, Xte_anova, y_train, y_test)
results.append(["ANOVA F-test", acc, f1, 10, time.time() - start])

# Chi-Squared
start = time.time()
chi = SelectKBest(score_func=chi2, k=10)
Xtr_chi = chi.fit_transform(X_train, y_train)
Xte_chi = chi.transform(X_test)
acc, f1 = evaluate_model(Xtr_chi, Xte_chi, y_train, y_test)
results.append(["Chi-Squared", acc, f1, 10, time.time() - start])

# Mutual Information
start = time.time()
mi = SelectKBest(score_func=mutual_info_classif, k=10)
Xtr_mi = mi.fit_transform(X_train, y_train)
Xte_mi = mi.transform(X_test)
acc, f1 = evaluate_model(Xtr_mi, Xte_mi, y_train, y_test)
results.append(["Mutual Information", acc, f1, 10, time.time() - start])

# =========================================================
# 5. Wrapper
# =========================================================
start = time.time()
rfe = RFE(
    estimator=LogisticRegression(max_iter=500),
    n_features_to_select=10
)
Xtr_rfe = rfe.fit_transform(X_train, y_train)
Xte_rfe = rfe.transform(X_test)
acc, f1 = evaluate_model(Xtr_rfe, Xte_rfe, y_train, y_test)
results.append(["RFE", acc, f1, 10, time.time() - start])

# =========================================================
# 6. Embedded
# =========================================================

# LASSO
start = time.time()
lasso = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    max_iter=500
)
lasso.fit(X_train, y_train)
mask = np.abs(lasso.coef_[0]) > 1e-4
Xtr_lasso = X_train[:, mask]
Xte_lasso = X_test[:, mask]
acc, f1 = evaluate_model(Xtr_lasso, Xte_lasso, y_train, y_test)
results.append(["LASSO", acc, f1, Xtr_lasso.shape[1], time.time() - start])

# Random Forest
start = time.time()
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
idx = np.argsort(importances)[-10:]
Xtr_rf = X_train[:, idx]
Xte_rf = X_test[:, idx]
acc, f1 = evaluate_model(Xtr_rf, Xte_rf, y_train, y_test)
results.append(["Random Forest", acc, f1, 10, time.time() - start])

# =========================================================
# 7. Results table
# =========================================================
df = pd.DataFrame(
    results,
    columns=["Method", "Accuracy", "F1", "Num Features", "Time (s)"]
)
print(df)

# =========================================================
# 8. Plots
# =========================================================
import os
os.makedirs("plots", exist_ok=True)

# Accuracy (zoomed)
plt.figure()
plt.bar(df["Method"], df["Accuracy"])
plt.ylim(0.95, 1.0)
plt.xticks(rotation=45, ha="right")
plt.title("Accuracy (zoomed)")
plt.tight_layout()
plt.savefig("plots/accuracy_zoomed.png")
plt.close()

# Delta Accuracy
baseline = df.loc[df["Method"] == "No FS", "Accuracy"].values[0]
delta = df["Accuracy"] - baseline

plt.figure()
plt.bar(df["Method"], delta)
plt.axhline(0, linestyle="--")
plt.xticks(rotation=45, ha="right")
plt.title("Δ Accuracy relative to No FS")
plt.tight_layout()
plt.savefig("plots/accuracy_delta.png")
plt.close()

# Error rate
plt.figure()
plt.bar(df["Method"], 1 - df["Accuracy"])
plt.xticks(rotation=45, ha="right")
plt.title("Error rate (1 - Accuracy)")
plt.tight_layout()
plt.savefig("plots/error_rate_comparison.png")
plt.close()

# Num features
plt.figure()
plt.bar(df["Method"], df["Num Features"])
plt.xticks(rotation=45, ha="right")
plt.title("Number of selected features")
plt.tight_layout()
plt.savefig("plots/num_features_comparison.png")
plt.close()

# Time
plt.figure()
plt.bar(df["Method"], df["Time (s)"])
plt.xticks(rotation=45, ha="right")
plt.title("Execution time")
plt.tight_layout()
plt.savefig("plots/time_comparison.png")
plt.close()
