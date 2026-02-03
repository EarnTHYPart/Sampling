# -*- coding: utf-8 -*-
"""Sampling assignment script.

This version is runnable as a standalone Python script.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

PLOTS_DIR = Path("plots")
DATA_PATH = Path("Creditcard_data.csv")


def save_plot(name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Place Creditcard_data.csv in the project root."
        )

    df = pd.read_csv(DATA_PATH)

    sns.countplot(x="Class", data=df)
    plt.title("Original Class Distribution")
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sampling_methods = {
        "RandomOver": RandomOverSampler(),
        "SMOTE": SMOTE(),
        "ADASYN": ADASYN(),
        "RandomUnder": RandomUnderSampler(),
        "NearMiss": NearMiss(),
    }

    models_basic = {
        "Logistic": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
    }

    results = pd.DataFrame(columns=["Model", "Sampling", "Accuracy"])

    for samp_name, sampler in sampling_methods.items():
        X_res, y_res = sampler.fit_resample(X_scaled, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42
        )

        for model_name, model in models_basic.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            results.loc[len(results)] = [model_name, samp_name, acc]

    pivot_table = results.pivot(index="Model", columns="Sampling", values="Accuracy")

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, cmap="viridis")
    plt.title("Accuracy Heatmap")
    plt.show()

    results.plot(kind="bar", x="Model", y="Accuracy", figsize=(12, 6))
    plt.title("Model Accuracy Comparison")
    plt.show()

    best_sampling = results.loc[results.groupby("Model")["Accuracy"].idxmax()]

    best_model_name = best_sampling.iloc[0]["Model"]
    best_sampling_name = best_sampling.iloc[0]["Sampling"]

    best_model = models_basic[best_model_name]
    best_sampler = sampling_methods[best_sampling_name]

    X_res, y_res = best_sampler.fit_resample(X_scaled, y)
    best_model.fit(X_res, y_res)

    def predict_sample(sample: np.ndarray) -> np.ndarray:
        sample_scaled = scaler.transform([sample])
        return best_model.predict(sample_scaled)

    _ = predict_sample(X.iloc[0].to_numpy())

    plt.figure(figsize=(10, 6))

    for model in results["Model"].unique():
        subset = results[results["Model"] == model]
        plt.plot(subset["Sampling"], subset["Accuracy"], marker="o", label=model)

    plt.title("Accuracy Comparison Across Sampling Techniques")
    plt.xlabel("Sampling Technique")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    avg_sampling = results.groupby("Sampling")["Accuracy"].mean().sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    avg_sampling.plot(kind="bar")
    plt.title("Average Accuracy per Sampling Technique")
    plt.ylabel("Accuracy")
    plt.xlabel("Sampling Technique")
    plt.show()

    best_sampling = results.loc[results.groupby("Model")["Accuracy"].idxmax()]

    plt.figure(figsize=(8, 5))
    plt.bar(best_sampling["Model"], best_sampling["Accuracy"])
    plt.title("Best Sampling Technique per Model")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.show()

    print(best_sampling)

    ranking = avg_sampling.rank(ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(ranking.index, ranking.values)
    plt.title("Sampling Technique Ranking")
    plt.ylabel("Rank (Lower is Better)")
    plt.xlabel("Sampling Technique")
    plt.gca().invert_yaxis()
    plt.show()

    variance_sampling = results.groupby("Sampling")["Accuracy"].var()

    plt.figure(figsize=(8, 5))
    variance_sampling.plot(kind="bar")
    plt.title("Sampling Technique Stability (Variance)")
    plt.ylabel("Variance")
    plt.xlabel("Sampling Technique")
    plt.show()

    pivot = results.pivot(index="Model", columns="Sampling", values="Accuracy")

    pivot.plot(kind="bar", figsize=(12, 6))
    plt.title("Accuracy Comparison: Models vs Sampling")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.legend(title="Sampling")
    plt.show()

    models = {
        "Logistic": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "NaiveBayes": GaussianNB(),
        "GradientBoost": GradientBoostingClassifier(),
    }

    results_rows = []

    for samp_name, sampler in sampling_methods.items():
        X_res, y_res = sampler.fit_resample(X_scaled, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42
        )

        for model_name, model in models.items():
            start = time.time()
            model.fit(X_train, y_train)
            end = time.time()

            preds = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
                roc = roc_auc_score(y_test, probs)
            else:
                roc = 0

            results_rows.append(
                [
                    model_name,
                    samp_name,
                    accuracy_score(y_test, preds),
                    precision_score(y_test, preds, zero_division=0),
                    recall_score(y_test, preds, zero_division=0),
                    f1_score(y_test, preds, zero_division=0),
                    roc,
                    end - start,
                ]
            )

    results_df = pd.DataFrame(
        results_rows,
        columns=["Model", "Sampling", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Time"],
    )

    pivot_acc = results_df.pivot(index="Model", columns="Sampling", values="Accuracy")

    pivot_acc.plot(kind="bar", figsize=(14, 6))
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.show()

    pivot_f1 = results_df.pivot(index="Model", columns="Sampling", values="F1")

    pivot_f1.plot(kind="bar", figsize=(14, 6))
    plt.title("F1 Score Comparison")
    plt.ylabel("F1 Score")
    plt.show()

    sns.boxplot(x="Sampling", y="Accuracy", data=results_df)
    plt.title("Accuracy Distribution by Sampling")
    plt.show()

    time_avg = results_df.groupby("Model")["Time"].mean()

    time_avg.plot(kind="bar")
    plt.title("Average Training Time per Model")
    plt.ylabel("Seconds")
    plt.show()

    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} AUC={auc(fpr, tpr):.2f}")

    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title("ROC Curves")
    plt.show()

    best = results_df.loc[results_df["Accuracy"].idxmax()]
    best_model = models[best["Model"]]
    best_sampler = sampling_methods[best["Sampling"]]

    X_res, y_res = best_sampler.fit_resample(X_scaled, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res)

    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)

    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix - Best Model")
    plt.show()

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    plt.bar(range(len(rf.feature_importances_)), rf.feature_importances_)
    plt.title("Feature Importance - Random Forest")
    plt.show()

    cv_scores = cross_val_score(RandomForestClassifier(), X_scaled, y, cv=5)
    print("Cross Validation Accuracy:", cv_scores.mean())

    avg_sampling = results_df.groupby("Sampling")["Accuracy"].mean()
    avg_sampling.plot(kind="bar")
    plt.title("Average Accuracy per Sampling")
    plt.show()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    pivot_acc.plot(kind="bar", figsize=(14, 6))
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")
    save_plot("accuracy_bar")

    pivot_f1.plot(kind="bar", figsize=(14, 6))
    plt.title("F1 Score Comparison")
    plt.ylabel("F1 Score")
    save_plot("f1_bar")

    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Sampling", y="Accuracy", data=results_df)
    plt.title("Sampling Stability")
    save_plot("sampling_boxplot")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    save_plot("roc_curve")

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    save_plot("confusion_matrix")

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(rf.feature_importances_)), rf.feature_importances_)
    plt.title("Feature Importance")
    save_plot("feature_importance")

    avg_sampling.plot(kind="bar", figsize=(8, 5))
    plt.title("Average Sampling Accuracy")
    save_plot("avg_sampling_accuracy")

    time_avg.plot(kind="bar", figsize=(8, 5))
    plt.title("Training Time per Model")
    save_plot("training_time")

    zip_name = "plots.zip"
    with ZipFile(zip_name, "w") as zipf:
        for file in os.listdir(PLOTS_DIR):
            zipf.write(PLOTS_DIR / file)

    print("All plots saved and zipped as plots.zip")


if __name__ == "__main__":
    main()