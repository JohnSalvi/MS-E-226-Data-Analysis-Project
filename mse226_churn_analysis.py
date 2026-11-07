# ============================================================
# ms&e 226 - project part 1: customer churn
# run this file in the same folder as Customer Churn.csv
# install packages: pip install pandas numpy matplotlib scikit-learn joblib
# ============================================================
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ============================================================
# 1. read csv and use every row as development data
# ============================================================
baseDir = Path(__file__).resolve().parent
outputDir = baseDir / "mse226_results"
outputDir.mkdir(exist_ok=True)
development = pd.read_csv(baseDir / "Customer Churn.csv")
development.columns = [" ".join(name.split()) for name in development.columns]
development = development.reset_index(drop=True)
y = development["Churn"]
missingValues = int(development.isna().sum().sum())
print("Development rows:", len(development))
print("Missing values:", missingValues)
print("The current CSV is the development set. This script does not split it again.")

# ============================================================
# 2. decide which columns to use
# ============================================================
categoricalCols = ["Complains", "Charge Amount", "Age Group", "Tariff Plan", "Status"]
numericCols = ["Call Failure", "Subscription Length", "Seconds of Use",
               "Frequency of use", "Frequency of SMS", "Distinct Called Numbers",
               "Age", "Customer Value"]
allFeatures = numericCols + categoricalCols
featureSets = {
    "all 13": allFeatures,
    "without status and customer value": [name for name in allFeatures
                                          if name not in ["Status", "Customer Value"]]
}

def makePreprocessing(features):
    numbers = [name for name in features if name in numericCols]
    categories = [name for name in features if name in categoricalCols]
    numberSteps = Pipeline([("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler())])
    categorySteps = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("encode", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numbers", numberSteps, numbers),
                              ("categories", categorySteps, categories)])

# ============================================================
# 3. explore the development data (not the holdout)
# ============================================================
churnCounts = y.value_counts().sort_index()
plt.figure(figsize=(5, 4))
plt.bar(["Stayed", "Churned"], [churnCounts.get(0, 0), churnCounts.get(1, 0)])
plt.ylabel("Customers")
plt.title("Customer churn in development data")
plt.tight_layout()
plt.savefig(outputDir / "01_churn_counts.png", dpi=160)
plt.close()

categoryResults = []
for name in categoricalCols:
    group = development.groupby(name)["Churn"].agg(["mean", "count"])
    for value, record in group.iterrows():
        categoryResults.append({"feature": name, "category": value,
                                "churnRate": record["mean"], "customers": record["count"]})
pd.DataFrame(categoryResults).to_csv(outputDir / "categorical_churn_rates.csv", index=False)

for name, fileName in [("Complains", "02_churn_by_complaint.png"),
                       ("Age Group", "03_churn_by_age_group.png")]:
    group = development.groupby(name)["Churn"].mean().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(group.index.astype(str), group.values * 100)
    plt.ylabel("Churn rate (%)")
    plt.xlabel(name)
    plt.title("Churn rate by " + name.lower())
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(outputDir / fileName, dpi=160)
    plt.close()

numericCorrelations = development[numericCols + ["Churn"]].corr()
numericCorrelations["Churn"].drop("Churn").sort_values().to_csv(
    outputDir / "numeric_churn_correlations.csv", header=["correlation"])
featureCorrelations = development[numericCols].corr()
plt.figure(figsize=(9, 7))
plt.imshow(featureCorrelations, vmin=-1, vmax=1, cmap="coolwarm")
plt.colorbar(label="Correlation")
plt.xticks(range(len(numericCols)), numericCols, rotation=60, ha="right")
plt.yticks(range(len(numericCols)), numericCols)
plt.title("Correlations between numeric predictors")
plt.tight_layout()
plt.savefig(outputDir / "04_feature_correlations.png", dpi=160)
plt.close()

pd.DataFrame({"column": development.columns, "missing": development.isna().sum().values}).to_csv(
    outputDir / "missing_values.csv", index=False)

# ============================================================
# 4. use the same 5 folds for every model
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=226)
metrics = {"ap": "average_precision", "auc": "roc_auc"}
baseModel = DummyClassifier(strategy="prior")
baseScores = cross_validate(baseModel, development[allFeatures], y,
                            cv=cv, scoring=metrics)
modelResults = [{"model": "Baseline (training churn rate)", "features": "none",
                 "averagePrecision": baseScores["test_ap"].mean(),
                 "rocAuc": baseScores["test_auc"].mean(),
                 "bestSettings": "training-fold class proportion", "tuningTrials": 1}]
trialResults = []

# ============================================================
# 5. logistic regression
# ============================================================
# fits a linear score, then converts it to a churn probability
models = {
    "Logistic regression": (LogisticRegression(max_iter=2000, random_state=226),
                            {"model__C": [0.1, 1, 10]}),

# ============================================================
# 6. decision tree
# ============================================================
# predicts using a series of if/else splits in customer features
    "Decision tree": (DecisionTreeClassifier(random_state=226),
                      {"model__max_depth": [3, 5, 8]}),

# ============================================================
# 7. random forest
# ============================================================
# averages predictions from many decision trees
    "Random forest": (RandomForestClassifier(n_estimators=100, random_state=226, n_jobs=-1),
                      {"model__max_depth": [5, None],
                       "model__min_samples_leaf": [2, 10]})
}

# ============================================================
# 8. compare all 3 algorithms with both feature sets
# ============================================================
bestModel = None
bestScore = -1
bestByAlgorithm = {}
for modelName, (model, settings) in models.items():
    for featureName, features in featureSets.items():
        pipeline = Pipeline([("preprocess", makePreprocessing(features)),
                             ("model", model)])
        search = GridSearchCV(pipeline, settings, cv=cv, scoring=metrics,
                              refit="ap", n_jobs=-1)
        search.fit(development[features], y)
        bestRow = search.best_index_
        ap = search.cv_results_["mean_test_ap"][bestRow]
        auc = search.cv_results_["mean_test_auc"][bestRow]
        modelResults.append({"model": modelName, "features": featureName,
                             "averagePrecision": ap, "rocAuc": auc,
                             "bestSettings": str(search.best_params_),
                             "tuningTrials": len(search.cv_results_["params"])})
        for i, params in enumerate(search.cv_results_["params"]):
            trialResults.append({"model": modelName, "features": featureName,
                                 "settings": str(params),
                                 "averagePrecision": search.cv_results_["mean_test_ap"][i],
                                 "rocAuc": search.cv_results_["mean_test_auc"][i]})
        print(modelName, "-", featureName, "AP:", round(ap, 4),
              "AUC:", round(auc, 4), "settings:", search.best_params_)
        if modelName not in bestByAlgorithm or ap > bestByAlgorithm[modelName]["score"]:
            bestByAlgorithm[modelName] = {"model": search.best_estimator_,
                                          "features": features, "score": ap}
        if ap > bestScore:
            bestScore = ap
            bestModel = search.best_estimator_
            bestFeatures = features
            bestName = modelName
            bestFeatureName = featureName
            bestSettings = search.best_params_

# ============================================================
# 9. save scores and graphs for the model comparisons
# ============================================================
comparison = pd.DataFrame(modelResults)
comparison.to_csv(outputDir / "model_comparison.csv", index=False)
pd.DataFrame(trialResults).to_csv(outputDir / "tuning_results.csv", index=False)
print("\nModel comparison:")
print(comparison[["model", "features", "averagePrecision", "rocAuc"]].round(4).to_string(index=False))

labels = [row["model"] + "\n" + row["features"] for row in modelResults]
values = [row["averagePrecision"] for row in modelResults]
plt.figure(figsize=(10, 5))
plt.bar(range(len(values)), values)
plt.xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8)
plt.ylabel("5-fold average precision")
plt.title("Model comparison: larger is better")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(outputDir / "05_model_comparison.png", dpi=160)
plt.close()

plt.figure(figsize=(6, 5))
for modelName, info in bestByAlgorithm.items():
    predictions = cross_val_predict(info["model"], development[info["features"]],
                                    y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    falsePositiveRate, truePositiveRate, thresholds = roc_curve(y, predictions)
    auc = roc_auc_score(y, predictions)
    plt.plot(falsePositiveRate, truePositiveRate,
             label=modelName + " (AUC " + str(round(auc, 3)) + ")")
plt.plot([0, 1], [0, 1], "--", label="Random guessing")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Cross-validated ROC curves")
plt.legend()
plt.tight_layout()
plt.savefig(outputDir / "06_roc_curves.png", dpi=160)
plt.close()

# ============================================================
# 10. save the selected model and the random forest
# ============================================================
# 0.5 is a fixed decision threshold; AP is used to select models
threshold = 0.5
joblib.dump({"pipeline": bestModel, "features": bestFeatures, "threshold": threshold},
            outputDir / "selected_model.joblib")
forestInfo = bestByAlgorithm["Random forest"]
joblib.dump({"pipeline": forestInfo["model"], "features": forestInfo["features"],
             "threshold": threshold}, outputDir / "random_forest_model.joblib")

bestPredictions = cross_val_predict(bestModel, development[bestFeatures], y,
                                    cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
confusion = confusion_matrix(y, bestPredictions >= threshold)
plt.figure(figsize=(5, 4))
plt.imshow(confusion, cmap="Blues")
for row in range(2):
    for col in range(2):
        plt.text(col, row, str(confusion[row, col]), ha="center", va="center")
plt.xticks([0, 1], ["Predict stay", "Predict churn"])
plt.yticks([0, 1], ["Actually stay", "Actually churn"])
plt.title("Selected model: cross-validated predictions")
plt.tight_layout()
plt.savefig(outputDir / "07_confusion_matrix.png", dpi=160)
plt.close()

# how many churners can a manager find by contacting the top 20%?
customersToContact = int(np.ceil(0.20 * len(y)))
topRows = np.argsort(bestPredictions)[-customersToContact:]
topRecall = y.iloc[topRows].sum() / y.sum()
with open(outputDir / "run_summary.txt", "w") as file:
    file.write("MS&E 226 - Iranian customer churn, project part 1\n")
    file.write("Dataset: UCI Iranian Churn / Customer Churn.csv\n")
    file.write("Random seed: 226\n")
    file.write("Missing values: " + str(missingValues) + "\n")
    file.write("Development size: " + str(len(development)) + " (entire current CSV; no further split)\n")
    file.write("5-fold stratified CV, seed 226, identical folds for all comparisons\n")
    file.write("Primary metric: average precision (AP); secondary: ROC AUC\n")
    file.write("Selected model: " + bestName + "; feature set: " + bestFeatureName + "\n")
    file.write("Selected settings: " + str(bestSettings) + "\n")
    file.write("Selected 5-fold AP: " + str(round(bestScore, 4)) + "\n")
    file.write("Cross-validated recall among top 20%: " + str(round(topRecall, 4)) + "\n")
    file.write("Classification threshold: " + str(threshold) + "\n")
    file.write("Python: " + sys.version.split()[0] + "\n")
    file.write("pandas: " + pd.__version__ + "; numpy: " + np.__version__ + "\n")
    file.write("sklearn: " + sklearn.__version__ + "; matplotlib: " + matplotlib.__version__ + "\n")
    file.write("The holdout was removed from this CSV and is not used here.\n")
print("\nSelected:", bestName, "with", bestFeatureName)
print("Recall of churners within top 20%:", round(topRecall, 4))
print("Plots, score tables, and models saved in:", outputDir)
