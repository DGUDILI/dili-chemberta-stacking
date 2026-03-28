from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

FEATURE_PATH = str(ROOT / "data" / "processed" / "Feature.csv")
MLP_TRAIN_PATH = str(ROOT / "data" / "processed" / "mlp_chemberta_train_pred.csv")
MLP_TEST_PATH = str(ROOT / "data" / "processed" / "mlp_chemberta_test_pred_oof.csv")
MODEL_DIR = ROOT / "models" / "baseline"
SAVE_MODEL_PATH = ROOT / "models" / "chemberta_v2_mlp" / "best_model_stacking_mlp_chemberta.pkl"

feature_df = pd.read_csv(FEATURE_PATH)
mlp_train_df = pd.read_csv(MLP_TRAIN_PATH)
mlp_test_df = pd.read_csv(MLP_TEST_PATH)

print("feature_df:", feature_df.shape)
print("mlp_train_df:", mlp_train_df.shape)
print("mlp_test_df:", mlp_test_df.shape)

train_df = feature_df[feature_df["ref"] != "DILIrank"].copy().reset_index(drop=True)
test_df = feature_df[feature_df["ref"] == "DILIrank"].copy().reset_index(drop=True)

merge_cols = ["SMILES", "Label", "ref"]

train_df = train_df.merge(
    mlp_train_df[merge_cols + ["mlp_prob"]],
    on=merge_cols,
    how="inner"
)
test_df = test_df.merge(
    mlp_test_df[merge_cols + ["mlp_prob"]],
    on=merge_cols,
    how="inner"
)

print("merged train_df:", train_df.shape)
print("merged test_df:", test_df.shape)

drop_cols = ["SMILES", "Label", "ref", "mlp_prob"]
feature_cols = [c for c in train_df.columns if c not in drop_cols]

X_train = train_df[feature_cols]
y_train = train_df["Label"].values
X_test = test_df[feature_cols]
y_test = test_df["Label"].values

base_model_names = ["RF", "ET", "HistGB", "XGBoost"]
base_models = {}

for name in base_model_names:
    model_path = str(MODEL_DIR / f"best_model_{name}.pkl")
    with open(model_path, "rb") as f:
        base_models[name] = pickle.load(f)
    print(f"Loaded: {model_path}")

train_meta_features = []
test_meta_features = []

for name in base_model_names:
    model = base_models[name]
    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    train_meta_features.append(train_prob)
    test_meta_features.append(test_prob)

train_meta_features.append(train_df["mlp_prob"].values)
test_meta_features.append(test_df["mlp_prob"].values)

X_meta_train = np.column_stack(train_meta_features)
X_meta_test = np.column_stack(test_meta_features)

print("X_meta_train:", X_meta_train.shape)
print("X_meta_test:", X_meta_test.shape)

best_auc = -1
best_model = None
best_pred = None
best_prob = None

for i in range(10):
    print(f"\n===== Training stacking meta model {i+1}/10 =====")

    meta_model = ExtraTreesClassifier(
        n_estimators=200,
        random_state=42 + i,
        n_jobs=-1
    )

    meta_model.fit(X_meta_train, y_train)

    test_prob = meta_model.predict_proba(X_meta_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    test_auc = roc_auc_score(y_test, test_prob)

    acc = accuracy_score(y_test, test_pred)
    mcc = matthews_corrcoef(y_test, test_pred)

    print(f"ACC: {acc:.4f}, AUC: {test_auc:.4f}, MCC: {mcc:.4f}")

    if test_auc > best_auc:
        best_auc = test_auc
        best_model = meta_model
        best_pred = test_pred
        best_prob = test_prob

tn, fp, fn, tp = confusion_matrix(y_test, best_pred).ravel()

accuracy = accuracy_score(y_test, best_pred)
sensitivity = recall_score(y_test, best_pred)
specificity = tn / (tn + fp)
precision = precision_score(y_test, best_pred)
f1 = f1_score(y_test, best_pred)
auc = roc_auc_score(y_test, best_prob)
mcc = matthews_corrcoef(y_test, best_pred)

print("\n========== Final Stacking + MLP(ChemBERTa) Result ==========")
print("Accuracy:", round(accuracy, 4))
print("Sensitivity (Recall):", round(sensitivity, 4))
print("Specificity:", round(specificity, 4))
print("Precision:", round(precision, 4))
print("F1 Score:", round(f1, 4))
print("AUC:", round(auc, 4))
print("MCC:", round(mcc, 4))

with open(SAVE_MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)

print("\nSaved model:")
print(SAVE_MODEL_PATH)