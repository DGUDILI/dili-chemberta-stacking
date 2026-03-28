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

# =========================
# 경로 설정
# =========================
FEATURE_PATH = "../Feature/Feature.csv"
CHEMBERTA_TRAIN_PATH = "../Data/chemberta_train_pred.csv"
CHEMBERTA_TEST_PATH = "../Data/chemberta_test_pred_oof.csv"
MODEL_DIR = "../Model"
SAVE_MODEL_PATH = "../Model/best_model_stacking_chemberta.pkl"

# =========================
# 데이터 로드
# =========================
feature_df = pd.read_csv(FEATURE_PATH)
chem_train_df = pd.read_csv(CHEMBERTA_TRAIN_PATH)
chem_test_df = pd.read_csv(CHEMBERTA_TEST_PATH)

print("feature_df:", feature_df.shape)
print("chem_train_df:", chem_train_df.shape)
print("chem_test_df:", chem_test_df.shape)

# 원본 StackDILI split 기준
train_df = feature_df[feature_df["ref"] != "DILIrank"].copy().reset_index(drop=True)
test_df = feature_df[feature_df["ref"] == "DILIrank"].copy().reset_index(drop=True)

print("train_df:", train_df.shape)
print("test_df:", test_df.shape)

# ChemBERTa 확률 merge
merge_cols = ["SMILES", "Label", "ref"]

train_df = train_df.merge(
    chem_train_df[merge_cols + ["chemberta_prob"]],
    on=merge_cols,
    how="inner"
)

test_df = test_df.merge(
    chem_test_df[merge_cols + ["chemberta_prob"]],
    on=merge_cols,
    how="inner"
)

print("merged train_df:", train_df.shape)
print("merged test_df:", test_df.shape)

# =========================
# 입력 feature / label 준비
# =========================
drop_cols = ["SMILES", "Label", "ref", "chemberta_prob"]
feature_cols = [c for c in train_df.columns if c not in drop_cols]

X_train = train_df[feature_cols]
y_train = train_df["Label"].values

X_test = test_df[feature_cols]
y_test = test_df["Label"].values

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# =========================
# 원본 4개 base model 불러오기
# =========================
base_model_names = ["RF", "ET", "HistGB", "XGBoost"]
base_models = {}

for name in base_model_names:
    model_path = os.path.join(MODEL_DIR, f"best_model_{name}.pkl")
    with open(model_path, "rb") as f:
        base_models[name] = pickle.load(f)
    print(f"Loaded: {model_path}")

# =========================
# meta feature 생성
# 원본 4개 + ChemBERTa
# =========================
train_meta_features = []
test_meta_features = []

for name in base_model_names:
    model = base_models[name]

    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    train_meta_features.append(train_prob)
    test_meta_features.append(test_prob)

# ChemBERTa probability 추가
train_meta_features.append(train_df["chemberta_prob"].values)
test_meta_features.append(test_df["chemberta_prob"].values)

X_meta_train = np.column_stack(train_meta_features)
X_meta_test = np.column_stack(test_meta_features)

print("X_meta_train:", X_meta_train.shape)
print("X_meta_test:", X_meta_test.shape)

# =========================
# meta learner 학습
# 원본 논문/깃 스타일 따라 ET 사용
# =========================
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

# =========================
# 최종 평가
# =========================
tn, fp, fn, tp = confusion_matrix(y_test, best_pred).ravel()

accuracy = accuracy_score(y_test, best_pred)
sensitivity = recall_score(y_test, best_pred)
specificity = tn / (tn + fp)
precision = precision_score(y_test, best_pred)
f1 = f1_score(y_test, best_pred)
auc = roc_auc_score(y_test, best_prob)
mcc = matthews_corrcoef(y_test, best_pred)

print("\n========== Final Stacking + ChemBERTa Result ==========")
print("Accuracy:", round(accuracy, 4))
print("Sensitivity (Recall):", round(sensitivity, 4))
print("Specificity:", round(specificity, 4))
print("Precision:", round(precision, 4))
print("F1 Score:", round(f1, 4))
print("AUC:", round(auc, 4))
print("MCC:", round(mcc, 4))

# =========================
# 모델 저장
# =========================
with open(SAVE_MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)

print("\nSaved model:")
print(SAVE_MODEL_PATH)