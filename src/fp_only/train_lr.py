from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, accuracy_score

ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = ROOT / "data" / "processed" / "Feature.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "chemberta_train.csv"
TEST_PATH = ROOT / "data" / "processed" / "chemberta_test.csv"
RESULT_DIR = ROOT / "results" / "fp_only"

LABEL_COL = "Label"
KEY_COL = "SMILES"
META_COLS = ["SMILES", "Label", "ref"]


def prepare_xy():
    feature_df = pd.read_csv(FEATURE_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # train/test split을 chemberta csv 기준으로 맞춤
    train_merged = train_df[[KEY_COL, LABEL_COL]].merge(
        feature_df, on=[KEY_COL, LABEL_COL], how="inner"
    )
    test_merged = test_df[[KEY_COL, LABEL_COL]].merge(
        feature_df, on=[KEY_COL, LABEL_COL], how="inner"
    )

    feature_cols = [c for c in train_merged.columns if c not in META_COLS]

    x_train = train_merged[feature_cols].astype(np.float32).values
    x_test = test_merged[feature_cols].astype(np.float32).values
    y_train = train_merged[LABEL_COL].values
    y_test = test_merged[LABEL_COL].values

    print("train matched:", len(train_merged), "/", len(train_df))
    print("test matched:", len(test_merged), "/", len(test_df))
    print("num features:", len(feature_cols))
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)

    return x_train, x_test, y_train, y_test, feature_cols


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test, feature_cols = prepare_xy()

    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(x_train, y_train)

    prob = clf.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "model": "fp_only_lr",
        "test_auc": float(roc_auc_score(y_test, prob)),
        "test_f1": float(f1_score(y_test, pred)),
        "test_mcc": float(matthews_corrcoef(y_test, pred)),
        "test_acc": float(accuracy_score(y_test, pred)),
        "num_features": len(feature_cols),
        "num_train": int(len(y_train)),
        "num_test": int(len(y_test)),
    }

    with open(RESULT_DIR / "lr_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nsaved to: {RESULT_DIR / 'lr_metrics.json'}")


if __name__ == "__main__":
    main()