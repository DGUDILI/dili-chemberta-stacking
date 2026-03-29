from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, accuracy_score

ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = ROOT / "data" / "processed" / "chemberta_train.csv"
TEST_CSV = ROOT / "data" / "processed" / "chemberta_test.csv"
EMB_DIR = ROOT / "data" / "embeddings"
RESULT_DIR = ROOT / "results" / "chemberta_only"

LABEL_COL = "Label"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--c", type=float, default=1.0)
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    x_train = np.load(EMB_DIR / f"chemberta_only_train_{args.pooling}.npy")
    x_test = np.load(EMB_DIR / f"chemberta_only_test_{args.pooling}.npy")
    y_train = train_df[LABEL_COL].values
    y_test = test_df[LABEL_COL].values

    clf = LogisticRegression(
        C=args.c,
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(x_train, y_train)

    prob = clf.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "pooling": args.pooling,
        "test_auc": float(roc_auc_score(y_test, prob)),
        "test_f1": float(f1_score(y_test, pred)),
        "test_mcc": float(matthews_corrcoef(y_test, pred)),
        "test_acc": float(accuracy_score(y_test, pred)),
    }

    out_path = RESULT_DIR / f"lr_metrics_{args.pooling}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nsaved to: {out_path}")


if __name__ == "__main__":
    main()