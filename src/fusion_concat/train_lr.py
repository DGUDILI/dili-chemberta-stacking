from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = ROOT / "data" / "processed" / "Feature.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "chemberta_train.csv"
TEST_PATH = ROOT / "data" / "processed" / "chemberta_test.csv"
EMB_DIR = ROOT / "data" / "embeddings"
RESULT_DIR = ROOT / "results" / "fusion_concat"

LABEL_COL = "Label"
KEY_COL = "SMILES"
META_COLS = ["SMILES", "Label", "ref"]


def load_fp_features():
    feature_df = pd.read_csv(FEATURE_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_merged = train_df[[KEY_COL, LABEL_COL]].merge(
        feature_df, on=[KEY_COL, LABEL_COL], how="inner"
    )
    test_merged = test_df[[KEY_COL, LABEL_COL]].merge(
        feature_df, on=[KEY_COL, LABEL_COL], how="inner"
    )

    feature_cols = [c for c in train_merged.columns if c not in META_COLS]

    x_train_fp = train_merged[feature_cols].astype(np.float32).values
    x_test_fp = test_merged[feature_cols].astype(np.float32).values
    y_train = train_merged[LABEL_COL].values
    y_test = test_merged[LABEL_COL].values

    print("FP train matched:", len(train_merged), "/", len(train_df))
    print("FP test matched:", len(test_merged), "/", len(test_df))
    print("num FP features:", len(feature_cols))

    return x_train_fp, x_test_fp, y_train, y_test, feature_cols


def load_chemberta_embeddings(pooling: str):
    x_train_chem = np.load(EMB_DIR / f"chemberta_only_train_{pooling}.npy").astype(np.float32)
    x_test_chem = np.load(EMB_DIR / f"chemberta_only_test_{pooling}.npy").astype(np.float32)

    print("ChemBERTa pooling:", pooling)
    print("x_train_chem shape:", x_train_chem.shape)
    print("x_test_chem shape:", x_test_chem.shape)

    return x_train_chem, x_test_chem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--scale", action="store_true")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    x_train_fp, x_test_fp, y_train, y_test, fp_cols = load_fp_features()
    x_train_chem, x_test_chem = load_chemberta_embeddings(args.pooling)

    if len(x_train_fp) != len(x_train_chem) or len(x_test_fp) != len(x_test_chem):
        raise ValueError("FP feature rows and ChemBERTa embedding rows do not match.")

    x_train = np.concatenate([x_train_fp, x_train_chem], axis=1)
    x_test = np.concatenate([x_test_fp, x_test_chem], axis=1)

    print("concat x_train shape:", x_train.shape)
    print("concat x_test shape:", x_test.shape)
    print("use scaling:", args.scale)

    if args.scale:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=args.c,
                max_iter=5000,
                class_weight="balanced",
                random_state=42,
            )),
        ])
    else:
        clf = LogisticRegression(
            C=args.c,
            max_iter=5000,
            class_weight="balanced",
            random_state=42,
        )

    clf.fit(x_train, y_train)

    prob = clf.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    suffix = f"{args.pooling}_{'scaled' if args.scale else 'raw'}"

    metrics = {
        "model": "fp_plus_chemberta_concat_lr",
        "pooling": args.pooling,
        "scaled": args.scale,
        "test_auc": float(roc_auc_score(y_test, prob)),
        "test_f1": float(f1_score(y_test, pred)),
        "test_mcc": float(matthews_corrcoef(y_test, pred)),
        "test_acc": float(accuracy_score(y_test, pred)),
        "num_fp_features": len(fp_cols),
        "num_chemberta_features": int(x_train_chem.shape[1]),
        "num_total_features": int(x_train.shape[1]),
        "num_train": int(len(y_train)),
        "num_test": int(len(y_test)),
    }

    out_path = RESULT_DIR / f"lr_metrics_{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nsaved to: {out_path}")


if __name__ == "__main__":
    main()