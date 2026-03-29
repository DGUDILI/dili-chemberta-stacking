from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = ROOT / "data" / "processed" / "Feature.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "chemberta_train.csv"
TEST_PATH = ROOT / "data" / "processed" / "chemberta_test.csv"
EMB_DIR = ROOT / "data" / "embeddings"
RESULT_DIR = ROOT / "results" / "fusion_concat"

LABEL_COL = "Label"
KEY_COL = "SMILES"
META_COLS = ["SMILES", "Label", "ref"]

SEED = 42
BATCH_SIZE = 64
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    y_train = train_merged[LABEL_COL].values.astype(np.float32)
    y_test = test_merged[LABEL_COL].values.astype(np.float32)

    print("FP train matched:", len(train_merged), "/", len(train_df))
    print("FP test matched:", len(test_merged), "/", len(test_df))
    print("num FP features:", len(feature_cols))

    return x_train_fp, x_test_fp, y_train, y_test, feature_cols


def load_chemberta_embeddings():
    x_train_chem = np.load(EMB_DIR / "chemberta_only_train_mean.npy").astype(np.float32)
    x_test_chem = np.load(EMB_DIR / "chemberta_only_test_mean.npy").astype(np.float32)

    print("ChemBERTa pooling: mean")
    print("x_train_chem shape:", x_train_chem.shape)
    print("x_test_chem shape:", x_test_chem.shape)

    return x_train_chem, x_test_chem


class ProjectedConcatModel(nn.Module):
    def __init__(self, fp_dim: int, chem_dim: int, proj_dim: int = 16):
        super().__init__()

        self.fp_proj = nn.Sequential(
            nn.Linear(fp_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, proj_dim),
        )

        self.chem_proj = nn.Sequential(
            nn.Linear(chem_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, proj_dim),
        )

        self.classifier = nn.Linear(proj_dim * 2, 1)

    def forward(self, x_fp, x_chem):
        z_fp = self.fp_proj(x_fp)
        z_chem = self.chem_proj(x_chem)
        z = torch.cat([z_fp, z_chem], dim=1)
        logits = self.classifier(z).squeeze(1)
        return logits


def prepare_data():
    x_train_fp, x_test_fp, y_train, y_test, fp_cols = load_fp_features()
    x_train_chem, x_test_chem = load_chemberta_embeddings()

    if len(x_train_fp) != len(x_train_chem) or len(x_test_fp) != len(x_test_chem):
        raise ValueError("FP feature rows and ChemBERTa embedding rows do not match.")

    fp_scaler = StandardScaler()
    chem_scaler = StandardScaler()

    x_train_fp = fp_scaler.fit_transform(x_train_fp).astype(np.float32)
    x_test_fp = fp_scaler.transform(x_test_fp).astype(np.float32)

    x_train_chem = chem_scaler.fit_transform(x_train_chem).astype(np.float32)
    x_test_chem = chem_scaler.transform(x_test_chem).astype(np.float32)

    return x_train_fp, x_test_fp, x_train_chem, x_test_chem, y_train, y_test, fp_cols


def train_model(model, train_loader, y_train):
    num_pos = float((y_train == 1).sum())
    num_neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([num_neg / max(num_pos, 1.0)], dtype=torch.float32, device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0

        for batch_fp, batch_chem, batch_y in train_loader:
            batch_fp = batch_fp.to(DEVICE)
            batch_chem = batch_chem.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_fp, batch_chem)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_fp.size(0)

        epoch_loss /= len(train_loader.dataset)

        if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"Epoch {epoch:03d} | train_loss={epoch_loss:.4f}")


@torch.no_grad()
def predict_proba(model, x_fp, x_chem):
    model.eval()
    x_fp = torch.tensor(x_fp, dtype=torch.float32, device=DEVICE)
    x_chem = torch.tensor(x_chem, dtype=torch.float32, device=DEVICE)
    logits = model(x_fp, x_chem)
    prob = torch.sigmoid(logits).cpu().numpy()
    return prob


def main():
    set_seed()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    x_train_fp, x_test_fp, x_train_chem, x_test_chem, y_train, y_test, fp_cols = prepare_data()

    print("scaled x_train_fp:", x_train_fp.shape)
    print("scaled x_train_chem:", x_train_chem.shape)
    print("device:", DEVICE)

    train_ds = TensorDataset(
        torch.tensor(x_train_fp, dtype=torch.float32),
        torch.tensor(x_train_chem, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = ProjectedConcatModel(
        fp_dim=x_train_fp.shape[1],
        chem_dim=x_train_chem.shape[1],
        proj_dim=16,
    ).to(DEVICE)

    train_model(model, train_loader, y_train)

    prob = predict_proba(model, x_test_fp, x_test_chem)
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "model": "fp_plus_chemberta_projected_concat",
        "pooling": "mean",
        "fp_input_dim": int(x_train_fp.shape[1]),
        "chem_input_dim": int(x_train_chem.shape[1]),
        "proj_dim": 16,
        "test_auc": float(roc_auc_score(y_test, prob)),
        "test_f1": float(f1_score(y_test, pred)),
        "test_mcc": float(matthews_corrcoef(y_test, pred)),
        "test_acc": float(accuracy_score(y_test, pred)),
        "num_train": int(len(y_train)),
        "num_test": int(len(y_test)),
    }

    out_path = RESULT_DIR / "projected_concat_mean_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    model_path = RESULT_DIR / "projected_concat_mean_model.pt"
    torch.save(model.state_dict(), model_path)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nsaved metrics to: {out_path}")
    print(f"saved model to: {model_path}")


if __name__ == "__main__":
    main()