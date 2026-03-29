from pathlib import Path
import copy
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = ROOT / "data" / "processed" / "Feature.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "chemberta_train.csv"
TEST_PATH = ROOT / "data" / "processed" / "chemberta_test.csv"
EMB_DIR = ROOT / "data" / "embeddings"
RESULT_DIR = ROOT / "results" / "fusion_crossattn"

LABEL_COL = "Label"
KEY_COL = "SMILES"
META_COLS = ["SMILES", "Label", "ref"]

SEED = 42
BATCH_SIZE = 64
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_TOKENS = 4
TOKEN_DIM = 16
NUM_HEADS = 4
DROPOUT = 0.2


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


class CrossAttentionFusionModel(nn.Module):
    def __init__(self, fp_dim: int, chem_dim: int, num_tokens: int = 4, token_dim: int = 16):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim

        self.fp_tokenizer = nn.Sequential(
            nn.Linear(fp_dim, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, num_tokens * token_dim),
        )

        self.chem_tokenizer = nn.Sequential(
            nn.Linear(chem_dim, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, num_tokens * token_dim),
        )

        self.fp_to_chem_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.chem_to_fp_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )

        self.fp_norm1 = nn.LayerNorm(token_dim)
        self.chem_norm1 = nn.LayerNorm(token_dim)
        self.fp_norm2 = nn.LayerNorm(token_dim)
        self.chem_norm2 = nn.LayerNorm(token_dim)

        self.fp_ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.chem_ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(token_dim * 2, token_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(token_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x_fp, x_chem):
        fp_tokens = self.fp_tokenizer(x_fp).view(-1, self.num_tokens, self.token_dim)
        chem_tokens = self.chem_tokenizer(x_chem).view(-1, self.num_tokens, self.token_dim)

        fp_attn, _ = self.fp_to_chem_attn(query=fp_tokens, key=chem_tokens, value=chem_tokens)
        chem_attn, _ = self.chem_to_fp_attn(query=chem_tokens, key=fp_tokens, value=fp_tokens)

        fp_tokens = self.fp_norm1(fp_tokens + fp_attn)
        chem_tokens = self.chem_norm1(chem_tokens + chem_attn)

        fp_tokens = self.fp_norm2(fp_tokens + self.fp_ffn(fp_tokens))
        chem_tokens = self.chem_norm2(chem_tokens + self.chem_ffn(chem_tokens))

        fp_pooled = fp_tokens.mean(dim=1)
        chem_pooled = chem_tokens.mean(dim=1)

        fused = torch.cat([fp_pooled, chem_pooled], dim=1)
        logits = self.classifier(fused).squeeze(1)
        return logits


def build_loaders(x_fp, x_chem, y):
    x_fp_tr, x_fp_val, x_chem_tr, x_chem_val, y_tr, y_val = train_test_split(
        x_fp, x_chem, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    train_ds = TensorDataset(
        torch.tensor(x_fp_tr, dtype=torch.float32),
        torch.tensor(x_chem_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(x_fp_val, dtype=torch.float32),
        torch.tensor(x_chem_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, y_tr, y_val


@torch.no_grad()
def predict_proba_loader(model, loader):
    model.eval()
    probs, ys = [], []

    for batch_fp, batch_chem, batch_y in loader:
        batch_fp = batch_fp.to(DEVICE)
        batch_chem = batch_chem.to(DEVICE)

        logits = model(batch_fp, batch_chem)
        prob = torch.sigmoid(logits).cpu().numpy()

        probs.append(prob)
        ys.append(batch_y.numpy())

    probs = np.concatenate(probs)
    ys = np.concatenate(ys)
    return probs, ys


@torch.no_grad()
def predict_proba_array(model, x_fp, x_chem):
    model.eval()
    x_fp = torch.tensor(x_fp, dtype=torch.float32, device=DEVICE)
    x_chem = torch.tensor(x_chem, dtype=torch.float32, device=DEVICE)
    logits = model(x_fp, x_chem)
    prob = torch.sigmoid(logits).cpu().numpy()
    return prob


def train_model(model, train_loader, val_loader, y_train):
    num_pos = float((y_train == 1).sum())
    num_neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([num_neg / max(num_pos, 1.0)], dtype=torch.float32, device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_auc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
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

        val_prob, val_y = predict_proba_loader(model, val_loader)
        val_auc = roc_auc_score(val_y, val_prob)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"Epoch {epoch:03d} | train_loss={epoch_loss:.4f} | val_auc={val_auc:.4f} | best_val_auc={best_val_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_auc


def main():
    set_seed()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    x_train_fp, x_test_fp, x_train_chem, x_test_chem, y_train, y_test, fp_cols = prepare_data()

    print("scaled x_train_fp:", x_train_fp.shape)
    print("scaled x_train_chem:", x_train_chem.shape)
    print("device:", DEVICE)

    train_loader, val_loader, y_tr, y_val = build_loaders(x_train_fp, x_train_chem, y_train)

    model = CrossAttentionFusionModel(
        fp_dim=x_train_fp.shape[1],
        chem_dim=x_train_chem.shape[1],
        num_tokens=NUM_TOKENS,
        token_dim=TOKEN_DIM,
    ).to(DEVICE)

    best_val_auc = train_model(model, train_loader, val_loader, y_tr)

    test_prob = predict_proba_array(model, x_test_fp, x_test_chem)
    test_pred = (test_prob >= 0.5).astype(int)

    metrics = {
        "model": "fp_plus_chemberta_crossattn",
        "pooling": "mean",
        "fp_input_dim": int(x_train_fp.shape[1]),
        "chem_input_dim": int(x_train_chem.shape[1]),
        "num_tokens_per_branch": NUM_TOKENS,
        "token_dim": TOKEN_DIM,
        "best_val_auc": float(best_val_auc),
        "test_auc": float(roc_auc_score(y_test, test_prob)),
        "test_f1": float(f1_score(y_test, test_pred)),
        "test_mcc": float(matthews_corrcoef(y_test, test_pred)),
        "test_acc": float(accuracy_score(y_test, test_pred)),
        "num_train": int(len(y_train)),
        "num_test": int(len(y_test)),
    }

    out_path = RESULT_DIR / "crossattn_mean_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    model_path = RESULT_DIR / "crossattn_mean_model.pt"
    torch.save(model.state_dict(), model_path)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nsaved metrics to: {out_path}")
    print(f"saved model to: {model_path}")


if __name__ == "__main__":
    main()
