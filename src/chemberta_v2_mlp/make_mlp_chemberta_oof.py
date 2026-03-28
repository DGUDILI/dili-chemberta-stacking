import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

TRAIN_CSV = "../Data/chemberta_train.csv"
TEST_CSV = "../Data/chemberta_test.csv"
TRAIN_EMB = "../Data/chemberta_train_embed.npy"
TEST_EMB = "../Data/chemberta_test_embed.npy"

OOF_SAVE = "../Data/mlp_chemberta_train_pred.csv"
TEST_SAVE = "../Data/mlp_chemberta_test_pred_oof.csv"
MODEL_SAVE = "../Model/best_mlp_chemberta_oof.pt"

N_SPLITS = 5
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

train_df = pd.read_csv(TRAIN_CSV).reset_index(drop=True)
test_df = pd.read_csv(TEST_CSV).reset_index(drop=True)

X_train_full = np.load(TRAIN_EMB)
X_test = np.load(TEST_EMB)

y_train_full = train_df["Label"].values.astype(np.float32)
y_test = test_df["Label"].values.astype(np.float32)

print("X_train_full:", X_train_full.shape)
print("X_test:", X_test.shape)
print("device:", DEVICE)

class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

def evaluate_probs(model, loader):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, probs)

    return acc, f1, auc, probs

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_probs = np.zeros(len(train_df), dtype=float)
test_probs_folds = []

best_global_auc = -1
best_global_state = None

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), start=1):
    print(f"\n========== Fold {fold}/{N_SPLITS} ==========")

    X_tr = X_train_full[tr_idx]
    y_tr = y_train_full[tr_idx]
    X_val = X_train_full[val_idx]
    y_val = y_train_full[val_idx]

    print("fold_train:", X_tr.shape)
    print("fold_val:", X_val.shape)

    train_ds = EmbeddingDataset(X_tr, y_tr)
    val_ds = EmbeddingDataset(X_val, y_val)
    test_ds = EmbeddingDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MLP(input_dim=X_train_full.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_auc = -1
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_acc, val_f1, val_auc, _ = evaluate_probs(model, val_loader)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"val_auc={val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    # val -> OOF
    val_acc, val_f1, val_auc, val_probs = evaluate_probs(model, val_loader)
    oof_probs[val_idx] = val_probs
    print(f"Fold {fold} val AUC: {val_auc:.4f}")

    # test -> fold prediction
    test_acc, test_f1, test_auc, test_probs = evaluate_probs(model, test_loader)
    test_probs_folds.append(test_probs)
    print(f"Fold {fold} test AUC: {test_auc:.4f}")

    if val_auc > best_global_auc:
        best_global_auc = val_auc
        best_global_state = copy.deepcopy(model.state_dict())

# OOF result
oof_pred = (oof_probs >= 0.5).astype(int)
oof_acc = accuracy_score(y_train_full, oof_pred)
oof_f1 = f1_score(y_train_full, oof_pred)
oof_auc = roc_auc_score(y_train_full, oof_probs)

print("\n========== OOF RESULT ==========")
print("OOF Accuracy:", round(oof_acc, 4))
print("OOF F1:", round(oof_f1, 4))
print("OOF AUC:", round(oof_auc, 4))

# mean test result
test_probs_mean = np.mean(np.vstack(test_probs_folds), axis=0)
test_pred = (test_probs_mean >= 0.5).astype(int)
test_acc = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred)
test_auc = roc_auc_score(y_test, test_probs_mean)

print("\n========== TEST RESULT (mean over folds) ==========")
print("Test Accuracy:", round(test_acc, 4))
print("Test F1:", round(test_f1, 4))
print("Test AUC:", round(test_auc, 4))

# save files
oof_df = train_df.copy()
oof_df["mlp_prob"] = oof_probs
oof_df.to_csv(OOF_SAVE, index=False)

test_out_df = test_df.copy()
test_out_df["mlp_prob"] = test_probs_mean
test_out_df.to_csv(TEST_SAVE, index=False)

torch.save(best_global_state, MODEL_SAVE)

print("\n저장 완료:")
print(OOF_SAVE)
print(TEST_SAVE)
print(MODEL_SAVE)