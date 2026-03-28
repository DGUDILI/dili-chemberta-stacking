import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

TRAIN_CSV = "../Data/chemberta_train.csv"
TEST_CSV = "../Data/chemberta_test.csv"
TRAIN_EMB = "../Data/chemberta_train_embed.npy"
TEST_EMB = "../Data/chemberta_test_embed.npy"

SAVE_MODEL = "../Model/best_mlp_on_chemberta_embed.pt"
SAVE_TEST_PRED = "../Data/mlp_chemberta_test_pred.csv"

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

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

X_train_full = np.load(TRAIN_EMB)
X_test = np.load(TEST_EMB)

y_train_full = train_df["Label"].values.astype(np.float32)
y_test = test_df["Label"].values.astype(np.float32)

print("X_train_full:", X_train_full.shape)
print("X_test:", X_test.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.1,
    stratify=y_train_full,
    random_state=SEED
)

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
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

train_ds = EmbeddingDataset(X_train, y_train)
val_ds = EmbeddingDataset(X_val, y_val)
test_ds = EmbeddingDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

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

model = MLP(input_dim=X_train.shape[1]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def evaluate(model, loader):
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
    val_acc, val_f1, val_auc, _ = evaluate(model, val_loader)

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

test_acc, test_f1, test_auc, test_probs = evaluate(model, test_loader)
test_preds = (test_probs >= 0.5).astype(int)

print("\n[MLP on ChemBERTa embedding - Test Result]")
print("Accuracy:", round(test_acc, 4))
print("F1:", round(test_f1, 4))
print("AUC:", round(test_auc, 4))

torch.save(model.state_dict(), SAVE_MODEL)

result_df = test_df.copy()
result_df["mlp_prob"] = test_probs
result_df["mlp_pred"] = test_preds
result_df.to_csv(SAVE_TEST_PRED, index=False)

print("\n저장 완료:")
print(SAVE_MODEL)
print(SAVE_TEST_PRED)