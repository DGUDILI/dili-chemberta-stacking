import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
TRAIN_PATH = "../Data/chemberta_train.csv"
TEST_PATH = "../Data/chemberta_test.csv"

TRAIN_EMB_PATH = "../Data/chemberta_train_embed.npy"
TEST_EMB_PATH = "../Data/chemberta_test_embed.npy"

BATCH_SIZE = 32
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("train shape:", train_df.shape)
print("test shape:", test_df.shape)
print("device:", DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

def extract_embeddings(smiles_list):
    all_embeddings = []

    for i in range(0, len(smiles_list), BATCH_SIZE):
        batch_smiles = smiles_list[i:i+BATCH_SIZE]

        encoded = tokenizer(
            batch_smiles,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            pooled = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])

        all_embeddings.append(pooled.cpu().numpy())

        if (i // BATCH_SIZE + 1) % 10 == 0 or i + BATCH_SIZE >= len(smiles_list):
            print(f"processed {min(i + BATCH_SIZE, len(smiles_list))} / {len(smiles_list)}")

    return np.vstack(all_embeddings)

train_emb = extract_embeddings(train_df["SMILES"].tolist())
test_emb = extract_embeddings(test_df["SMILES"].tolist())

print("train_emb shape:", train_emb.shape)
print("test_emb shape:", test_emb.shape)

np.save(TRAIN_EMB_PATH, train_emb)
np.save(TEST_EMB_PATH, test_emb)

print("\n저장 완료:")
print(TRAIN_EMB_PATH)
print(TEST_EMB_PATH)