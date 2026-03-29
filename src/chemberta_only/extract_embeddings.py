from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
TRAIN_PATH = ROOT / "data" / "processed" / "chemberta_train.csv"
TEST_PATH = ROOT / "data" / "processed" / "chemberta_test.csv"
EMB_DIR = ROOT / "data" / "embeddings"

BATCH_SIZE = 32
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def cls_pooling(last_hidden_state: torch.Tensor) -> torch.Tensor:
    return last_hidden_state[:, 0, :]


def extract_embeddings(smiles_list, tokenizer, model, pooling: str):
    all_embeddings = []

    for i in range(0, len(smiles_list), BATCH_SIZE):
        batch_smiles = smiles_list[i:i + BATCH_SIZE]

        encoded = tokenizer(
            batch_smiles,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

            if pooling == "mean":
                pooled = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
            elif pooling == "cls":
                pooled = cls_pooling(outputs.last_hidden_state)
            else:
                raise ValueError(f"Unsupported pooling: {pooling}")

        all_embeddings.append(pooled.cpu().numpy())

        done = min(i + BATCH_SIZE, len(smiles_list))
        if (i // BATCH_SIZE + 1) % 10 == 0 or done == len(smiles_list):
            print(f"[{pooling}] processed {done} / {len(smiles_list)}")

    return np.vstack(all_embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    args = parser.parse_args()

    EMB_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("train shape:", train_df.shape)
    print("test shape:", test_df.shape)
    print("device:", DEVICE)
    print("pooling:", args.pooling)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    train_emb = extract_embeddings(train_df["SMILES"].tolist(), tokenizer, model, args.pooling)
    test_emb = extract_embeddings(test_df["SMILES"].tolist(), tokenizer, model, args.pooling)

    train_emb_path = EMB_DIR / f"chemberta_only_train_{args.pooling}.npy"
    test_emb_path = EMB_DIR / f"chemberta_only_test_{args.pooling}.npy"

    np.save(train_emb_path, train_emb)
    np.save(test_emb_path, test_emb)

    print("\n저장 완료:")
    print(train_emb_path)
    print(test_emb_path)
    print("train_emb shape:", train_emb.shape)
    print("test_emb shape:", test_emb.shape)


if __name__ == "__main__":
    main()