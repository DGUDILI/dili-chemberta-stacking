import os
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
TRAIN_PATH = "../Data/chemberta_train.csv"
TEST_PATH = "../Data/chemberta_test.csv"

OOF_SAVE_PATH = "../Data/chemberta_train_pred.csv"
TEST_SAVE_PATH = "../Data/chemberta_test_pred_oof.csv"

N_SPLITS = 5
MAX_LEN = 128
NUM_EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = softmax(logits)
    preds = probs.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "auc": roc_auc_score(labels, probs[:, 1]),
    }

set_seed(SEED)

train_df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)
test_df = pd.read_csv(TEST_PATH).reset_index(drop=True)

print("train shape:", train_df.shape)
print("test shape:", test_df.shape)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["SMILES"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_probs = np.zeros(len(train_df), dtype=float)
test_probs_folds = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, train_df["Label"]), start=1):
    print(f"\n========== Fold {fold}/{N_SPLITS} ==========")

    fold_train_df = train_df.iloc[tr_idx].reset_index(drop=True)
    fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

    print("fold_train:", fold_train_df.shape)
    print("fold_val:", fold_val_df.shape)

    train_ds = Dataset.from_pandas(fold_train_df[["SMILES", "Label"]])
    val_ds = Dataset.from_pandas(fold_val_df[["SMILES", "Label"]])
    test_ds = Dataset.from_pandas(test_df[["SMILES", "Label"]])

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    train_ds = train_ds.rename_column("Label", "labels")
    val_ds = val_ds.rename_column("Label", "labels")
    test_ds = test_ds.rename_column("Label", "labels")

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    fold_output_dir = f"../Model/chemberta_fold_{fold}"

    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # validation probability -> OOF
    val_pred = trainer.predict(val_ds)
    val_prob = softmax(val_pred.predictions)[:, 1]
    oof_probs[val_idx] = val_prob

    val_auc = roc_auc_score(fold_val_df["Label"], val_prob)
    print(f"Fold {fold} val AUC: {val_auc:.4f}")

    # test probability
    test_pred = trainer.predict(test_ds)
    test_prob = softmax(test_pred.predictions)[:, 1]
    test_probs_folds.append(test_prob)

# 전체 OOF 성능
oof_auc = roc_auc_score(train_df["Label"], oof_probs)
oof_pred_label = (oof_probs >= 0.5).astype(int)
oof_acc = accuracy_score(train_df["Label"], oof_pred_label)
oof_f1 = f1_score(train_df["Label"], oof_pred_label)

print("\n========== OOF RESULT ==========")
print("OOF Accuracy:", round(oof_acc, 4))
print("OOF F1:", round(oof_f1, 4))
print("OOF AUC:", round(oof_auc, 4))

# test 평균 확률
test_probs_mean = np.mean(np.vstack(test_probs_folds), axis=0)

test_auc = roc_auc_score(test_df["Label"], test_probs_mean)
test_pred_label = (test_probs_mean >= 0.5).astype(int)
test_acc = accuracy_score(test_df["Label"], test_pred_label)
test_f1 = f1_score(test_df["Label"], test_pred_label)

print("\n========== TEST RESULT (mean over folds) ==========")
print("Test Accuracy:", round(test_acc, 4))
print("Test F1:", round(test_f1, 4))
print("Test AUC:", round(test_auc, 4))

# 저장
oof_df = train_df.copy()
oof_df["chemberta_prob"] = oof_probs
oof_df.to_csv(OOF_SAVE_PATH, index=False)

test_out_df = test_df.copy()
test_out_df["chemberta_prob"] = test_probs_mean
test_out_df.to_csv(TEST_SAVE_PATH, index=False)

print("\n저장 완료:")
print(OOF_SAVE_PATH)
print(TEST_SAVE_PATH)