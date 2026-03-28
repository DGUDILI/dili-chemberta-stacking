import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
TRAIN_PATH = "../Data/chemberta_train.csv"
TEST_PATH = "../Data/chemberta_test.csv"
SAVE_DIR = "../Model/best_model_chemberta"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("full train shape:", train_df.shape)
print("test shape:", test_df.shape)

# train 내부에서 validation 분리
train_sub_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["Label"],
    random_state=42
)

print("train_sub shape:", train_sub_df.shape)
print("val shape:", val_df.shape)
print("test shape:", test_df.shape)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["SMILES"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_ds = Dataset.from_pandas(train_sub_df[["SMILES", "Label"]])
val_ds = Dataset.from_pandas(val_df[["SMILES", "Label"]])
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

training_args = TrainingArguments(
    output_dir="../Model/chemberta_runs",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
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

os.makedirs(SAVE_DIR, exist_ok=True)
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

pred = trainer.predict(test_ds)
probs = softmax(pred.predictions)
pred_labels = probs.argmax(axis=1)

result_df = test_df.copy()
result_df["chemberta_prob"] = probs[:, 1]
result_df["chemberta_pred"] = pred_labels
result_df.to_csv("../Data/chemberta_test_pred.csv", index=False)

acc = accuracy_score(test_df["Label"], pred_labels)
f1 = f1_score(test_df["Label"], pred_labels)
auc = roc_auc_score(test_df["Label"], probs[:, 1])

print("\n[ChemBERTa test result]")
print("Accuracy:", round(acc, 4))
print("F1:", round(f1, 4))
print("AUC:", round(auc, 4))
print("\n저장 완료:")
print("../Model/best_model_chemberta")
print("../Data/chemberta_test_pred.csv")