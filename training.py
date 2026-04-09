# !pip install -q -U transformers==4.47.0 accelerate==1.2.1 datasets scikit-learn matplotlib seaborn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
)

MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
EPOCHS = 3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.05
OUTPUT_DIR = "./model"
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_NAMES = ["NFS", "UFS", "CFS"]


gt = pd.read_csv("groundtruth.csv")
cs = pd.read_csv("crowdsourced.csv")
cs = cs[cs.File_id != "2016-10-09.txt"]  # reserve for showcase
df = pd.concat([gt, cs], ignore_index=True)
df = df.drop_duplicates(subset="Sentence_id", keep="first")

df["label"] = df["Verdict"].map(LABEL_MAP)
df = df[["Text", "label"]].rename(columns={"Text": "text"})

print(f"Total samples after merge+dedup: {len(df)}")
for i, name in enumerate(LABEL_NAMES):
    n = (df["label"] == i).sum()
    print(f"  {name}: {n} ({n / len(df) * 100:.1f}%)")

train_df, test_df = train_test_split(
    df, test_size=0.15, stratify=df["label"], random_state=42
)
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

print(f"Train: {len(train_dataset)}  |  Test: {len(test_dataset)}")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)


train_dataset = train_dataset.map(tokenize_fn, batched=True)
test_dataset = test_dataset.map(tokenize_fn, batched=True)


train_labels = np.array(train_df["label"])
raw_weights = compute_class_weight(
    "balanced", classes=np.unique(train_labels), y=train_labels
)
class_weights = torch.FloatTensor(raw_weights).to("cuda")

for name, w in zip(LABEL_NAMES, raw_weights):
    print(f"  {name} weight: {w:.4f}")


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=class_weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.config.label2id = {name: i for i, name in enumerate(LABEL_NAMES)}
model.config.id2label = {i: name for i, name in enumerate(LABEL_NAMES)}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    logging_steps=50,
    fp16=True,
    report_to="none",
)

collator = DataCollatorWithPadding(tokenizer)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)
trainer.train()


preds_output = trainer.predict(test_dataset)
y_pred = preds_output.predictions.argmax(axis=-1)
y_true = preds_output.label_ids

print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))

history = trainer.state.log_history
train_logs = [h for h in history if "loss" in h and "eval_loss" not in h]
eval_logs = [h for h in history if "eval_loss" in h]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(
    [h["step"] for h in train_logs], [h["loss"] for h in train_logs], label="train"
)
ax1.plot(
    [h["step"] for h in eval_logs],
    [h["eval_loss"] for h in eval_logs],
    label="eval",
    marker="o",
)
ax1.set(title="Loss", xlabel="Step", ylabel="Loss")
ax1.legend()

ax2.plot(
    [h["epoch"] for h in eval_logs],
    [h["eval_f1_weighted"] for h in eval_logs],
    label="weighted",
    marker="o",
)
ax2.plot(
    [h["epoch"] for h in eval_logs],
    [h["eval_f1_macro"] for h in eval_logs],
    label="macro",
    marker="s",
)
ax2.set(title="F1", xlabel="Epoch", ylabel="F1")
ax2.legend()

# plt.tight_layout()
plt.show()


model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done")
