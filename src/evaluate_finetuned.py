import os
gpu_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    T5Tokenizer, T5ForConditionalGeneration
)
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import argparse

from transformers import AdamW
from torch.nn.functional import cross_entropy
from adaptimizer import Adaptimizer
from rmsprop import RMSProp
import time
import statistics
from t5seq import T5ForSequenceClassification
from transformers import T5Tokenizer

# === Argument parsing ===
parser = argparse.ArgumentParser(description="Stream evaluation of a model")
parser.add_argument("--model", "-m", type=str, default="bert", help="Model name")
parser.add_argument("--years", "-y", type=int, default=10, help="Number of years to evaluate on")
parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
parser.add_argument("--log_steps", "-l", type=int, default=5, help="Log every N steps")
parser.add_argument("--train", "-t", action="store_true", default=False, help="Train the model on the stream data")
parser.add_argument("--test_run", "-r", action="store_true", help="Run a test run with a small dataset")
parser.add_argument("--optimizer", "-o", type=str, default="adamw", help="Optimizer to use")
parser.add_argument("--model_dir", "-d", action="store_true", default=False, help="Loads the model version fine-tuned with adaptimizer")
parser.add_argument("--window_size", "-w", type=int, default=10, help="Window size for Adaptimizer")
parser.add_argument("--drift_metric", "-dm", type=str, default="l2", help="Drift metric to use for Adaptimizer")
parser.add_argument("--drift_detector", "-dd", action="store_true", default=False, help="Enable drift detection with Adaptimizer")
parser.add_argument("--drift_threshold", "-dt", type=float, default=1.5, help="Drift threshold for Adaptimizer")

# === Config ===
args = parser.parse_args()
model_name = args.model  # Options: 'deberta', 'roberta', 'bert'
model_dir = f"models/{model_name}_10yrs_test_run"
data_path = "data/articles_mapped.h5"
batch_size = args.batch_size
save_dir = f"results/stream/{model_name}_{args.years}yrs"
log_steps = args.log_steps
optimizer_name = args.optimizer
drift_threshold = args.drift_threshold

if args.model_dir:
    model_dir += "_adaptimizer"
    save_dir = f"results/stream/{model_name}_adaptimizer_{args.years}yrs"


# === Load model & tokenizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model_name == "t5":
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
label2id = model.config.label2id if hasattr(model.config, "label2id") else {}
id2label = {v: k for k, v in label2id.items()} if label2id else {}

model.eval()

# === Load data ===
df = pd.read_hdf(data_path)
df['year'] = pd.to_datetime(df['pub_date'], errors='coerce').dt.year
df = df.dropna(subset=['year', 'text', 'mapped_section'])
df['year'] = df['year'].astype(int)
if args.test_run:
    df = df.sample(2000, random_state=42)

# Split stream data: simulate "unseen future"
stream_data = df[df['year'] >= 2015].reset_index(drop=True)

if args.train:
    save_dir += f"_finetuned_bs_{batch_size}_ws_{args.window_size}"
    model.train()  # switch to training mode
    if optimizer_name == "adaptimizer":
        optimizer = Adaptimizer(model.parameters(), lr=2e-5, window_size=args.window_size)
        save_dir += "_adaptimizer"
    elif optimizer_name == "rmsprop":
        optimizer = RMSProp(model.parameters(), lr=2e-5)
        save_dir += "_rmsprop"
    else:
        optimizer = AdamW(model.parameters(), lr=2e-5)
    if args.drift_detector:
        save_dir += f'dm_{args.drift_metric}_dt_{args.drift_threshold}'
    os.makedirs(save_dir, exist_ok=True)

# === Stream evaluation ===
true_labels, pred_labels = [], []
acc_log = []
ds_log = []
ds = .0
ds_mean = .0
ds_variance = .0
start_time = time.time()
min_history = 10  # Minimum history for drift detection

for i in tqdm(range(0, len(stream_data), batch_size), desc="Streaming Evaluation"):
    batch_df = stream_data.iloc[i:i + batch_size]
    texts = batch_df['text'].tolist()
    labels = batch_df['mapped_section'].tolist()
    if optimizer_name == "adaptimizer":
        optimizer.apply_weights('slow')
    if model_name == "t5":
        # === Inference ===
        inputs = tokenizer([f"classify: {t}" for t in texts],
                        padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=4)
        pred_ids = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds = [id2label.get(int(pid), "other") if pid.isdigit() and int(pid) in id2label else "other" for pid in pred_ids]

        # === Fine-tuning ===
        if args.train:
            model.train()

            input_encodings = tokenizer([f"classify: {t}" for t in texts],
                                        padding=True, truncation=True, return_tensors="pt", max_length=512)
            target_labels = [str(label2id[l]) for l in labels]
            label_encodings = tokenizer(target_labels,
                                        padding=True, truncation=True, return_tensors="pt", max_length=4)

            input_encodings = {k: v.to(device) for k, v in input_encodings.items()}
            labels_input_ids = label_encodings["input_ids"].to(device)
            labels_input_ids[labels_input_ids == tokenizer.pad_token_id] = -100

            output = model(**input_encodings, labels=labels_input_ids)
            loss = output.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.eval()

    else:
        # === Inference ===
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        label_ids = torch.tensor([label2id[l] for l in labels]).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
        preds_ids = torch.argmax(logits, dim=1).cpu().tolist()
        preds = [id2label[p] for p in preds_ids]

        # === Fine-tuning ===
        if args.train:
            if optimizer_name == "adaptimizer":
                optimizer.restore_fast_weights()

            model.train()
            logits = model(**inputs).logits
            loss = cross_entropy(logits, label_ids)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.eval()

    # === Accumulate predictions ===
    true_labels.extend(labels)
    pred_labels.extend(preds)
    if len(true_labels) % log_steps == 0 or i + batch_size >= len(stream_data):
        acc = accuracy_score(true_labels, pred_labels)
        acc_log.append({"step": len(true_labels), "accuracy": acc})
        if optimizer_name == "adaptimizer":
            ds = optimizer.compute_drift_score(metric=args.drift_metric)
            ds_log.append({"step": len(true_labels), "drift_score": ds})
            if args.drift_detector and len(ds_log) >= min_history:
                ds_values = [item['drift_score'] for item in ds_log]
                ds_mean = statistics.mean(ds_values)
                if(ds > drift_threshold * ds_mean):
                    print(f"Drift detected at step {len(true_labels)}: Drift Score = {ds}, Mean = {ds_mean}, Variance = {ds_variance}")
                    optimizer.sync_slow_weights()
        print(f"Model {model_name} Optim {optimizer_name} BS {batch_size} Step {len(true_labels)} - Acc: {acc:.4f} - Drift Score: {ds}")

# === Save metrics & predictions ===
end_time = time.time()
total_runtime = end_time - start_time
pd.DataFrame(acc_log).to_csv(os.path.join(save_dir, "metrics.csv"), index=False)
pd.DataFrame({"prediction": pred_labels, "ground_truth": true_labels}).to_csv(
    os.path.join(save_dir, "predictions.csv"), index=False
)
if optimizer_name == "adaptimizer":
    pd.DataFrame(ds_log).to_csv(os.path.join(save_dir, "drift_scores.csv"), index=False)

# === Save classification report ===
with open(os.path.join(save_dir, "report.txt"), "w") as f:
    report = classification_report(true_labels, pred_labels, digits=6)
    print(report)
    f.write("Classification Report:\n")
    f.write(report + "\n\n")
    f.write(f"Total runtime (seconds): {total_runtime:.2f}\n")


