import os
gpu_id = 1
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

from transformers import (
    BertConfig, BertForSequenceClassification, BertTokenizer,
    DebertaConfig, DebertaForSequenceClassification, DebertaTokenizer,
    RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
    DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer,
    AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer,
    T5Config, T5ForConditionalGeneration, T5Tokenizer,
    ElectraConfig, ElectraTokenizer, ElectraForSequenceClassification,
    Trainer, TrainingArguments
)

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
parser.add_argument("--drift_metric", "-dm", type=str, default="cosine", help="Drift metric to use for Adaptimizer")
parser.add_argument("--drift_detector", "-dd", action="store_true", default=False, help="Enable drift detection with Adaptimizer")
parser.add_argument("--drift_threshold", "-dt", type=float, default=2, help="Drift threshold for Adaptimizer")
parser.add_argument("--pretrained_model", "-p", action="store_true", default=False, help="Use pretrained models")
parser.add_argument("--partial_train", "-pt", action="store_true", default=False, help="Partially train the model on a small portion of the data")

# === Config ===
args = parser.parse_args()
model_name = args.model  # Options: 'deberta', 'roberta', 'bert'
model_dir = f"models/{model_name}_{str(args.years)}yrs"
data_path = "data/articles_mapped.h5"
batch_size = args.batch_size  # for true stream evaluation
save_dir = ""
if args.pretrained_model:
    save_dir = f"results/stream/PRETRAIN_{model_name}_{args.years}yrs"
else:
    save_dir = f"results/stream/SCRATCH_{model_name}_{args.years}yrs"
log_steps = args.log_steps
optimizer_name = args.optimizer
drift_threshold = args.drift_threshold

if args.model_dir:
    model_dir += "_adaptimizer"
    save_dir = f"results/stream/{model_name}_adaptimizer_{args.years}yrs"

# === Load model & tokenizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load data ===
df = pd.read_hdf(data_path)
df['year'] = pd.to_datetime(df['pub_date'], errors='coerce').dt.year
df = df.dropna(subset=['year', 'text', 'mapped_section'])
df['year'] = df['year'].astype(int)
if args.test_run:
    df = df.sample(20000, random_state=42)

# Split stream data: simulate "unseen future"
years_filter = 2025 - args.years
stream_data = df[df['year'] >= years_filter].reset_index(drop=True)

label2id = {label: idx for idx, label in enumerate(sorted(df['mapped_section'].unique()))}
id2label = {idx: label for label, idx in label2id.items()}

def get_model_and_tokenizer_pretrained(name):
    if name == "t5":
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
    else:
        model_ckpt = {
            "bert": "bert-base-uncased",
            "deberta": "microsoft/deberta-base",
            "roberta": "roberta-base",
            "distilbert": "distilbert-base-uncased",
            "electra": "google/electra-base-discriminator",
            "albert": "albert-base-v2"
        }[name]
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_ckpt,
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label
        )
    return model.to(device), tokenizer


from transformers import (
    T5Config, T5Tokenizer, T5ForConditionalGeneration,
    BertConfig, BertTokenizer, BertForSequenceClassification,
    DebertaConfig, DebertaTokenizer, DebertaForSequenceClassification,
    RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification,
    DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification,
    AlbertConfig, AlbertTokenizer, AlbertForSequenceClassification,
    ElectraConfig, ElectraTokenizer, ElectraForSequenceClassification
)

def get_model_and_tokenizer_scratch(name):
    if name == "t5":
        config = T5Config()
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = T5ForConditionalGeneration(config)
    else:
        model_ckpt = {
            "bert": "bert-base-uncased",
            "deberta": "microsoft/deberta-base",
            "roberta": "roberta-base",
            "distilbert": "distilbert-base-uncased",
            "electra": "google/electra-base-discriminator",
            "albert": "albert-base-v2"
        }
        config_class = {
            "bert": BertConfig,
            "deberta": DebertaConfig,
            "roberta": RobertaConfig,
            "distilbert": DistilBertConfig,
            "electra": ElectraConfig,
            "albert": AlbertConfig
        }[name]
        model_class = {
            "bert": BertForSequenceClassification,
            "deberta": DebertaForSequenceClassification,
            "roberta": RobertaForSequenceClassification,
            "distilbert": DistilBertForSequenceClassification,
            "electra": ElectraForSequenceClassification,
            "albert": AlbertForSequenceClassification
        }[name]
        tokenizer_class = {
            "bert": BertTokenizer,
            "deberta": DebertaTokenizer,
            "roberta": RobertaTokenizer,
            "distilbert": DistilBertTokenizer,
            "electra": ElectraTokenizer,
            "albert": AlbertTokenizer
        }[name]
        config = config_class(num_labels=len(label2id), label2id=label2id, id2label=id2label)
        tokenizer = tokenizer_class.from_pretrained(model_ckpt[name])
        model = model_class(config)
    return model.to(device), tokenizer

if args.pretrained_model:
    model, tokenizer = get_model_and_tokenizer_pretrained(model_name)
else:
    model, tokenizer = get_model_and_tokenizer_scratch(model_name)

model.eval()

if args.train:
    save_dir += f"_finetuned_bs_{batch_size}_ws_{args.window_size}"
    model.train() 
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
min_history = 10 

if args.partial_train:
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    # For partial training, we only use a small portion of the data
    train_data = stream_data.sample(frac=0.01, random_state=42).reset_index(drop=True)
    stream_data = stream_data.drop(train_data.index).reset_index(drop=True)
    print(f"Partial training on {len(train_data)} samples, remaining stream data: {len(stream_data)} samples")
    batch_size_train = args.batch_size // 4
    for i in tqdm(range(0, len(train_data), batch_size_train), desc="Partial Training"):
        batch_df = train_data.iloc[i:i + batch_size_train]
        texts = batch_df['text'].tolist()
        labels = batch_df['mapped_section'].tolist()
        # Preprocess input
        if model_name == "t5":
            inputs = tokenizer([f"classify: {t}" for t in texts], padding=True, truncation=True, return_tensors="pt", max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=16)
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            preds = [p.strip().lower() for p in preds]
            labels = [l.strip().lower() for l in labels]
        else:
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            label_ids = torch.tensor([label2id[l] for l in labels]).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            if id2label:
                preds = [id2label[p] for p in preds]
                labels = [l for l in labels]

        # Train the model
        model.zero_grad() 
        label_ids = torch.tensor([label2id[l] for l in labels]).to(device)
        logits = model(**inputs).logits
        loss = cross_entropy(logits, label_ids)
        loss.backward()
        optimizer.step()
model.eval()

for i in tqdm(range(0, len(stream_data), batch_size), desc="Streaming Evaluation"):
    batch_df = stream_data.iloc[i:i + batch_size]
    texts = batch_df['text'].tolist()
    labels = batch_df['mapped_section'].tolist()
    if optimizer_name == "adaptimizer":
        optimizer.apply_weights('slow')
    # Preprocess input
    if model_name == "t5":
        inputs = tokenizer([f"classify: {t}" for t in texts], padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=16)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds = [p.strip().lower() for p in preds]
        labels = [l.strip().lower() for l in labels]
    else:
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        label_ids = torch.tensor([label2id[l] for l in labels]).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        if id2label:
            preds = [id2label[p] for p in preds]
            labels = [l for l in labels]

        # === Train if enabled ===
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
            if optimizer_name == "adaptimizer":
                try:
                    ds = optimizer.compute_drift_score(metric=args.drift_metric)
                except Exception as e:
                    ds = 0.0
                ds_log.append({"step": len(true_labels), "drift_score": ds})
                if args.drift_detector and len(ds_log) >= min_history:
                    ds_values = [item['drift_score'] for item in ds_log]
                    ds_mean = statistics.mean(ds_values)
                    if(ds > drift_threshold * ds_mean):
                        tqdm.write(f"Drift detected at step {len(true_labels)}: Drift Score = {ds}, Mean = {ds_mean}, Variance = {ds_variance}")
                        optimizer.sync_slow_weights()

    true_labels.extend(labels)
    pred_labels.extend(preds)

    if len(true_labels) % log_steps == 0 or i + batch_size >= len(stream_data):
        acc = accuracy_score(true_labels, pred_labels)
        acc_log.append({"step": len(true_labels), "accuracy": acc})
        tqdm.write(f"Model {model_name} Optim {optimizer_name} BS {batch_size} Step {len(true_labels)} - Acc: {acc:.4f} - Drift Score: {ds}")

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


