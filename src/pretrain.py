import os
import warnings
import argparse
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from transformers import (
    BertConfig, BertForSequenceClassification, BertTokenizer,
    DebertaConfig, DebertaForSequenceClassification, DebertaTokenizer,
    RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
    DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer,
    AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer,
    T5Config, T5ForConditionalGeneration, T5Tokenizer,
    Trainer, TrainingArguments
)

# ========== SETUP ==========
gpu_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
warnings.filterwarnings("ignore", category=UserWarning)
print(f'Using GPU: {torch.cuda.get_device_name(0)}')
torch.cuda.empty_cache()

# ========== ARGUMENTS ==========
parser = argparse.ArgumentParser(description="Fine-tune a model")
parser.add_argument("--model", "-m", type=str, default="roberta")
parser.add_argument("--years", "-y", type=int, default=10)
parser.add_argument("--batch_size", "-b", type=int, default=32)
parser.add_argument("--epochs", "-e", type=int, default=3)
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--train_size", "-s", type=int, default=-1)
parser.add_argument("--test_run", "-r", action="store_true")
parser.add_argument("--optimizer", "-o", type=str, default="adamw")
args = parser.parse_args()

# ========== CONFIG ==========
model_name = args.model
n_years = args.years
optimizer_name = args.optimizer

data_path = "data/articles_mapped.h5"
model_dir = f"models/SCRATCH_{model_name}_{n_years}yrs"
if optimizer_name != "adamw":
    model_dir += f"_{optimizer_name}"
if args.test_run:
    model_dir += "_test_run"
os.makedirs(model_dir, exist_ok=True)

# ========== DATA ==========
df = pd.read_hdf(data_path)
if args.test_run:
    df = df.sample(2000, random_state=42)
if args.train_size > 0:
    df = df.groupby('mapped_section').apply(lambda x: x.sample(args.train_size)).reset_index(drop=True)

if 'year' not in df.columns:
    df['year'] = pd.to_datetime(df['pub_date'], errors='coerce').dt.year
    df.dropna(subset=['year'], inplace=True)
    df['year'] = df['year'].astype(int)

recent_years = sorted(df['year'].unique())[-n_years:]
df_train = df[df['year'].isin(recent_years)].dropna(subset=['text', 'mapped_section'])

X_train, X_eval, y_train, y_eval = train_test_split(
    df_train['text'], df_train['mapped_section'], test_size=0.1,
    random_state=42, stratify=df_train['mapped_section']
)

label2id = {label: idx for idx, label in enumerate(sorted(df['mapped_section'].unique()))}
id2label = {idx: label for label, idx in label2id.items()}

model_ckpt = {
    "bert": "bert-base-uncased",
    "deberta": "microsoft/deberta-base",
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "electra": "google/electra-base-discriminator",
    "albert": "albert-base-v2"
}

# ========== MODEL & TOKENIZER ==========
def get_model_and_tokenizer(name):
    if name == "t5":
        config = T5Config()
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = T5ForConditionalGeneration(config)
    else:
        config = {
            "bert": BertConfig,
            "deberta": DebertaConfig,
            "roberta": RobertaConfig,
            "distilbert": DistilBertConfig,
            "albert": AlbertConfig
        }[name](num_labels=len(label2id), label2id=label2id, id2label=id2label)

        tokenizer = {
            "bert": BertTokenizer,
            "deberta": DebertaTokenizer,
            "roberta": RobertaTokenizer,
            "distilbert": DistilBertTokenizer,
            "albert": AlbertTokenizer
        }[name].from_pretrained(model_ckpt[name])

        model = {
            "bert": BertForSequenceClassification,
            "deberta": DebertaForSequenceClassification,
            "roberta": RobertaForSequenceClassification,
            "distilbert": DistilBertForSequenceClassification,
            "albert": AlbertForSequenceClassification
        }[name](config)
    return model, tokenizer

model, tokenizer = get_model_and_tokenizer(model_name)

# ========== TOKENIZATION ==========
def tokenize(batch, labels=None):
    inputs = tokenizer(batch, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    if model_name == "t5" and labels is not None:
        label_ids = tokenizer(labels, padding='max_length', truncation=True, max_length=16, return_tensors="pt").input_ids
        label_ids[label_ids == tokenizer.pad_token_id] = -100
        inputs["labels"] = label_ids
    return inputs

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenize(texts.tolist(), labels.tolist() if model_name == "t5" else None)
        self.labels = [label2id[label] for label in labels]

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if model_name != "t5":
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(X_train, y_train)
eval_dataset = NewsDataset(X_eval, y_eval)

# ========== TRAINING ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

training_args = TrainingArguments(
    output_dir=model_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"{model_dir}/logs",
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=args.fp16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

train_result = trainer.train()
train_loss = train_result.training_loss

# ========== EVALUATION ==========
results_file = os.path.join(model_dir, "results.txt")
preds_file = os.path.join(model_dir, "predictions.csv")

with open(results_file, "w") as f:
    if model_name == "t5":
        model.eval()
        decoded_preds = []
        loader = DataLoader(eval_dataset, batch_size=32)
        with torch.no_grad():
            for batch in loader:
                inputs = batch['input_ids'].to(device)
                masks = batch['attention_mask'].to(device)
                outputs = model.generate(input_ids=inputs, attention_mask=masks, max_length=16)
                preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                decoded_preds.extend([p.strip().lower() for p in preds])
        y_true = [label.strip().lower() for label in y_eval]
        report = classification_report(y_true, decoded_preds, labels=sorted(set(y_true)))
        f.write("Classification Report:\n" + report + "\n")
        pd.DataFrame({"prediction": decoded_preds, "ground_truth": y_true}).to_csv(preds_file, index=False)

    else:
        preds_output = trainer.predict(eval_dataset)
        preds = preds_output.predictions.argmax(axis=1)
        eval_loss = preds_output.metrics.get("test_loss") or preds_output.metrics.get("eval_loss", None)
        y_true_ids = [label2id[label] for label in y_eval]
        id2label = {v: k for k, v in label2id.items()}
        pred_labels = [id2label[p] for p in preds]
        true_labels = [id2label[t] for t in y_true_ids]
        report = classification_report(y_true_ids, preds, target_names=label2id.keys())
        f.write("Classification Report:\n" + report + "\n")
        if eval_loss:
            f.write(f"Evaluation loss: {eval_loss:.4f}\n")
        if train_loss:
            f.write(f"Training loss: {train_loss:.4f}\n")
        pd.DataFrame({"prediction": pred_labels, "ground_truth": true_labels}).to_csv(preds_file, index=False)

# ========== SAVE ==========
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)
print(f"Model saved at {model_dir}")