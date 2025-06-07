import os
import warnings
import argparse
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from transformers import (
    BertForSequenceClassification, BertTokenizer,
    DebertaForSequenceClassification, DebertaTokenizer,
    RobertaForSequenceClassification, RobertaTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    Trainer, TrainingArguments,
    AutoTokenizer, AutoModelForSequenceClassification
)

from t5seq import T5ForSequenceClassification  # Custom module

# ========== Setup ==========
gpu_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
warnings.filterwarnings("ignore", category=UserWarning)
print(f'Using GPU: {torch.cuda.get_device_name(0)}')
torch.cuda.empty_cache()

# ========== Argument parsing ==========
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

model_name = args.model
n_years = args.years
optimizer_name = args.optimizer

# ========== Paths ==========
data_path = "data/articles_mapped.h5"
model_save_path = f"models/{model_name}_{n_years}yrs"
if optimizer_name != "adamw":
    model_save_path += f"_{optimizer_name}"
if args.test_run:
    model_save_path += "_test_run"
os.makedirs(model_save_path, exist_ok=True)

# ========== Load and preprocess data ==========
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
    df_train['text'], df_train['mapped_section'],
    test_size=0.1, random_state=42, stratify=df_train['mapped_section']
)

label_list = sorted(df['mapped_section'].unique())
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}
label2id_str = {label: str(idx) for label, idx in label2id.items()}
id2label_str = {str(idx): label for idx, label in id2label.items()}

# ========== Load model and tokenizer ==========
def get_model_and_tokenizer(name):
    if name == "t5":
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
    else:
        ckpt = {
            "bert": "bert-base-uncased",
            "deberta": "microsoft/deberta-base",
            "roberta": "roberta-base",
            "distilbert": "distilbert-base-uncased",
            "electra": "google/electra-base-discriminator",
            "albert": "albert-base-v2"
        }[name]
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForSequenceClassification.from_pretrained(
            ckpt, num_labels=len(label2id), label2id=label2id, id2label=id2label
        )
    return model, tokenizer

model, tokenizer = get_model_and_tokenizer(model_name)

# ========== Tokenization ==========
def tokenize(batch, labels=None):
    if model_name == "t5":
        inputs = tokenizer([f"classify: {text}" for text in batch], padding='max_length',
                           truncation=True, max_length=512, return_tensors="pt")
        if labels is not None:
            label_strs = [label2id_str[label] for label in labels]
            label_inputs = tokenizer(label_strs, padding='max_length', truncation=True,
                                     max_length=4, return_tensors="pt")
            labels = label_inputs.input_ids
            labels[labels == tokenizer.pad_token_id] = -100
            inputs["labels"] = labels
        return inputs
    else:
        return tokenizer(batch, padding='max_length', truncation=True, max_length=512)

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenize(texts.tolist(), labels.tolist() if model_name == "t5" else None)
        self.labels = [label2id[label] for label in labels]

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if model_name != "t5":
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(X_train, y_train)
eval_dataset = NewsDataset(X_eval, y_eval)

# ========== Training ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

training_args = TrainingArguments(
    output_dir=model_save_path,
    num_train_epochs=args.epochs,
    fp16=args.fp16,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"{model_save_path}/logs",
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

train_result = trainer.train()
train_loss = train_result.training_loss

# ========== Evaluation ==========
results_file = os.path.join(model_save_path, "results.txt")
preds_file = os.path.join(model_save_path, "predictions.csv")

with open(results_file, "w") as f:
    if model_name == "t5":
        model.eval()
        decoded_preds = []
        loader = DataLoader(eval_dataset, batch_size=32)
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                gen_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                         max_length=4, num_beams=5, early_stopping=True)
                pred_strs = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                for p in pred_strs:
                    pred_clean = p.strip()
                    decoded_preds.append(id2label_str.get(pred_clean, "other"))
        y_true = [label for label in y_eval]
        report = classification_report(y_true, decoded_preds, labels=label_list, digits=6)
        print(report)
        f.write("Classification Report:\n" + report + "\n")
        pd.DataFrame({"prediction": decoded_preds, "ground_truth": y_true}).to_csv(preds_file, index=False)

    else:
        preds_output = trainer.predict(eval_dataset)
        preds = preds_output.predictions.argmax(axis=1)
        eval_loss = preds_output.metrics.get("test_loss") or preds_output.metrics.get("eval_loss", None)
        y_true = [label2id[label] for label in y_eval]
        pred_labels = [id2label[p] for p in preds]
        true_labels = [id2label[t] for t in y_true]
        report = classification_report(y_true, preds, target_names=label2id.keys())
        print(report)
        f.write("Classification Report:\n" + report + "\n")
        if eval_loss:
            f.write(f"Evaluation loss: {eval_loss:.4f}\n")
        if train_loss:
            f.write(f"Training loss: {train_loss:.4f}\n")
        pd.DataFrame({"prediction": pred_labels, "ground_truth": true_labels}).to_csv(preds_file, index=False)

# ========== Save ==========
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved at {model_save_path}")