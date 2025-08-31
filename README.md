# LLM-OFA
This repository contains the code and data for our CIKM 2025 paper: LLM-OFA: On-the-Fly Adaptation of Large Language Models to Address Temporal Drift Across Two Decades of News

## 📁 Project Structure

```
├── src/
│   ├── map_section.py              # Map sections from the original data into 9 broader categories
│   ├── pretrain.py                 # Create pretrain LLMs from HuggingFace and fine-tune them on 1% of data
│   ├── finetune.py                 # Create fine-tune LLMs by training them on 10 years of data (2005-2015)
│   ├── evaluate_finetuned.py       # On-the-fly evaluation of Fine-tuned LLMs (with online fine-tuning)
│   ├── evaluate_stream.py          # On-the-fly evaluation of Pretrained LLMs (with online fine-tuning)
│   ├── visualize_wordclouds.py     # Yearly & section-based word cloud generation
│   ├── adaptimizer.py              # Our proposed optimizer for online fine-tuning of LLMs
│   ├── plot_drift_and_accuracy.py  # Plot prequential accuracy with drift score over time
├── data/                           # Data files
├── models/                         # Saved model checkpoints
├── results/                        # Evaluation logs and predictions
├── wordclouds/                     # Generated word clouds
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/pouyaghahramanian/LLM-OFA
cd LLM-OFA

# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

Download the dataset from this private link:
[Download articles.h5 from Figshare](https://figshare.com/s/744c6b067b2818009688?file=55195175)

Place it in the `data/` folder.

## 🚀 Usage

### Map Sections

```bash
python src/map_section.py
```

### Create Pretrained LLMs

```bash
python src/pretrain.py --model < model_name > --years 10 --batch_size 32
```

### Create a Fine-tuned Model

```bash
python src/finetune.py --model < model_name > --years 10 --batch_size 32
```

### Evaluate a Pretrained Model in Static Settings

```bash
python src/evaluate_pretrained.py --model < model_name >
```

### Evaluate a Fine-tuned Model in Static Settings

```bash
python src/evaluate_finetuned.py --model < model_name >
```

### Evaluate a Pretrained Model in OFA Settings

```bash
python src/evaluate_pretrained.py --model < model_name > --train
```

### Evaluate a Fine-tuned Model in OFA Settings

```bash
python src/evaluate_finetuned.py --model < model_name > --train
```

### Evaluate a Pretrained Model in OFA Settings with Adaptimizer

```bash
python src/evaluate_pretrained.py --model < model_name > --train --optimizer adaptimizer
```

### Evaluate a Fine-tuned Model in OFA Settings with Adaptimizer

```bash
python src/evaluate_finetuned.py --model < model_name > --train --optimizer adaptimizer
```

### Generate Word Clouds

```bash
python src/visualize_wordclouds.py
```

## ⚡ Available Models

Use one of the following with `--model`:
- `bert`
- `roberta`
- `deberta`
- `albert`
- `distilbert`

## ✍️ Notes

- 1M-News data must be downloaded and stored in `data/articles.h5`
- `map_section.py` must be run before running experiments to generate `data/articles_mapped.h5` data.

## 📚 Citation

If you use this repository, please cite our paper:

```bibtex
@inproceedings{ghahramanian2025ofa,
  author    = {Pouya Ghahramanian and Sepehr Bakhshi and Fazli Can},
  title     = {LLM-OFA: On-the-Fly Adaptation of Large Language Models to Address Temporal Drift Across Two Decades of News},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25)},
  year      = {2025},
  publisher = {ACM},
  doi       = {10.1145/3746252.3760846}
}

## 📄 License

MIT License
