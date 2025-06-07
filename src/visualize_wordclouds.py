import os
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
from PIL import Image
import numpy as np
from wordcloud import STOPWORDS
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

custom_stopwords = set(STOPWORDS)

# ==== CONFIGURATION ====
mask_circle = np.array(Image.open("circle.png"))
data_path = "data/articles_mapped.csv"
output_base = "wordclouds"
font_dir = "fonts"
font_path = os.path.join(font_dir, "AbhayaLibre-Regular.ttf")
os.makedirs(font_dir, exist_ok=True)

# ==== DOWNLOAD ROBOTO FONT IF NOT EXIST ====
def download_font(font_url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading font to {save_path}...")
        r = requests.get(font_url)
        with open(save_path, "wb") as f:
            f.write(r.content)

# ==== LOAD DATA ====
print("Loading data...")
df = pd.read_csv(data_path, parse_dates=["pub_date"])
df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce")
df["year"] = df["pub_date"].dt.year
df = df.dropna(subset=["text", "mapped_section", "year"])

def preprocess_text_for_freq(text, stopwords):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    return [w for w in tokens if w not in stopwords and len(w) > 2]

def get_top_global_words(df, stopwords, top_n=50):
    all_tokens = []
    for txt in tqdm(df["text"].dropna(), desc="Building global word frequency"):
        all_tokens.extend(preprocess_text_for_freq(txt, stopwords))
    freq = Counter(all_tokens)
    return set([word for word, _ in freq.most_common(top_n)])

# load global words if it exists
top_n = 50
def get_top_section_words(section_df, stopwords, top_n=100):
    all_tokens = []
    for txt in section_df["text"].dropna():
        all_tokens.extend(preprocess_text_for_freq(txt, stopwords))
    freq = Counter(all_tokens)
    return set([word for word, _ in freq.most_common(top_n)])

def preprocess_text(text, stopwords, top_global_words):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stopwords and w not in top_global_words and len(w) > 2]
    return " ".join(tokens)

# ==== GENERATE WORD CLOUD ====
def generate_wordcloud(text, font_path, stopwords, top_global_words):
    clean_text = preprocess_text(text, stopwords, top_global_words)
    wc = WordCloud(
        mask = mask_circle,
        width=2000,
        height=1000,
        background_color="white",
        font_path=font_path,
        max_words=50,
        collocations=False,
        colormap="viridis"
    ).generate(clean_text)
    return wc

# ==== MAIN FUNCTION ====
def create_wordclouds(selected_years, selected_sections):

    for section in tqdm(selected_sections, desc="Sections"):
        section_df = df[df["mapped_section"] == section]

        top_section_words = get_top_section_words(section_df, custom_stopwords, top_n=top_n)

        for year in tqdm(selected_years, desc=f"Years ({section})", leave=False):
            yearly_df = section_df[section_df["year"] == year]
            text_data = " ".join(yearly_df["text"].dropna().astype(str))

            if not text_data.strip():
                continue

            wc = generate_wordcloud(
                text_data,
                font_path,
                custom_stopwords,
                top_section_words
            )

            section_dir = os.path.join(output_base, section.replace(" ", "_"))
            os.makedirs(section_dir, exist_ok=True)
            out_path = os.path.join(section_dir, f"{year}.png")

            wc.to_file(out_path)

def create_wordclouds_range(selected_years, selected_sections):

    top_global_words = get_top_global_words(df, custom_stopwords, top_n=50)
    for section in tqdm(selected_sections, desc="Sections"):
        section_df = df[df["mapped_section"] == section]

        top_section_words = get_top_section_words(section_df, custom_stopwords, top_n=50)
        top_section_words = top_section_words.union(top_global_words)  

        for year in tqdm(selected_years, desc=f"Years ({section})", leave=False):
            year_start, year_end = year
            yearly_df = section_df[section_df["year"] >= year_start]
            yearly_df = yearly_df[yearly_df["year"] <= year_end]
            text_data = " ".join(yearly_df["text"].dropna().astype(str))

            if not text_data.strip():
                continue

            wc = generate_wordcloud(
                text_data,
                font_path,
                custom_stopwords,
                top_section_words 
            )
            output_base = "wordclouds"
            section_dir = os.path.join(output_base, section.replace(" ", "_"))
            os.makedirs(section_dir, exist_ok=True)
            out_path = os.path.join(section_dir, f"{year_start}_{year_end}.png")

            wc.to_file(out_path)

def get_distinctive_tfidf_words(yearly_docs, rest_docs, top_n=100, stopwords=None):
    """
    Compare TF-IDF scores in a year span vs. the rest of the data.
    Return words that are distinctive (high difference in scores).
    """
    try:
        vectorizer = TfidfVectorizer(
            stop_words=list(STOPWORDS),
            lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b"
        )
        combined_docs = yearly_docs + rest_docs
        tfidf_matrix = vectorizer.fit_transform(combined_docs)

        n_yearly = len(yearly_docs)
        feature_names = vectorizer.get_feature_names_out()

        # Split into yearly and rest
        tfidf_yearly = tfidf_matrix[:n_yearly, :].sum(axis=0).A1
        tfidf_rest = tfidf_matrix[n_yearly:, :].sum(axis=0).A1

        # Compute difference
        diff_scores = tfidf_yearly - tfidf_rest
        word_scores = list(zip(feature_names, diff_scores))
        word_scores = [(w, s) for w, s in word_scores if s > 0]
        sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)[:top_n]

        return dict(sorted_words)

    except Exception as e:
        print(f"[Error in distinctive TF-IDF] {e}")
        return {}


def get_top_tfidf_words(docs, top_n=100, stopwords=None):
    """
    Given a list of documents (strings), return a dictionary of the top N TF-IDF words and their scores.
    """
    if not docs:
        return {}

    docs = [doc for doc in docs if doc.strip()]
    if not docs:
        return {}

    try:
        vectorizer = TfidfVectorizer(
            stop_words=list(stopwords),
            max_features=None,
            lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b"
        )

        tfidf_matrix = vectorizer.fit_transform(docs)

        if tfidf_matrix.shape[1] == 0:
            print("[Warning] TF-IDF produced empty vocabulary for this time span.")
            return {}

        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        words = vectorizer.get_feature_names_out()
        word_scores = list(zip(words, tfidf_scores))
        sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)[:top_n]
        return dict(sorted_words)

    except ValueError as e:
        print(f"[Error] Skipping year span due to TF-IDF error: {e}")
        return {}


def create_wordclouds_range_all(selected_years):
    """
    For each (start_year, end_year) in selected_years, generate a TF-IDF-based word cloud.
    """
    output_base = "wordclouds"
    os.makedirs(output_base, exist_ok=True)
    top_global_words = get_top_global_words(df, custom_stopwords, top_n=20)
    cstopwords = custom_stopwords.union(top_global_words)

    for year_start, year_end in tqdm(selected_years, desc="Years", leave=False):
        yearly_df = df[(df["year"] >= year_start) & (df["year"] <= year_end)]
        texts = yearly_df["text"].dropna().astype(str).tolist()

        if not texts:
            continue

        top_words = get_top_tfidf_words(texts, top_n=100, stopwords=cstopwords)

        if not top_words:
            print(f"[Skip] No valid TF-IDF words for {year_start}-{year_end}")
            continue

        wc = WordCloud(
                mask = mask_circle,
                width=2000,
                height=1000,
                background_color="white",
                font_path=font_path,
                max_words=50,
                collocations=False,
                colormap="viridis"
            ).generate_from_frequencies(top_words)
        out_path = os.path.join(output_base, f"{year_start}_{year_end}.png")
        wc.to_file(out_path)

selected_sections = ["Business & Economy", "Politics & Government",
                     "International News", "Sports & Athletics",
                     "Health & Wellbeing", "Culture, Arts & Lifestyle",
                     "Tech, Science & Education", "Lifestyle & Leisure", "Other"]


selected_years = [(2015, 2016), (2017, 2019), (2020, 2022), (2023, 2025)]
create_wordclouds_range_all(selected_years)

