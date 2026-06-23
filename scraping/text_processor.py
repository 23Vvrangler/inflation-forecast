"""
Text Processor - CommonCrawl Pipeline: HTML Cleaning and NLP Feature Extraction.

This module processes raw HTML files (simulating CommonCrawl WARC/WET extraction)
to produce clean text and NLP features for the inflation prediction model.

Pipeline stages (CommonCrawl analogy):
    1. HTML Cleaning (WARC -> WET extraction analogy)
       - Remove scripts, styles, navigation, ads, comments
       - Extract main content text
    2. Text Normalization
       - Lowercase, remove URLs, emails, numbers, special characters
       - Unicode normalization
    3. Tokenization (NLTK word_tokenize)
    4. Stopword Removal (English and Spanish)
    5. Lemmatization (optional, reduces dimensionality)
    6. Feature Extraction
       - TF-IDF vectorization (scikit-learn TfidfVectorizer)
       - Keyword frequency counts (inflation, crisis, recession, etc.)
       - Sentiment scoring (simple dictionary-based for MVP)
    7. Temporal Aggregation
       - Group articles by month
       - Compute mean TF-IDF, keyword counts, sentiment per month
       - Merge with economic time-series data

Input:
    data/external/raw_html/*.html (from news_scraper.py)
    data/external/scraping_metadata.json

Output:
    data/external/processed_text.csv (one row per article with features)
    data/external/monthly_text_features.csv (aggregated by month, ready for merge)

Usage:
    python -m scraping.text_processor

Dependencies:
    beautifulsoup4, nltk, scikit-learn, pandas, numpy
"""
import os
import sys
import json
import logging
import re
import string
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK imports (ensure nltk data is downloaded)
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Directories
EXTERNAL_DIR = Path("data/external")
RAW_HTML_DIR = EXTERNAL_DIR / "raw_html"
METADATA_FILE = EXTERNAL_DIR / "scraping_metadata.json"
PROCESSED_TEXT_FILE = EXTERNAL_DIR / "processed_text.csv"
MONTHLY_FEATURES_FILE = EXTERNAL_DIR / "monthly_text_features.csv"

# Ensure NLTK data is available (download if missing)
def ensure_nltk_data() -> None:
    """Download required NLTK corpora if not present."""
    required = ["punkt", "stopwords", "wordnet", "omw-1.4"]
    for item in required:
        try:
            nltk.data.find(f"tokenizers/{item}" if item == "punkt" else f"corpora/{item}")
        except LookupError:
            logger.info(f"Downloading NLTK data: {item}")
            nltk.download(item, quiet=True)

# Economic keywords for frequency counting (English + Spanish)
ECONOMIC_KEYWORDS = {
    "inflation": ["inflation", "inflacion", "inflación"],
    "crisis": ["crisis", "recession", "recesion", "recesión", "crash", "colapso"],
    "growth": ["growth", "crecimiento", "expansion", "expansión", "boom"],
    "interest": ["interest rate", "tasa de interes", "tasa de interés", "fed", "bce", "bcp"],
    "unemployment": ["unemployment", "desempleo", "paro", "jobs", "empleo"],
    "gdp": ["gdp", "pib", "producto interno", "gross domestic"],
    "currency": ["currency", "moneda", "dollar", "dolar", "dólar", "euro", "sol", "peso"],
    "market": ["market", "mercado", "stock", "bolsa", "bourse", "indices", "indice"],
}

# Simple sentiment word lists (English + Spanish)
SENTIMENT_POSITIVE = [
    "growth", "expansion", "boom", "recovery", "strong", "positive", "gain", "rise",
    "crecimiento", "expansión", "recuperación", "fuerte", "positivo", "ganancia", "aumento",
    "optimistic", "optimista", "bullish", "confident", "confianza",
]

SENTIMENT_NEGATIVE = [
    "crisis", "recession", "recesión", "crash", "collapse", "weak", "negative", "loss", "fall",
    "colapso", "débil", "negativo", "pérdida", "caída", "decline", "decrease",
    "pessimistic", "pesimista", "bearish", "fear", "miedo", "panic", "pánico",
    "inflation", "inflación", "stagflation", "hyperinflation", "hiperinflación",
]


def clean_html(html_content: str) -> str:
    """
    Remove HTML tags, scripts, styles, and extract plain text.

    This simulates the WET (WARC Encapsulated Text) extraction phase
    of the CommonCrawl pipeline.

    Args:
        html_content: Raw HTML string.

    Returns:
        Extracted plain text with minimal formatting.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        element.decompose()

    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.strip().startswith("<!--")):
        comment.extract()

    # Get text
    text = soup.get_text(separator=" ")

    # Collapse whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = " ".join(chunk for chunk in chunks if chunk)

    return text


def normalize_text(text: str) -> str:
    """
    Normalize text: lowercase, remove URLs, emails, numbers, special chars.

    Args:
        text: Raw text string.

    Returns:
        Normalized text string.
    """
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", text)
    text = re.sub(r"www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove numbers (optional, can keep for dates)
    text = re.sub(r"\d+", "", text)

    # Remove punctuation (keep spaces)
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove non-ASCII characters (or normalize)
    text = text.encode("ascii", "ignore").decode("ascii")

    return text


def tokenize_and_filter(text: str, language: str = "english") -> List[str]:
    """
    Tokenize text and remove stopwords.

    Args:
        text: Normalized text.
        language: Primary language for stopwords ("english" or "spanish").

    Returns:
        List of filtered tokens.
    """
    # Tokenize
    tokens = word_tokenize(text)

    # Get stopwords (combine English and Spanish for robustness)
    try:
        stop_words_en = set(stopwords.words("english"))
    except:
        stop_words_en = set()
    try:
        stop_words_es = set(stopwords.words("spanish"))
    except:
        stop_words_es = set()

    stop_words = stop_words_en.union(stop_words_es)

    # Filter: remove stopwords, short words, and non-alphabetic
    filtered = [
        token for token in tokens
        if token not in stop_words
        and len(token) > 2
        and token.isalpha()
    ]

    return filtered


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Lemmatize tokens to reduce dimensionality.

    Args:
        tokens: List of word tokens.

    Returns:
        List of lemmatized tokens.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def count_keywords(tokens: List[str]) -> Dict[str, int]:
    """
    Count occurrences of economic keywords in tokens.

    Args:
        tokens: List of word tokens.

    Returns:
        Dictionary mapping keyword category to count.
    """
    token_set = set(tokens)
    counts = {}

    for category, keywords in ECONOMIC_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in token_set)
        counts[f"keyword_{category}"] = count

    return counts


def compute_sentiment(tokens: List[str]) -> float:
    """
    Compute a simple sentiment score based on word lists.

    Score ranges from -1 (very negative) to +1 (very positive).
    0 indicates neutral sentiment.

    Args:
        tokens: List of word tokens.

    Returns:
        Sentiment score as float.
    """
    token_set = set(tokens)

    pos_count = sum(1 for word in SENTIMENT_POSITIVE if word in token_set)
    neg_count = sum(1 for word in SENTIMENT_NEGATIVE if word in token_set)

    total = pos_count + neg_count
    if total == 0:
        return 0.0

    score = (pos_count - neg_count) / total
    return score


def process_single_article(html_path: str, metadata: Dict) -> Optional[Dict]:
    """
    Process a single HTML file through the full text pipeline.

    Args:
        html_path: Path to the HTML file.
        metadata: Metadata dictionary for the article.

    Returns:
        Dictionary with processed text features, or None if failed.
    """
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    except Exception as e:
        logger.error(f"Failed to read {html_path}: {e}")
        return None

    # Stage 1: HTML cleaning (WET extraction)
    raw_text = clean_html(html_content)
    text_length = len(raw_text)

    if text_length < 100:
        logger.warning(f"Very short text ({text_length} chars) in {html_path}, skipping")
        return None

    # Stage 2: Normalization
    normalized = normalize_text(raw_text)

    # Stage 3: Tokenization and filtering
    tokens = tokenize_and_filter(normalized)
    token_count = len(tokens)

    if token_count < 10:
        logger.warning(f"Very few tokens ({token_count}) in {html_path}, skipping")
        return None

    # Stage 4: Lemmatization
    lemmas = lemmatize_tokens(tokens)

    # Stage 5: Keyword counts
    keyword_counts = count_keywords(lemmas)

    # Stage 6: Sentiment
    sentiment = compute_sentiment(lemmas)

    # Stage 7: Unique words and vocabulary richness
    unique_words = len(set(lemmas))
    vocabulary_richness = unique_words / token_count if token_count > 0 else 0

    # Build result
    result = {
        "html_path": html_path,
        "source": metadata.get("source", "unknown"),
        "url": metadata.get("url", ""),
        "published": metadata.get("published", ""),
        "title": metadata.get("title", ""),
        "text_length": text_length,
        "token_count": token_count,
        "unique_words": unique_words,
        "vocabulary_richness": round(vocabulary_richness, 4),
        "sentiment_score": round(sentiment, 4),
        "clean_text": " ".join(lemmas),
    }

    # Add keyword counts
    result.update({k: v for k, v in keyword_counts.items()})

    logger.info(
        f"Processed {Path(html_path).name}: {token_count} tokens, "
        f"sentiment={sentiment:.3f}, keywords={sum(keyword_counts.values())}"
    )

    return result


def process_all_articles() -> pd.DataFrame:
    """
    Process all HTML files in the raw_html directory.

    Returns:
        DataFrame with one row per processed article.
    """
    if not METADATA_FILE.exists():
        logger.error(f"Metadata file not found: {METADATA_FILE}")
        logger.error("Run news_scraper.py first.")
        return pd.DataFrame()

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    articles_meta = metadata.get("articles", [])
    fallback_meta = metadata.get("fallback_pages", [])
    all_records = articles_meta + fallback_meta

    if not all_records:
        logger.error("No records found in metadata file")
        return pd.DataFrame()

    logger.info(f"Processing {len(all_records)} HTML records...")

    processed = []
    for record in all_records:
        html_path = record.get("html_path")
        if not html_path or not Path(html_path).exists():
            logger.warning(f"HTML file not found: {html_path}")
            continue

        result = process_single_article(html_path, record)
        if result:
            processed.append(result)

    if not processed:
        logger.error("No articles could be processed")
        return pd.DataFrame()

    df = pd.DataFrame(processed)
    logger.info(f"Successfully processed {len(df)} articles")
    return df


def compute_tfidf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TF-IDF features for the corpus of cleaned texts.

    Args:
        df: DataFrame with a 'clean_text' column.

    Returns:
        DataFrame with TF-IDF mean score added as a column.
    """
    if df.empty or "clean_text" not in df.columns:
        logger.warning("No text data available for TF-IDF")
        df["tfidf_mean"] = 0.0
        return df

    # Filter out very short texts
    valid_mask = df["token_count"] >= 20
    valid_texts = df.loc[valid_mask, "clean_text"].tolist()

    if not valid_texts:
        logger.warning("No valid texts for TF-IDF computation")
        df["tfidf_mean"] = 0.0
        return df

    logger.info(f"Computing TF-IDF for {len(valid_texts)} documents...")

    vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 2),
    )

    tfidf_matrix = vectorizer.fit_transform(valid_texts)

    # Compute mean TF-IDF score per document
    mean_scores = np.array(tfidf_matrix.mean(axis=1)).flatten()

    # Assign back to dataframe (only for valid rows)
    df["tfidf_mean"] = 0.0
    df.loc[valid_mask, "tfidf_mean"] = mean_scores

    logger.info(f"TF-IDF computed. Feature matrix shape: {tfidf_matrix.shape}")
    logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    return df


def aggregate_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate text features by month to align with economic time series.

    Args:
        df: DataFrame with processed articles and 'published' datetime.

    Returns:
        DataFrame with one row per month, aggregated features.
    """
    if df.empty:
        logger.warning("No data to aggregate")
        return pd.DataFrame()

    # Parse published dates
    df["published_dt"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published_dt"])

    if df.empty:
        logger.warning("No valid dates for aggregation")
        return pd.DataFrame()

    # Create year-month column
    df["year_month"] = df["published_dt"].dt.to_period("M")

    # Aggregate numeric features by month
    agg_features = [
        "sentiment_score",
        "tfidf_mean",
        "vocabulary_richness",
        "keyword_inflation",
        "keyword_crisis",
        "keyword_growth",
        "keyword_interest",
        "keyword_unemployment",
        "keyword_gdp",
        "keyword_currency",
        "keyword_market",
    ]

    # Only aggregate columns that exist
    available_features = [f for f in agg_features if f in df.columns]

    monthly = df.groupby("year_month").agg({
        "sentiment_score": "mean",
        "tfidf_mean": "mean",
        "vocabulary_richness": "mean",
        **{f: "sum" for f in available_features if f not in ["sentiment_score", "tfidf_mean", "vocabulary_richness"]}
    }).reset_index()

    # Rename columns for clarity
    monthly.columns = ["year_month" if c == "year_month" else f"text_{c}" for c in monthly.columns]

    # Convert period to string for merging
    monthly["year_month"] = monthly["year_month"].astype(str)

    # Count articles per month
    article_counts = df.groupby("year_month").size().reset_index(name="text_article_count")
    article_counts["year_month"] = article_counts["year_month"].astype(str)

    monthly = monthly.merge(article_counts, on="year_month", how="left")

    logger.info(f"Monthly aggregation: {len(monthly)} months, {df['year_month'].nunique()} unique")
    return monthly


def save_outputs(df_articles: pd.DataFrame, df_monthly: pd.DataFrame) -> Tuple[str, str]:
    """
    Save processed outputs to CSV.

    Args:
        df_articles: DataFrame with per-article features.
        df_monthly: DataFrame with monthly aggregated features.

    Returns:
        Tuple of (articles_path, monthly_path).
    """
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    # Save per-article features
    df_articles.to_csv(PROCESSED_TEXT_FILE, index=False)
    logger.info(f"Saved per-article features: {PROCESSED_TEXT_FILE} ({len(df_articles)} rows)")

    # Save monthly aggregated features
    df_monthly.to_csv(MONTHLY_FEATURES_FILE, index=False)
    logger.info(f"Saved monthly features: {MONTHLY_FEATURES_FILE} ({len(df_monthly)} rows)")

    return str(PROCESSED_TEXT_FILE), str(MONTHLY_FEATURES_FILE)


def generate_summary(df_articles: pd.DataFrame, df_monthly: pd.DataFrame) -> None:
    """
    Print a summary of the text processing operation.

    Args:
        df_articles: Per-article DataFrame.
        df_monthly: Monthly aggregated DataFrame.
    """
    print("
" + "=" * 60)
    print("TEXT PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Articles processed:         {len(df_articles)}")
    print(f"Total tokens (all articles): {df_articles['token_count'].sum()}")
    print(f"Avg tokens per article:     {df_articles['token_count'].mean():.1f}")
    print(f"Avg sentiment score:          {df_articles['sentiment_score'].mean():.4f}")
    print(f"Monthly aggregations:       {len(df_monthly)}")

    if "text_article_count" in df_monthly.columns:
        print(f"Avg articles per month:     {df_monthly['text_article_count'].mean():.1f}")

    print("
Keyword totals (all articles):")
    for col in df_articles.columns:
        if col.startswith("keyword_"):
            total = df_articles[col].sum()
            print(f"  {col}: {total}")

    print("
" + "=" * 60)
    print("Output files:")
    print(f"  {PROCESSED_TEXT_FILE}")
    print(f"  {MONTHLY_FEATURES_FILE}")
    print("=" * 60)
    print("
Next step: Merge with economic data in feature engineering.")


def main():
    """
    Main execution entry point.

    Processes all raw HTML files through the CommonCrawl-inspired pipeline:
    HTML cleaning -> text normalization -> tokenization -> NLP feature extraction
    -> TF-IDF -> monthly aggregation.
    """
    logger.info("=" * 60)
    logger.info("TEXT PROCESSOR - COMMONCRAWL PIPELINE")
    logger.info("=" * 60)
    logger.info("Stages: HTML cleaning -> Normalization -> Tokenization ->")
    logger.info("        Lemmatization -> Keywords -> Sentiment -> TF-IDF ->")
    logger.info("        Monthly aggregation")
    logger.info("=" * 60)

    # Ensure NLTK data is available
    ensure_nltk_data()

    # Stage 1: Process all HTML files
    logger.info("
Stage 1: Processing HTML files")
    df_articles = process_all_articles()

    if df_articles.empty:
        logger.error("No articles to process. Exiting.")
        sys.exit(1)

    # Stage 2: Compute TF-IDF features
    logger.info("
Stage 2: Computing TF-IDF features")
    df_articles = compute_tfidf_features(df_articles)

    # Stage 3: Aggregate by month
    logger.info("
Stage 3: Aggregating features by month")
    df_monthly = aggregate_monthly_features(df_articles)

    # Stage 4: Save outputs
    logger.info("
Stage 4: Saving processed outputs")
    save_outputs(df_articles, df_monthly)

    # Stage 5: Summary
    generate_summary(df_articles, df_monthly)

    logger.info("
" + "=" * 60)
    logger.info("TEXT PROCESSING COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
