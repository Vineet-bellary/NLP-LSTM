import pandas as pd
import re
from pathlib import Path

from imdb_sentiment.config import LOGS_DIR, CLASS_MAP
from imdb_sentiment.util.logger import setup_logger

logger = setup_logger(LOGS_DIR / Path(__file__).stem, mode="w")


def clean_text(text: str) -> str:
    """Cleans the input text by removing special characters, extra spaces, urls, html tags, and converting to lowercase."""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes column names by converting to lowercase and replacing spaces with underscores."""
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    df_copy = df.copy()
    original_shape = df_copy.shape
    df_copy = normalize_columns(df_copy)

    raw_sample = df_copy.head()
    df_copy["review"] = df_copy["review"].astype(str).apply(clean_text)
    cleaned_sample = df_copy.head()

    df_copy.drop_duplicates(inplace=True)
    df_copy.reset_index(drop=True, inplace=True)
    df_copy["sentiment"] = df_copy["sentiment"].map(CLASS_MAP)

    report = [
        f"Sample Raw data:\n{raw_sample}",
        f"Original data shape: {original_shape}",
        "",
        f"Sample Cleaned data:\n{cleaned_sample}",
        f"Preprocessed data shape: {df_copy.shape}",
    ]
    return df_copy, "\n".join(report)


def preprocess_report(report: str, name: str) -> None:
    logger.info(f"\nPreprocessing Report for {name}:\n{report}")
