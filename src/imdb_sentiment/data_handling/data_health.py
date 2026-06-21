import pandas as pd
from pathlib import Path

from imdb_sentiment.config import DATA_DIR, LOGS_DIR
from imdb_sentiment.util.logger import setup_logger
from imdb_sentiment.data_handling.preprocessing import (
    preprocess_data,
    preprocess_report,
)

logger = setup_logger(LOGS_DIR / Path(__file__).stem, mode="w")


def get_health_report(df: pd.DataFrame, df_name: str) -> None:
    """Generates a cleanly formatted, cohesive string report of the dataset."""

    shape_info = f"{df.shape[0]} rows, {df.shape[1]} columns"
    missing_vals = df.isnull().sum().to_string()
    duplicates = df.duplicated().sum()

    label_col = df.columns[1]
    class_dist = df[label_col].value_counts().to_string()

    df_head = df.head().to_string()

    report = [
        f"\n{'='*30} {df_name.upper()} DATA HEALTH REPORT {'='*30}",
        f"Dimensions:     {shape_info}",
        f"Columns:        {df.columns.tolist()}",
        f"{'Duplicate Rows: ' + str(duplicates) if duplicates > 0 else 'No duplicate rows found'}",
        f"\n[ MISSING VALUES ]\n{missing_vals}",
        f"\n[ CLASS DISTRIBUTION ({label_col}) ]\n{class_dist}",
        f"\n[ DATA PREVIEW (FIRST 5 ROWS) ]\n{df_head}",
        f"{'='*80}\n",
    ]

    return "\n".join(report)


def main():
    data = pd.read_csv(DATA_DIR / "IMDB_dataset.csv")

    logger.info(get_health_report(data, "IMDB SENTIMENT"))

    preprocessed_data, preprocess_report_str = preprocess_data(data)
    logger.info(
        "Preprocessing complete. Generating data health report for preprocessed dataset..."
    )
    preprocess_report(preprocess_report_str, "IMDB SENTIMENT")

    logger.info(get_health_report(preprocessed_data, "IMDB SENTIMENT PREPROCESSED"))


if __name__ == "__main__":
    main()
