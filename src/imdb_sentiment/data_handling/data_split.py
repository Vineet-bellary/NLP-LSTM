import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify_col: str = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the input DataFrame into training, validation and testing sets."""

    stratify = df[stratify_col] if stratify_col else None

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )

    adjusted_val_size = val_size / (1 - test_size)
    stratify_tmp = train_df[stratify_col] if stratify_col else None

    train_df, val_df = train_test_split(
        train_df,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=stratify_tmp,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df
