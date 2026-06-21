import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from imdb_sentiment.config import (
    LOGS_DIR,
    MAX_VOCAB_SIZE,
    MAX_SEQ_LEN,
    BATCH_SIZE,
    NUM_CLASSES,
)
from imdb_sentiment.util.logger import setup_logger
from imdb_sentiment.data_handling.preprocessing import preprocess_data, clean_text

# from imdb_sentiment.model import lstm

logger = setup_logger(LOGS_DIR / Path(__file__).stem, mode="w")

"""
    1. Build vocabulary
    2. Tokenize Docs
    3. Vectorize Docs
    4. Pad vectorized docs
    5. Create DataLoader object
"""


def build_vocab(df: pd.DataFrame, text_col: str, max_vocab_size: int) -> set[str]:
    """Builds a vocabulary set from the specified text column in the DataFrame."""
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }
    for text in df[text_col]:
        if len(vocab) >= max_vocab_size:
            break
        for word in str(text).split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def tokenize_docs(df: pd.DataFrame, text_col: str) -> list[list[str]]:
    """Tokenizes the specified text column in the DataFrame into a list of token lists."""
    return [str(text).split() for text in df[text_col]]


def vectorize_docs(
    tokenized_docs: list[list[str]], vocab: set[str]
) -> tuple[list[list[int]], int]:
    """Converts tokenized documents into lists of indices based on the provided vocabulary."""
    vectorized_docs = [
        [vocab.get(token, vocab["<UNK>"]) for token in doc] for doc in tokenized_docs
    ]

    max_doc_length = (
        MAX_SEQ_LEN
        if MAX_SEQ_LEN is not None
        else max(len(doc) for doc in vectorized_docs)
    )

    return vectorized_docs, max_doc_length


def pad_vectorized_docs(
    vectorized_docs: list[list[int]], max_doc_length: int
) -> list[list[int]]:
    """Pads vectorized documents to a specified maximum document length."""
    result = []
    for doc in vectorized_docs:
        if len(doc) >= max_doc_length:
            result.append(doc[:max_doc_length])
        else:
            result.append(doc + [0] * (max_doc_length - len(doc)))
    return result


def create_dataloader(x: torch.Tensor, y: torch.Tensor, batch_size: int):
    """Creates a DataLoader object from vectorized documents and their corresponding labels."""
    dataset = torch.utils.data.TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def prepare_data(
    df: pd.DataFrame, vocab: set[str] = None, doc_length: int = None, name: str = "data"
) -> tuple[DataLoader, set[str], int]:

    logger.info(f"{name.capitalize()} data loaded with shape: {df.shape}")

    preprocessed_df = df
    min_class_index = int(preprocessed_df["sentiment"].min())
    max_class_index = int(preprocessed_df["sentiment"].max())

    assert (
        min_class_index >= 0 and max_class_index < NUM_CLASSES
    ), f"label range invalid: {min_class_index}..{max_class_index}"

    logger.info(
        f"Class index range: {min_class_index} to {max_class_index} (total classes: {NUM_CLASSES})"
    )

    if vocab is None:
        vocab = build_vocab(
            preprocessed_df, text_col="review", max_vocab_size=MAX_VOCAB_SIZE
        )
    logger.info(f"Vocabulary size: {len(vocab)}")

    tokenized_docs = tokenize_docs(preprocessed_df, text_col="review")
    vectorized_docs, max_doc_length = vectorize_docs(tokenized_docs, vocab)

    # Testing
    # lengths = [len(doc) for doc in vectorized_docs]

    # print(min(lengths))
    # print(max(lengths))
    # print(sum(lengths) / len(lengths))

    if doc_length is not None:
        max_doc_length = doc_length

    padded_docs = pad_vectorized_docs(vectorized_docs, max_doc_length=max_doc_length)

    label_tensor = torch.tensor(preprocessed_df["sentiment"].values, dtype=torch.long)
    vectorized_tensor = torch.tensor(padded_docs)

    dataloader = create_dataloader(
        x=vectorized_tensor, y=label_tensor, batch_size=BATCH_SIZE
    )

    return dataloader, vocab, max_doc_length


def prepare_sample_text(
    sample_text: str, vocab: set[str], max_doc_length: int
) -> torch.Tensor:
    """Prepares a sample text for inference by tokenizing, vectorizing, and padding it."""
    cleaned_text = clean_text(sample_text)
    tokenized_text = cleaned_text.split()
    vectorized_doc, _ = vectorize_docs([tokenized_text], vocab)
    vectorized_text = vectorized_doc[0]
    padded_vectorized_text = pad_vectorized_docs([vectorized_text], max_doc_length)[0]

    return torch.tensor(padded_vectorized_text)
