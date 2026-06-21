import pandas as pd
from pathlib import Path
import torch

from imdb_sentiment.config import DATA_DIR, NUM_CLASSES
from imdb_sentiment.data_handling.preprocessing import preprocess_data
from imdb_sentiment.data_handling.build_dataloader import prepare_data
from imdb_sentiment.data_handling.data_split import split_data
from imdb_sentiment.model.lstm_model import LSTMModel
from imdb_sentiment.model.attention_model import AttentionModel
from imdb_sentiment.model.bilstm_model import BiLSTMModel
from imdb_sentiment.model.biattention_model import BiAttentionModel

def make_dataloaders():
    """Loads the Dataset, splits it into train/val/test sets, and prepares DataLoader objects for each."""
    df = pd.read_csv(DATA_DIR / "IMDB_dataset.csv")

    preprocessed_df, _unused_report = preprocess_data(df)

    train_df, val_df, test_df = split_data(df=preprocessed_df, stratify_col="sentiment")

    train_dataloader, vocabulary, max_doc_length = prepare_data(
        df=train_df, name="train"
    )
    val_dataloader, _unused_vocab, _unused_max_doc_length = prepare_data(
        df=val_df, vocab=vocabulary, doc_length=max_doc_length, name="validation"
    )
    test_dataloader, _unused_vocab, _unused_max_doc_length = prepare_data(
        df=test_df, vocab=vocabulary, doc_length=max_doc_length, name="test"
    )

    return train_dataloader, val_dataloader, test_dataloader, vocabulary, max_doc_length


def load_model_metadata(path: Path, name: str = "model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_state = torch.load(path, map_location=device)
    vocab = model_state["vocab"]
    max_doc_length = model_state["max_doc_length"]

    if name == "lstm":
        model = LSTMModel(
            vocab_size=len(vocab),
        )
        model.load_state_dict(model_state["model_state_dict"])
        model.to(device)
    elif name == "attention":
        model = AttentionModel(
            vocab_size=len(vocab),
        )
        model.load_state_dict(model_state["model_state_dict"])
        model.to(device)
    elif name == "bilstm":
        model = BiLSTMModel(
            vocab_size=len(vocab),
        )
        model.load_state_dict(model_state["model_state_dict"])
        model.to(device)
    elif name == "biattention":
        model = BiAttentionModel(
            vocab_size=len(vocab),
        )
        model.load_state_dict(model_state["model_state_dict"])
        model.to(device)
    else:
        raise ValueError(f"Unknown model name '{name}' provided for loading.")

    return model, vocab, max_doc_length
