import torch
from pathlib import Path
from torch.utils.data import DataLoader

from imdb_sentiment.load_data import load_model_metadata
from imdb_sentiment.config import (
    LOGS_DIR,
    BEST_BI_LSTM_MODEL_SAVE_PATH,
    BEST_BI_LSTM_ATTENTION_MODEL_SAVE_PATH,
)
from imdb_sentiment.util.logger import setup_logger
from imdb_sentiment.load_data import make_dataloaders

logger = setup_logger(LOGS_DIR / Path(__file__).stem, mode="w")


def evaluate_model(model_path: Path, loader: DataLoader, device: torch.device):
    """Evaluates the model on the provided DataLoader and returns accuracy."""
    model, _unused_vocab, _unused_max_doc_length = load_model_metadata(
        model_path, "biattention"
    )
    model.eval()

    total_correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            pred_class = torch.argmax(model(inputs), dim=1)

            total_correct_predictions += (pred_class == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct_predictions / total_samples
    logger.info(f"Evaluation Accuracy: {accuracy:.2f}")

    return accuracy


if __name__ == "__main__":
    _, _, test_loader, _unused_vocab, _unused_max_doc_length = make_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy = evaluate_model(
        Path(BEST_BI_LSTM_ATTENTION_MODEL_SAVE_PATH), test_loader, device
    )
