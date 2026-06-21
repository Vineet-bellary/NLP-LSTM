import torch
from pathlib import Path
from torch.utils.data import DataLoader

# from imdb_sentiment.model.lstm_model import LSTMModel
# from imdb_sentiment.model.attention_model import AttentionModel
from imdb_sentiment.load_data import load_model_metadata
from imdb_sentiment.data_handling.build_dataloader import prepare_sample_text
from imdb_sentiment.config import LOGS_DIR, BEST_ATTENTION_MODEL_SAVE_PATH, CLASS_MAP
from imdb_sentiment.util.logger import setup_logger

logger = setup_logger(LOGS_DIR / Path(__file__).stem, mode="w")


def infer(
    model_path: Path,
    sample_text: list[str],
    vocab: set[str],
    max_doc_length: int,
    device: torch.device,
):
    """Performs inference on a sample text using the specified model."""
    model, _unused_vocab, _unused_max_doc_length = load_model_metadata(
        model_path, "attention"
    )
    model.eval()
    reverse_class_map = {v: k for k, v in CLASS_MAP.items()}
    with torch.no_grad():
        for text in sample_text:
            sample_tensor = (
                prepare_sample_text(text, vocab, max_doc_length)
                .unsqueeze(dim=0)
                .to(device)
            )
            output = model(sample_tensor)
            predicted_class_id = torch.argmax(output, dim=1).item()
            predicted_class = reverse_class_map[predicted_class_id]

            logger.info(f"Input Text: {text[:25]}...: {predicted_class}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_text = [
        "The cinematography was stunning and the acting was phenomenal.",
        "Waste of time. The plot was boring and predictable.",
        "It had some slow parts, but overall an amazing story with great characters.",
        "The beginning was interesting, but it fell apart halfway through.",
        "Oh wow, a five-minute scene of someone staring at a wall. Brilliant filmmaking.",
        "Not the best movie I've seen, but surprisingly decent and worth watching.",
        "It tries to be deep but ends up being pretentious and confusing.",
        "Perfect.",
        "Terrible.",
        "From the opening scene to the credits, every minute kept me engaged. The director's vision, the soundtrack, the emotional depth of characters - all exceptional.",
        "I sat through this expecting something meaningful but instead got confused dialogue, inconsistent character development, and a ending that made no sense whatsoever.",
    ]
    model, vocab, max_doc_length = load_model_metadata(
        Path(BEST_ATTENTION_MODEL_SAVE_PATH), "attention"
    )
    infer(
        Path(BEST_ATTENTION_MODEL_SAVE_PATH), sample_text, vocab, max_doc_length, device
    )
