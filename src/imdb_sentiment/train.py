from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from imdb_sentiment.config import (
    LOGS_DIR,
    NUM_EPOCHS,
    LSTM_MODEL_SAVE_PATH,
    BEST_LSTM_MODEL_SAVE_PATH,
    BI_LSTM_MODEL_SAVE_PATH,
    BEST_BI_LSTM_MODEL_SAVE_PATH,
    ATTENTION_MODEL_SAVE_PATH,
    BEST_ATTENTION_MODEL_SAVE_PATH,
    BI_LSTM_ATTENTION_MODEL_SAVE_PATH,
    BEST_BI_LSTM_ATTENTION_MODEL_SAVE_PATH,
)
from imdb_sentiment.util.logger import setup_logger
from imdb_sentiment.load_data import make_dataloaders
from imdb_sentiment.model.lstm_model import LSTMModel
from imdb_sentiment.model.attention_model import AttentionModel
from imdb_sentiment.model.bilstm_model import BiLSTMModel
from imdb_sentiment.model.biattention_model import BiAttentionModel

# from imdb_sentiment.model.bilstm_model import BiAttentionModel

logger = setup_logger(LOGS_DIR / Path(__file__).stem, mode="w")


def val_model(data: DataLoader, model, loss_fn, device: torch.device):
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_samples = 0

    with torch.no_grad():

        for x, y in data:

            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            loss = loss_fn(pred, y)

            predicted_classes = torch.argmax(pred, dim=1)

            val_correct += (predicted_classes == y).sum().item()

            val_samples += y.size(0)

            val_loss += loss.item() * y.size(0)

    val_loss /= val_samples

    val_accuracy = val_correct / val_samples

    return val_loss, val_accuracy


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn,
    optimizer,
    scheduler,
    device,
    vocab,
    max_doc_length,
    epochs: int = NUM_EPOCHS,
):
    logger.info(f"Using device: {device}")
    model.to(device)
    logger.info(f"Starting training for {epochs} epochs...")
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()

            predicted_classes = torch.argmax(preds, dim=1)
            epoch_correct += (predicted_classes == batch_y).sum().item()
            epoch_samples += batch_y.size(0)
            epoch_loss += loss.item() * batch_y.size(0)

        epoch_loss /= epoch_samples
        epoch_accuracy = epoch_correct / epoch_samples

        val_loss, val_accuracy = val_model(val_loader, model, loss_fn, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if isinstance(model, LSTMModel):
                save_model(
                    BEST_LSTM_MODEL_SAVE_PATH, model, optimizer, vocab, max_doc_length
                )
            elif isinstance(model, BiLSTMModel):
                save_model(
                    BEST_BI_LSTM_MODEL_SAVE_PATH,
                    model,
                    optimizer,
                    vocab,
                    max_doc_length,
                )
            elif isinstance(model, AttentionModel):
                save_model(
                    BEST_ATTENTION_MODEL_SAVE_PATH,
                    model,
                    optimizer,
                    vocab,
                    max_doc_length,
                )
            elif isinstance(model, BiAttentionModel):
                save_model(
                    BEST_BI_LSTM_ATTENTION_MODEL_SAVE_PATH,
                    model,
                    optimizer,
                    vocab,
                    max_doc_length,
                )
            else:
                logger.warning("Unknown model type. Model not saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"LR: {current_lr:.6f}\nEpoch: {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f},  Train Accuracy: {epoch_accuracy:.2f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}"
        )


def save_model(path: Path, model, optimizer, vocab, max_doc_length):
    model_save = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab": vocab,
        "max_doc_length": max_doc_length,
    }
    torch.save(model_save, path)
    logger.info(f"Model saved to {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, vocabulary, max_doc_length = (
        make_dataloaders()
    )

    lstm_model = LSTMModel(vocab_size=len(vocabulary))
    bilstm_model = BiLSTMModel(vocab_size=len(vocabulary))
    attention_model = AttentionModel(vocab_size=len(vocabulary))
    biattention_model = BiAttentionModel(vocab_size=len(vocabulary))

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(biattention_model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    train_model(
        model=biattention_model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        vocab=vocabulary,
        max_doc_length=max_doc_length,
    )

    save_model(
        BEST_BI_LSTM_ATTENTION_MODEL_SAVE_PATH,
        biattention_model,
        optimizer,
        vocabulary,
        max_doc_length,
    )


if __name__ == "__main__":
    main()
