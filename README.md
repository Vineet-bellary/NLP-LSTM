# IMDB Sentiment Classification (LSTM Variants)

Minimal PyTorch project for binary sentiment classification on the IMDB dataset.

## Project Layout

- `data/IMDB_dataset.csv`: dataset file
- `src/imdb_sentiment/train.py`: training entrypoint
- `src/imdb_sentiment/eval.py`: evaluation entrypoint
- `src/imdb_sentiment/infer.py`: inference entrypoint
- `src/imdb_sentiment/config.py`: paths and global hyperparameters
- `models/`: saved checkpoints

## Quick Start

Run from the project root.

### 1) Activate environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install torch pandas scikit-learn
```

### 3) Run

```powershell
cd src
```

Train:

```powershell
python -m imdb_sentiment.train
```

Evaluate:

```powershell
python -m imdb_sentiment.eval
```

Inference:

```powershell
python -m imdb_sentiment.infer
```

## Change Models

Supported model keys used by `load_model_metadata(...)`:

- `lstm`
- `bilstm`
- `attention`
- `biattention`

### Training model switch

Edit `src/imdb_sentiment/train.py` inside `main()`:

1. Choose the model instance you want to train.
2. Set optimizer to that model's parameters.
3. Pass that model to `train_model(...)`.
4. Save to the matching checkpoint constant from `src/imdb_sentiment/config.py`.

Example pattern:

```python
    model = AttentionModel(vocab_size=len(vocabulary))
    optimizer = Adam(model.parameters(), lr=0.001)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        vocab=vocabulary,
        max_doc_length=max_doc_length,
    )

    save_model(BEST_ATTENTION_MODEL_SAVE_PATH, model, optimizer, vocabulary, max_doc_length)
```

### Evaluation / inference model switch

Edit these two things together in each script:

- checkpoint path constant (for example `BEST_LSTM_MODEL_SAVE_PATH`)
- model key string passed to `load_model_metadata(...)` (for example `"lstm"`)

Files:

- `src/imdb_sentiment/eval.py`
- `src/imdb_sentiment/infer.py`

If path and key do not match, checkpoint loading will fail.
