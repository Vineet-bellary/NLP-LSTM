from pathlib import Path

# Directories
ROOT_DIR = Path(__file__).parent.parent.parent

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

LSTM_MODEL_SAVE_PATH = MODELS_DIR / "lstm_model.pth"
BEST_LSTM_MODEL_SAVE_PATH = MODELS_DIR / "best_lstm_model.pth"

BI_LSTM_MODEL_SAVE_PATH = MODELS_DIR / "bilstm_model.pth"
BEST_BI_LSTM_MODEL_SAVE_PATH = MODELS_DIR / "best_bilstm_model.pth"

ATTENTION_MODEL_SAVE_PATH = MODELS_DIR / "attention_model.pth"
BEST_ATTENTION_MODEL_SAVE_PATH = MODELS_DIR / "best_attention_model.pth"

BI_LSTM_ATTENTION_MODEL_SAVE_PATH = MODELS_DIR / "bilstm_attention_model.pth"
BEST_BI_LSTM_ATTENTION_MODEL_SAVE_PATH = MODELS_DIR / "best_bilstm_attention_model.pth"

# Genral parameters
RANDOM_SEED = 42

# Data parameters
MAX_VOCAB_SIZE = 10000
BATCH_SIZE = 8
NUM_CLASSES = 2
MAX_SEQ_LEN = 300

# Class Mapping
CLASS_MAP = {
    "negative": 0,
    "positive": 1,
}

# Train Parameters
NUM_EPOCHS = 50

# Misc
# Clearly positive
sample_text = "The cinematography was stunning and the acting was phenomenal."

# Clearly negative
sample_text = "Waste of time. The plot was boring and predictable."

# Mixed but ultimately positive
sample_text = (
    "It had some slow parts, but overall an amazing story with great characters."
)

# Mixed but ultimately negative
sample_text = "The beginning was interesting, but it fell apart halfway through."

# Sarcasm (hard for models)
sample_text = (
    "Oh wow, a five-minute scene of someone staring at a wall. Brilliant filmmaking."
)

# Nuanced positive
sample_text = (
    "Not the best movie I've seen, but surprisingly decent and worth watching."
)

# Nuanced negative
sample_text = "It tries to be deep but ends up being pretentious and confusing."

# Extremely short positive
sample_text = "Perfect."

# Extremely short negative
sample_text = "Terrible."

# Long rambling positive
sample_text = "From the opening scene to the credits, every minute kept me engaged. The director's vision, the soundtrack, the emotional depth of characters - all exceptional."

# Long rambling negative
sample_text = "I sat through this expecting something meaningful but instead got confused dialogue, inconsistent character development, and a ending that made no sense whatsoever."
