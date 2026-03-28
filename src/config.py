from pathlib import Path


# Project root folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data folder
DATA_DIR = PROJECT_ROOT / "data"

# Results folder
RESULTS_DIR = PROJECT_ROOT / "results"
BERT_RESULTS_DIR = RESULTS_DIR / "bert"
GPT_RESULTS_DIR = RESULTS_DIR / "gpt"
BERT_MODELS_DIR = BERT_RESULTS_DIR / "models"
GPT_MODELS_DIR = GPT_RESULTS_DIR / "models"

# Dataset file
DATASET_PATH = DATA_DIR / "IMDB Dataset.csv"

# Column names used in the IMDB dataset
TEXT_COLUMN = "review"
LABEL_COLUMN = "sentiment"

# Training and preprocessing settings
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Model input settings.
MAX_SEQUENCE_LENGTH = 256

# Training settings
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
NUM_LABELS = 2
TRAIN_FRACTION = 1

# Model names for Hugging Face api
BERT_MODEL_NAME = "bert-base-uncased"
GPT_MODEL_NAME = "openai-gpt"

def format_train_fraction_tag(train_fraction):
    """
    Convert the training fraction into a short tag like:
    - 1.0 -> train100
    - 0.1 -> train10
    - 0.05 -> train5
    """
    percent_value = train_fraction * 100

    if float(percent_value).is_integer():
        return f"train{int(percent_value)}"

    percent_text = f"{percent_value:g}".replace(".", "_")
    return f"train{percent_text}"


def format_learning_rate_tag(learning_rate):
    """
    Convert learning rate into a compact tag like:
    - 2e-5 -> lr2e5
    """
    scientific_text = f"{learning_rate:.0e}"
    scientific_text = scientific_text.replace("e-0", "e").replace("e-", "e")
    return f"lr{scientific_text}"


RUN_TAG = f"{format_train_fraction_tag(TRAIN_FRACTION)}_{format_learning_rate_tag(LEARNING_RATE)}_seed{RANDOM_SEED}"


# Output files
BERT_METRICS_PATH = BERT_RESULTS_DIR / f"bert_{RUN_TAG}.txt"
GPT_METRICS_PATH = GPT_RESULTS_DIR / f"gpt_{RUN_TAG}.txt"
BERT_MODEL_PATH = BERT_MODELS_DIR / f"bert_{RUN_TAG}.pt"
GPT_MODEL_PATH = GPT_MODELS_DIR / f"gpt_{RUN_TAG}.pt"
