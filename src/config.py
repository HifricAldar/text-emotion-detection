# Data parameters
CLASS_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
SAMPLES_PER_CLASS = 15000
TEST_SIZE = 0.2
VAL_SIZE = 0.5

# Model parameters
VOCAB_SIZE = 14500
MAX_LENGTH = 32
EMBEDDING_DIM = 16
NUM_CLASSES = 6

# Training parameters
EPOCHS = 10
BATCH_SIZE = 32

# Paths
DATA_PATH = "data/raw/data.jsonl"
MODEL_SAVE_PATH = "results/models/emotion_classification_model.h5"
TOKENIZER_SAVE_PATH = "results/models/tokenizer.pickle"
TRAINING_HISTORY_GRAPH_IMG = "results/graphics/training_history.png"
CONFUSION_MATRIX_IMG = "results/graphics/confusion_matrix.png"
TRAIN_PROCESSED_PATH = "data/processed/train.csv"
VAL_PROCESSED_PATH = "data/processed/val.csv"
TEST_PROCESSED_PATH = "data/processed/test.csv"