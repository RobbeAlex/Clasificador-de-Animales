# Variables utilizadas en el flujo de trabajo
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_PATH = str(BASE_DIR / "data")
TRAIN_PATH = str(BASE_DIR / "data")
TEST_PATH = str(BASE_DIR / "data")

# Random seed
SEED = 42

# Models path
MODEL_PATH = str(BASE_DIR / "models" / "modelo.pth")

# TRAINING
EPOCHS = 100
LEARNING_RATE = 0.05
HIDDEN_UNITS = 20
