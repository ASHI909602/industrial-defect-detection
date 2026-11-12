from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = ROOT / "raw_data"
MODEL_PATH = ROOT / "models" / "best_model.pt"
REPORT_DIR = ROOT / "reports"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
SEED = 42

# Dataset folder me "no_defect" naam ka class hona chahiye
NO_DEFECT_CLASS = "no_defect"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
