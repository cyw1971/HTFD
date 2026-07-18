"""SPX default run settings (PaD-TS-style config module)."""

DATASET = "spx"
SEQ_LEN = 128
TRAIN_EPOCHS = 200
BATCH_SIZE = 2000
D_MODEL = 64
LEARNING_RATE = 1e-4
NORM_MODE = "revin"
ROOT_PATH = "./dataset/"
DATA_PATH = "SPX.csv"
