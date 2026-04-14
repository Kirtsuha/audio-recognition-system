from pathlib import Path

NOISE_LEVEL = 0.02
PITCH_SHIFT_N_STEPS = 2
TIME_STRETCH_RATE = 1.1

SEGMENT_SECONDS = 5
SR = 16000
N_SEGMENTS = 10

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "data" / "model.pt"