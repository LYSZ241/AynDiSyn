"""Generic ID batch helpers."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import batch_tensors, seed_everything
