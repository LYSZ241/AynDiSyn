"""Generic ranking evaluation, not research-specific inference."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.training import evaluate
from common.metrics import ranking_metrics
