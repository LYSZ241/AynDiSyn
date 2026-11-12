"""Run the synthetic-only public demonstration."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.training import run
from gcn.parser import parse_args
from gcn.load_data5_myfold import Data
from gcn.gcn_model3 import DemoModel

if __name__ == '__main__':
    run(parse_args(), 'gcn', Data, DemoModel)
