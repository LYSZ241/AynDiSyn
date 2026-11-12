"""Run the generic baseline using the selected numeric dataset."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.training import run
from R01.parser import parse_args
from R01.load_data5_myfold import Data
from R01.gcn_model3 import DemoModel

if __name__ == '__main__':
    run(parse_args(), 'R01', Data, DemoModel)
