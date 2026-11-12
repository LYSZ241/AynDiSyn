"""CLI options for generic demonstrations, not the research hyperparameters."""
import argparse
import ast
from pathlib import Path
from .data import DEFAULT_DATA, ROOT


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Synthetic-only public runnable example')
    parser.add_argument('--data_path', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output_dir', type=Path, default=ROOT/'outputs')
    parser.add_argument('--fold', type=int, choices=range(5), default=0)
    parser.add_argument('--all_folds', type=int, choices=[0, 1], default=0)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--embed_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--Ks', default='[5,10,15,20]')
    parser.add_argument('--save_checkpoint', type=int, choices=[0, 1], default=0)
    args = parser.parse_args(argv)
    args.Ks = ast.literal_eval(args.Ks)
    if not isinstance(args.Ks, list) or not args.Ks or any(type(k) is not int or k < 1 for k in args.Ks):
        parser.error('--Ks must be a nonempty list of positive integers')
    if args.epoch < 1 or args.batch_size < 1 or args.embed_size < 1 or args.lr <= 0:
        parser.error('Epoch, batch size, embedding size and learning rate must be positive')
    return args
