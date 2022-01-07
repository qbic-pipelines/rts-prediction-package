import abc
from argparse import ArgumentParser

import pytorch_lightning as pl

__all__ = ['UNetsuper']


class UNetsuper(pl.LightningModule):
    def __init__(self, num_classes, len_test_set: int, input_channels=1, min_filter=32, **kwargs):
        super(UNetsuper, self).__init__()
        self.num_classes = num_classes
        self.args = kwargs
        self.len_test_set = len_test_set

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers', type=int, default=16, metavar='N', help='number of workers (default: 3)')
        return parser

    @abc.abstractmethod
    def forward(self, x):
        pass
