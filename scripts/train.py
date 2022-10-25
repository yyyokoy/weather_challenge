import os
import sys
from pathlib import Path

from lib.sat_train import sat_train


class Train():
    """ The training process. """

    def __init__(self, arguments):

        self.args = arguments

        self.feature_dir = self.args.feature_dir
        self.weight_dir = self.args.weight_dir
        self.nt = self.args.nt
        self.batch_size = self.args.batch_size
        self.nb_epoch = self.args.nb_epoch

        self.suffix = self.args.suffix

    def process(self):
        print(self.nb_epoch)

        sat_train(self.feature_dir, self.weight_dir,
                  self.nt, self.batch_size, self.nb_epoch, self.suffix)
