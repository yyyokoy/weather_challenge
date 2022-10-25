import os
import sys
from pathlib import Path

from lib.extrap_finetune_sat import extrap_finetune_sat


class Extrap():
    """ The extrap finetune process. """

    def __init__(self, arguments):

        self.args = arguments

        self.feature_dir = self.args.feature_dir
        self.weight_dir = self.args.weight_dir
        self.nt = self.args.nt
        self.extrap_start_time = self.args.extrap_start_time
        self.nb_epoch = self.args.nb_epoch
        self.batch_size = self.args.batch_size

        self.suffix = self.args.suffix

    def process(self):
        print(self.nb_epoch)

        extrap_finetune_sat(self.feature_dir, self.weight_dir, self.nt,
                            self.extrap_start_time, self.nb_epoch, self.batch_size, self.suffix)
