import os
import sys
from pathlib import Path

from lib.predict_extrap import predict_extrap, submit


class Predict():
    """ The predicting process. """

    def __init__(self, arguments):

        self.args = arguments

        self.weight_dir = self.args.weight_dir
        self.feature_dir = self.args.feature_dir
        self.result_dir = self.args.result_dir
        self.submit_dir = self.args.submit_dir
        self.resize = self.args.resize
        self.nt = self.args.nt
        self.batch_size = self.args.batch_size

        self.suffix = self.args.suffix

    def process(self):

     #   predict_extrap(self.weight_dir, self.feature_dir,
      #                 self.result_dir, self.batch_size, self.nt, self.suffix)
        submit(self.result_dir, self.submit_dir,
               self.nt, self.resize, self.suffix)
