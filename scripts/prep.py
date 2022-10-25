import os
import sys
from pathlib import Path

from lib.prep_sat_img import vectorize_train_sat_image, vectorize_test_sat_image


class Prep():
    """ The preprocessing process. """

    def __init__(self, arguments):

        self.args = arguments

        self.external_dir = self.args.external_dir
        self.feature_dir = self.args.feature_dir
        self.input_dir = self.args.input_dir
        self.sat_dir = self.args.sat_dir
        self.resize = self.args.resize
        self.nt = self.args.nt
        self.window = self.args.window
        self.ratio = self.args.ratio

        self.suffix = self.args.suffix

    def process(self):
        print(self.mode)

        vectorize_train_sat_image(self.feature_dir, self.external_dir,
                                  self.resize, self.nt, self.window, self.ratio, self.suffix)
        vectorize_test_sat_image(
            self.input_dir, self.feature_dir, self.sat_dir, self.resize, self.nt, self.suffix)
