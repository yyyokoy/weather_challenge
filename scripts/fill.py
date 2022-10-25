import os
import sys
from pathlib import Path

from lib.fill_images import create_data_df
from lib.fill_images import create_dummy_image


class Fill():
    """ The filling process. """

    def __init__(self, arguments):

        self.args = arguments
        self.sat_dir = self.args.sat_dir
        self.external_dir = self.args.external_dir

    def process(self):

        print(self.sat_dir)
        print(self.external_dir)

        if 'train' in self.sat_dir:
            split = 'train'
        elif 'test' in self.sat_dir:
            split = 'test'

        create_data_df(self.sat_dir, self.external_dir)
        create_dummy_image(split, self.external_dir)
