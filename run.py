""" The master run.py script"""

import sys

import lib.cli as cli

if sys.version_info[0] < 3:
    raise Exception("This program requires at least python3.6")
if sys.version_info[0] == 3 and sys.version_info[1] < 6:
    raise Exception("This program requires at least python3.6")


def bad_args(args):
    """ Print help on bad arguments """
    PARSER.print_help()
    exit(0)


if __name__ == "__main__":
    PARSER = cli.FullHelpArgumentParser()
    SUBPARSER = PARSER.add_subparsers()
    FILL_IMG = cli.FillImgArgs(SUBPARSER,
                               "fill",
                               "fill missing images with dummy.")
    PREPROCESS = cli.PreprocessArgs(SUBPARSER,
                                "prep",
                                "preprocess images.")
    TRAIN = cli.TrainArgs(SUBPARSER,
                              "train",
                              "train prednet.")
    EXTRAP = cli.ExtrapArgs(SUBPARSER,
                            "extrap",
                            "extrap finetune prednet.")
    PREDICT = cli.PredictArgs(SUBPARSER,
                              "predict",
                              "predict extrap.")


PARSER.set_defaults(func=bad_args)
ARGUMENTS = PARSER.parse_args()
ARGUMENTS.func(ARGUMENTS)
