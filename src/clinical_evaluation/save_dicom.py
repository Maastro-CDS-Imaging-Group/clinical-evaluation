# -*- coding: utf-8 -*-
"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
         fibonacci = clinical_evaluation.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Union

try:
    import clinical_evaluation
except:
    print("Importing as local module")
    sys.path.append(f"{os.getcwd()}/src")

from clinical_evaluation import __version__
from clinical_evaluation.dicom_saver import dicom_saver

__author__ = "Suraj Pai"
__copyright__ = "Suraj Pai"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def save(args):
    ds = dicom_saver.DicomSaver(output_dir=args.output_dir)
    ds.save(args.image_path, args.dicom_path)


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Peform registration using SimpleElastix between two nrrd volumes")
    parser.add_argument("--version",
                        action="version",
                        version="clinical-evaluation {ver}".format(ver=__version__))

    parser.add_argument(dest="image_path",
                        help="Path to image that can be read by SimpleITK",
                        type=Path)

    parser.add_argument("--dicom_path",
                        dest="dicom_path",
                        help="Path to dicom file to carry over metadata from",
                        type=Path)

    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        help="Path to save the created dicom, if directory does not exist it will be created",
        default="dicom_saver_output",
        type=Path)

    parser.add_argument("-v",
                        "--verbose",
                        dest="loglevel",
                        help="set loglevel to INFO",
                        action="store_const",
                        const=logging.INFO)

    parser.add_argument("-vv",
                        "--very-verbose",
                        dest="loglevel",
                        help="set loglevel to DEBUG",
                        action="store_const",
                        const=logging.DEBUG)

    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel,
                        stream=sys.stdout,
                        format=logformat,
                        datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")
    save(args)
    _logger.info("Script ends here")


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
