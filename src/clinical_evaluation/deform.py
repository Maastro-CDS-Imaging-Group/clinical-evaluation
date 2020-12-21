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
import logging
from pathlib import Path

from clinical_evaluation import __version__
from clinical_evaluation.pipeline import EvaluationPipeline

__author__ = "Suraj Pai"
__copyright__ = "Suraj Pai"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def deform(args):
    """Deform and save the volume
    """
    pipeline = EvaluationPipeline()

    # Load the source and target
    source = pipeline.load(args.source)
    target = pipeline.load(args.target)

    if args.preprocess_source:
        source = pipeline.preprocess(source, preprocess_fn=args.preprocess_source)

    if args.preprocess_target:
        target = pipeline.preprocess(target, preprocess_fn=args.preprocess_target)

    deformed_image, _ = pipeline.deform(source, target, args.params)
    
    # Save all 3 images
    pipeline.save(deformed_image, args.output_dir)
    pipeline.save(source, args.output_dir, tag='source')
    pipeline.save(target, args.output_dir, tag='target')


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Peform registration using SimpleElastix between two nrrd volumes")
    parser.add_argument(
        "--version",
        action="version",
        version="clinical-evaluation {ver}".format(ver=__version__))


    parser.add_argument(
        dest="source",
        help="Source volume, the volume to be deformed",
        type=Path)

    parser.add_argument(
        "--preprocess_source",
        dest="preprocess_source",
        help="Enable preprocessing with a certain schema. This schema can be defined in preprocess.py and the name of the function \
              can be provided here to run the op. This argument processes source image", type=str)
            
    
    parser.add_argument(
      dest="target",
      help="Target volume, the volume that is used as reference for the deformation",
      type=Path)

    parser.add_argument(
        "--preprocess_target",
        dest="preprocess_target",
        help="Enable preprocessing with a certain schema. This schema can be defined in preprocess.py and the name of the function \
              can be provided here to run the op. This argument processes target image", type=str)      


    parser.add_argument(
      "-p",
      "--params",
      dest="params",
      help="Path to the parameters file to be used for the registration. See here: \
      https://simpleelastix.readthedocs.io/ParameterMaps.html",
      type=Path)      
    

    parser.add_argument(
      "-o",
      "--output_dir",
      dest="output_dir",
      help="Path to output the result image, if it is a not existing, it will be created",
      default='.',
      type=Path)      
    

    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)



    parser.add_argument(
        "-vv",
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
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")
    deform(args)
    _logger.info("Script ends here")


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
