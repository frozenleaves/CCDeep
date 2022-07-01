import os
import argparse
import sys

import prediction


parser = argparse.ArgumentParser(description="Welcome to use CCDeep!", add_help=False)
help_content = """
    using this script to auto segment the cell images and identify each cell's  cycle phase.
    usage:
        python main.py -pcna <your pcna image filepath>  -bf <your bf image filepath> -o <output result filepath> 
"""
parser.add_argument("-h", "--help", action="help", help=help_content)
parser.add_argument('-pcna', default=False, help="input image filepath of pcna")
parser.add_argument('-o', default=False, help='output json file path')
parser.add_argument('-bf', default=False, help='input image filepath of bright field')

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(0)

if args.pcna is False:
    raise ValueError("pcna image must be given!")
else:
    pcna = args.pcna
if args.bf is False:
    raise ValueError("bf image must be given!")
else:
    bf = args.bf
if args.o is False:
    output = os.path.basename(pcna.replace('.tif', '.json'))
    print(f"-o  not provided, using the default output file name: {output}")
else:
    if not args.o.endswith('.json'):
        raise ValueError("output filename need <.json> extend name")
    output = args.o

prediction.segment(pcna=pcna, bf=bf, output=output, segment_model=None)
