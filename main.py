import os
import argparse
import sys
from . import prediction, track
import logging


sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))


parser = argparse.ArgumentParser(description="Welcome to use CCDeep!", add_help=False)
help_content = """
    using this script to auto segment the cell images and identify each cell's  cycle phase.
    usage:
        python main.py -pcna <your pcna image filepath>  -bf <your bf image filepath> -o <output result filepath> 
"""
parser.add_argument('-ns', '--ns', action='store_true', default=False,
                    help='Optional parameter, segment or not, if call -ns, means do not execute segmentation.')
parser.add_argument('-t', "--track", action='store_true', help='Optional parameter, track or not')
parser.add_argument("-h", "--help", action="help", help=help_content)
parser.add_argument('-p', "--pcna", default=False, help="input image filepath of pcna")
parser.add_argument('-o', "--output", default=False, help='output json file path')
parser.add_argument('-bf', "--bf", default=False, help='input image filepath of bright field')
parser.add_argument('-ot', "--ot", default=False, help='tracking output result saved dir')
parser.add_argument('-js', "--js", default=False, help='tracking output result saved dir')

args = parser.parse_args()

print(args.ns)

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
if args.output is False:
    output = os.path.basename(pcna.replace('.tif', '.json'))
    logging.warning(f"-o  not provided, using the default output file name: {output}")
else:
    if not args.output.endswith('.json'):
        raise ValueError("output filename need <.json> extend name")
    output = args.output

if args.track is True and args.ns is True and args.js is False:
    raise ValueError("If you just want to do tracking, please give the `-js` parameter")
if args.track is True and not args.ns and args.js:
    raise ValueError("Parameters are ambiguous, please do not give `-js` when you do the segmentation and tracking.")

if not args.ns:
    jsons = prediction.segment(pcna=pcna, bf=bf, output=output, segment_model=None)

if args.ns and args.js:
    jsons = args.js


if args.track:
    if args.ot:
        track_output = args.ot
        print(args.track)
        print(args.ot)
    else:
        track_output = os.path.dirname(output)
        print(track_output)
    # track.start_track(fjson=jsons, fpcna=pcna, fbf=bf, fout=track_output)
