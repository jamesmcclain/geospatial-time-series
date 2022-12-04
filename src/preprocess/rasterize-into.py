#!/usr/bin/env python3

# The MIT License (MIT)
# =====================
#
# Copyright © 2022 James McClain
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the “Software”), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

import argparse
import copy
import json
import os


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True, type=str, nargs='+')
    return parser


if __name__ == '__main__':
    args = cli_parser().parse_args()

    for tif in list(filter(lambda s: s.endswith('.tif'), args.inputs)):
        if 'label' in tif:
            continue

        command = f'gdalinfo -json {tif}'
        gdalinfo = json.loads(os.popen(command).read())
        [width, height] = list(map(int, gdalinfo.get('size')))
        [x1, y1] = list(map(float, gdalinfo.get('cornerCoordinates').get('upperLeft')))
        [x2, y2] = list(map(float, gdalinfo.get('cornerCoordinates').get('lowerRight')))
        [xmin, xmax] = sorted([x1, x2])
        [ymin, ymax] = sorted([y1, y2])

        stem = tif[:-4].replace('-imagery', '')
        infile = f'/vsigzip/{stem}.geojson.gz'
        outfile = f'{stem}-label.tif'
        command = f'OGR_GEOJSON_MAX_OBJ_SIZE=0 gdal_rasterize -at -co "TILED=YES" -co "BIGTIFF=YES" -co "COMPRESS=deflate" -co "PREDICTOR=2" -a default -te {xmin} {ymin} {xmax} {ymax} -ts {width} {height} -ot Byte {infile} {outfile}'
        os.system(command)
