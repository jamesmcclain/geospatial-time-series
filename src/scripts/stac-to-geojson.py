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
import zipfile

import pystac
from pystac.stac_io import DefaultStacIO, StacIO
from shapely.geometry import mapping, shape
from shapely.ops import unary_union


class CustomStacIO(DefaultStacIO):

    def read_text(self, source, *args, **kwargs) -> str:
        zip_file, stac_file = source.split('.zip/')
        try:
            with zipfile.Path(f'{zip_file}.zip', stac_file) as path:
                return path.read_text(encoding='UTF-8')
        except:
            with zipfile.ZipFile(f'{zip_file}.zip') as z:
                return z.read(stac_file)

        def write_text(self, dest, txt, *args, **kwargs) -> None:
            pass


StacIO.set_default(CustomStacIO)


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stac', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=False, default='./')
    return parser


M = {
    "Agricultural areas": 1,
    "Artificial areas": 2,
    "Forest and semi-natural areas": 3,
    "Water bodies": 4,
    "Wetlands": 5,
}

if __name__ == '__main__':
    args = parser().parse_args()

    cat = pystac.Catalog.from_file(f'{args.stac}/catalog.json')
    cat.make_all_asset_hrefs_absolute()

    for collection in cat.get_collections():
        if 'scene' not in collection.id:
            continue
        for item in collection.get_items():
            name = str(item.id)

    for collection in cat.get_collections():
        if 'labels' not in collection.id:
            continue
        for item in collection.get_items():
            stac_loc, data_loc = item.assets.get('data').href.split('.zip/')
            stac_loc = stac_loc + '.zip'
            try:
                with zipfile.Path(stac_loc, data_loc) as path:
                    data = json.loads(path.read_text(encoding='UTF-8'))
            except:
                with zipfile.ZipFile(stac_loc) as z:
                    data = json.loads(z.read(data_loc))
            link = item.get_links('source')[0]
            other_item = link.resolve_stac_object().target
            name = str(other_item.id)

            # For each label, append label to list
            data2 = {}
            for feature in data.get('features'):
                sh = shape(feature.get('geometry')).buffer(1e-5)
                default = feature.get('properties').get('default')
                if default not in data2:
                    data2.update({default: []})
                data2.get(default).append(sh)

            # For each label type, convert labels into single label
            stem = name
            data3 = {
                'type': 'FeatureCollection',
                'features': [],
            }
            for k, v in data2.items():
                data3.get('features').append({
                    'type':
                    'Feature',
                    'geometry':
                    mapping(unary_union(v)),
                    'properties': {
                        'default': M.get(k),
                    },
                })

            # Write to a file
            filename = f"{args.output_dir}/{stem}.geojson"
            print(filename)
            with open(filename, "w") as f:
                json.dump(data3, f, indent=4, sort_keys=True)
