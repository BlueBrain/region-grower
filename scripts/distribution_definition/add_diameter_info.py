#!/usr/bin/env python
import os
import argparse
import json

def main(args):
    with open(args.input_distrib) as f:
        d = json.load(f)

    mtypes_input_distrib = set(d['mtypes'])
    mtypes_from_diameter_folder = set()

    for f in os.listdir(args.diameter_folder):
        if not f.endswith('.json'):
            continue
        mtypes_from_diameter_folder.add(f[:-5])

    missing_from_input_distrib = mtypes_from_diameter_folder - mtypes_input_distrib
    if missing_from_input_distrib:
        print('Mtypes missing from keys of {}: {}'.format(args.input_distrib, missing_from_input_distrib))

    missing_from_diameter_folder = mtypes_input_distrib - mtypes_from_diameter_folder
    if missing_from_diameter_folder:
        print('Mtypes missing from diameter folder: {}'.format(missing_from_diameter_folder))

    for mtype in d['mtypes'].keys():
        with open(os.path.join(args.diameter_folder, '{}.json'.format(mtype))) as f:
            diam_info = json.load(f)
            if not 'method' in diam_info:
                diam_info['method'] = 'M5'
            d['mtypes'][mtype]['diameter'] = diam_info

    with open(args.output_distrib, 'w+') as f:
        json.dump(d, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Add the diameters distribution to the input distribution")
    parser.add_argument(
        "input_distrib",
        help="Input distributions"
    )
    parser.add_argument(
        "diameter_folder",
        help="Diameter folder",
    )
    parser.add_argument(
        "output_distrib",
        help="Output distrib"
    )
    main(parser.parse_args())
