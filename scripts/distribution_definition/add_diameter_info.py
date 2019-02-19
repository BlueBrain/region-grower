import argparse
import json

def main(args):
    with open(args.input_distrib) as f:
        d = json.load(f)

    with open(args.diameter_info) as f:
        diam = json.load(f)


    for mtype in d['mtypes'].keys():
        d['mtypes'][mtype]['diameter'] = diam

    with open(args.output_distrib, 'w+') as f:
        json.dump(d, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Add the diameters distribution to the input distribution")
    parser.add_argument(
        "input_distrib",
        help="Input distributions"
    )
    parser.add_argument(
        "diameter_info",
        help="Diameter info",
    )
    parser.add_argument(
        "output_distrib",
        help="Output distrib"
    )
    main(parser.parse_args())
