#!/usr/bin/env python3

import argparse

from RISCluster.environment import KPDR_sac2mseed


def main(args):
    datadir = args.datadir
    destdir = args.destdir
    response = args.response

    KPDR_sac2mseed(datadir, destdir, response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts station KPDR SAC files to MSEED."
    )
    parser.add_argument(
        "--datadir",
        default='.',
        nargs='?',
        help="Source path ['.']"
    )
    parser.add_argument(
        "--destdir",
        default='.',
        nargs='?',
        help="Destination path ['.']"
    )
    parser.add_argument(
        "--response",
        dest='response',
        action="store_true",
        help="Rmv instr resp? True/[False]"
    )
    parser.set_defaults(response=False)
    args = parser.parse_args()
    main(args)