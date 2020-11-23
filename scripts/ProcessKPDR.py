import argparse
import sys
sys.path.insert(0, '../RISCluster/')

from processing import file2dt, KPDR_sac2mseed, remove_trace

debug = False

if __name__ == "__main__":
    if debug:
        datadir = '/Users/williamjenkins/Research/Data/RIS_Seismic/HDH/Test'
        destdir = '/Users/williamjenkins/Research/Data/RIS_Seismic/HDH/Test/Processing'
        response = True
        pass
    else:
        parser = argparse.ArgumentParser(
            description="Converts station KPDR SAC files to MSEED."
        )
        parser.add_argument("--datadir", default='.', nargs='?', help="Source path ['.']")
        parser.add_argument("--destdir", default='.', nargs='?', help="Destination path ['.']")
        parser.add_argument("--response", dest='response', action="store_true", help="Rmv instr resp? True/[False]")
        parser.set_defaults(response=False)
        args = parser.parse_args()
        datadir = args.datadir
        destdir = args.destdir
        response = args.response

    KPDR_sac2mseed(datadir, destdir, response)
