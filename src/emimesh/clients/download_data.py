import argparse
from pathlib import Path
from emimesh.utils import np2pv
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cloudpath",
        help="path to cloud data",
        type=str,
        default="precomputed://gs://iarpa_microns/minnie/minnie65/seg",
    )
    parser.add_argument("--mip", help="resolution (0 is highest)", type=int, default=0)
    parser.add_argument(
        "--position",
        help="point position in x-y-z integer pixel position \
              (can be copied from neuroglancer)",
        type=str,
    )
    parser.add_argument(
        "--size",
        help="cube side length of the volume to be downloaded (in nm)",
        type=str,
        default=1000,
    )
    parser.add_argument(
        "--output", help="output filename", type=str, default="data.xdmf"
    )

    args = parser.parse_args()

    position = args.position.split("-")
    try:
        size = [int(args.size)] * 3
    except ValueError:
        size = [int(s) for s in args.size.split("-")]

    try:
        img,res = download_cloudvolume(args.cloudpath, args.mip, position, size)
    except:
        img,res = download_webknossos(args.cloudpath, args.mip, position, size)
        
    print(res)
    data = np2pv(img, res)
    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    data.save(args.output)


if __name__ == "__main__":
    main()