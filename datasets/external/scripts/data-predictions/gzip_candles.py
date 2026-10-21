"""Losslessly gzip the plain .json Kalshi candle files (run only when no fetcher is running).

Each file is compressed to <name>.gz, the compressed copy is verified to decompress to the
identical bytes, and only then is the original removed. Readers (kalshi_lib) accept both.
"""
import glob
import gzip
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW  # noqa: E402


def main():
    files = glob.glob(os.path.join(RAW, "kalshi", "candles", "*", "*_60.json"))
    before = after = 0
    for f in files:
        gz = f + ".gz"
        data = open(f, "rb").read()
        if os.path.exists(gz):
            if gzip.decompress(open(gz, "rb").read()) != data:
                print("CONFLICT (kept both):", f)
                continue
            os.remove(f)
            continue
        comp = gzip.compress(data, 6)
        assert gzip.decompress(comp) == data
        with open(gz + ".part", "wb") as fh:
            fh.write(comp)
        os.replace(gz + ".part", gz)
        assert gzip.decompress(open(gz, "rb").read()) == data
        os.remove(f)
        before += len(data)
        after += len(comp)
    print(f"compressed {len(files)} files: {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
