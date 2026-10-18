#!/usr/bin/python3
"""Convert a raw RGBA dump (<prefix>.rgba + <prefix>.txt with "width height") to PNG, stdlib only."""
import struct
import sys
import zlib

prefix = sys.argv[1]
w, h = map(int, open(prefix + '.txt').read().split())
raw = open(prefix + '.rgba', 'rb').read()
rows = b''.join(b'\x00' + raw[y * w * 4:(y + 1) * w * 4] for y in range(h))
chunk = lambda t, d: struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(rows, 6)) + chunk(b'IEND', b'')
open(prefix + '.png', 'wb').write(png)
print(prefix + '.png', w, h)
