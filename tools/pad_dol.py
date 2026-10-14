#!/usr/bin/env python3
"""
Pad a .dol's sections to multiples of 32 bytes, in place.

    python3 tools/pad_dol.py iiama2.dol

Dolphin 2606 and later refuse to boot a DOL whose section sizes aren't 32-byte
aligned, and devkitPPC's elf2dol doesn't pad them (whether a build trips this
depends on code size). Real-hardware loaders don't care. The padding is zeros
and is checked not to land on a section that loads earlier. Stdlib only.
"""
import struct
import sys


def pad(path):
    with open(path, "rb") as f:
        data = f.read()
    hdr = list(struct.unpack(">64I", data[:0x100]))
    offs, addrs, sizes = hdr[0:18], hdr[18:36], hdr[36:54]  # 7 text + 11 data sections each

    body = bytearray()
    new_offs, new_sizes = [0] * 18, [0] * 18
    for i in range(18):
        if sizes[i] == 0:
            continue
        if addrs[i] % 32:
            sys.exit(f"{path}: section {i} starts at {addrs[i]:#x}, not 32-byte aligned; padding can't fix that")
        padded = -(-sizes[i] // 32) * 32
        lo, hi = addrs[i] + sizes[i], addrs[i] + padded
        for j in range(i):  # sections load in header order, so only earlier ones could be clobbered
            if sizes[j] and lo < addrs[j] + sizes[j] and addrs[j] < hi:
                sys.exit(f"{path}: padding section {i} would overwrite section {j}")
        new_offs[i] = 0x100 + len(body)
        new_sizes[i] = padded
        body += data[offs[i]:offs[i] + sizes[i]] + bytes(padded - sizes[i])

    hdr[0:18], hdr[36:54] = new_offs, new_sizes
    with open(path, "wb") as f:
        f.write(struct.pack(">64I", *hdr) + body)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        pad(p)
