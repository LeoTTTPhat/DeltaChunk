"""Stream the single LZMA2 entry of a single-folder .7z archive to stdout (bsdtar 3.5.3 rejects the simplewiki
history archive's entry name). The 7z header is parsed with py7zr; decompression is the stdlib lzma raw LZMA2 decoder.
usage: python -m deltachunk.data.sevenzip_cat <file.7z>"""
import lzma, sys
import py7zr


def lzma2_dict_size(prop: int) -> int:
    if prop == 40:
        return 0xFFFFFFFF
    return (2 | (prop & 1)) << (prop // 2 + 11)


def main(path: str, out=None):
    out = out or sys.stdout.buffer
    z = py7zr.SevenZipFile(path)
    mi = z.header.main_streams
    folders = mi.unpackinfo.folders
    assert len(folders) == 1 and len(folders[0].coders) == 1 and folders[0].coders[0]["method"] == b"!", "expect one LZMA2 coder"
    start = 32 + mi.packinfo.packpos
    size = mi.packinfo.packsizes[0]
    dec = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2,
                                "dict_size": lzma2_dict_size(folders[0].coders[0]["properties"][0])}])
    z.close()
    with open(path, "rb") as f:
        f.seek(start); left = size
        while left > 0:
            buf = f.read(min(1 << 22, left)); left -= len(buf)
            if not buf:
                break
            out.write(dec.decompress(buf))
    out.flush()


if __name__ == "__main__":
    main(sys.argv[1])
