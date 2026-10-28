"""Level 3, the count: per station, how many waves are taller than k x Hs (pre-registration + A3, PRODUCTION_LOG.md).

    py -3.13 rogue_waves/tools/fowd_count.py STATION_FILE [STATION_FILE ...] [--workers N]

For each FOWD-CDIP filtered file (ERDA), reads 4 columns only: wave_height (H), sea_state_30m_significant_wave_
height_spectral (Hs), wave_crest_height (C), sea_state_30m_peak_wave_period (Tp). Each column's chunks lie back to
back in the file (each chunk = a zlib stream + 4 bytes), so the exact (offset, size) of every chunk comes from the
HDF5 index, back-to-back chunks are merged into range requests of <= 16 MB, and every chunk is checked (zlib header,
10,000 float32 after unshuffle, ceil(n / 10,000) chunks, chunk 0 equal to h5py's own read) (A3 (a)). Writes runs/L3/stations/<file>.json; never overwrites
a finished station (every run is kept: a re-read goes to a new file name with a timestamp).
"""
import argparse
import io
import concurrent.futures as cf
import json
import math
import os
import sys
import time
import zlib

import fsspec
import h5py
import numpy as np
import requests

BASE = "https://sid.erda.dk/public/archives/969a4d819822c8f0325cb22a18f64eb8/Ocean/fowd-out-v5/"
EP = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(EP, "runs", "L3", "stations")
COLS = dict(H="wave_height", Hs="sea_state_30m_significant_wave_height_spectral", C="wave_crest_height",
            Tp="sea_state_30m_peak_wave_period")
KS = np.round(np.arange(1.0, 2.5001, 0.05), 2)                           # thresholds k = 1.0 .. 2.5, step 0.05
TP_EDGES = [7.0, 9.0, 11.0, 13.0, 15.0]                                  # bands centred on 6, 8, ..., 16 s (A3 (c))
CHUNK = 10_000


class HTTPFile(io.RawIOBase):
    """Read-only remote file for h5py: Range requests with hard timeouts, a 64 KB block cache, and up to 6 retries
    on a fresh connection (replaces fsspec, whose connections hung when ERDA closed them; 2026-10-18)."""
    BLOCK = 64 * 1024

    def __init__(self, url):
        self.url, self.pos, self.cache = url, 0, {}
        self.sess = requests.Session()
        r = self._get({"Range": "bytes=0-0"})
        self.size = int(r.headers["Content-Range"].split("/")[1])

    def _get(self, headers):
        for attempt in range(6):
            try:
                r = self.sess.get(self.url, headers=headers, timeout=(10, 30))
                if r.status_code not in (200, 206):
                    raise IOError(f"status {r.status_code}")
                return r
            except Exception:
                if attempt == 5:
                    raise
                self.sess = requests.Session()
                time.sleep(3 * (attempt + 1))

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            if len(self.cache) > 4096:
                self.cache.clear()
            o0 = i * self.BLOCK
            o1 = min(o0 + self.BLOCK, self.size) - 1
            self.cache[i] = self._get({"Range": f"bytes={o0}-{o1}"}).content
        return self.cache[i]

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        out, got = memoryview(b), 0
        while got < n:
            i, off = divmod(self.pos + got, self.BLOCK)
            blk = self._block(i)
            take = min(len(blk) - off, n - got)
            out[got:got + take] = blk[off:off + take]
            got += take
        self.pos += n
        return n


def unshuffle(raw, itemsize=4):
    return np.frombuffer(raw, np.uint8).reshape(itemsize, -1).T.copy().view("<f4").ravel()


def chunk_map(d):
    """(coord, byte offset, size) of every chunk of a (1, n) dataset, from the HDF5 index."""
    out = []
    d.id.chunk_iter(lambda ci: out.append((ci.chunk_offset[1], ci.byte_offset, ci.size)))
    return sorted(out)


def fetch_runs(url, cmap, session, max_bytes=16 << 20):
    """Merge back-to-back chunks into range requests (<= 16 MB); return the chunks' bytes in coord order, bytes read."""
    runs, cur = [], [cmap[0]]
    for c in cmap[1:]:
        last = cur[-1]
        if c[1] == last[1] + last[2] and (c[1] + c[2] - cur[0][1]) <= max_bytes:
            cur.append(c)
        else:
            runs.append(cur)
            cur = [c]
    runs.append(cur)
    parts, nbytes = [], 0
    for run in runs:
        o0, o1 = run[0][1], run[-1][1] + run[-1][2] - 1
        size = o1 - o0 + 1
        deadline = max(180.0, size / 30e3)                              # a whole range must arrive at >= 30 kB/s
        for attempt in range(5):
            try:
                t_start = time.time()
                with session.get(url, headers={"Range": f"bytes={o0}-{o1}"}, timeout=(30, 60), stream=True) as r:
                    if r.status_code != 206:
                        raise IOError(f"range {o0}-{o1}: status {r.status_code}")
                    buf = bytearray()
                    for piece in r.iter_content(chunk_size=1 << 18):
                        buf += piece
                        if time.time() - t_start > deadline:
                            raise IOError(f"range {o0}-{o1}: slower than 30 kB/s, retried")
                if len(buf) != size:
                    raise IOError(f"range {o0}-{o1}: {len(buf)} bytes")
                break
            except Exception:
                if attempt == 4:
                    raise
                session = requests.Session()                            # fresh connection
                time.sleep(5)
        b = bytes(buf)
        nbytes += len(b)
        for coord, off, size in run:
            parts.append(b[off - o0:off - o0 + size])
    return parts, nbytes, len(runs)


def inflate(chunk):
    if chunk[0] != 0x78:
        raise IOError("no zlib header")
    d = zlib.decompressobj()
    raw = d.decompress(chunk)
    if not d.eof or len(raw) != 4 * CHUNK:
        raise IOError(f"inflate: eof {d.eof}, {len(raw)} bytes")
    return unshuffle(raw)


def read_columns(url, n_waves):
    """The 4 columns of one station: chunk maps from the HDF5 index (one at a time), then the 4 columns fetched at
    the same time (one connection each), then inflated and checked against h5py's own read of chunk 0."""
    import aiohttp                                                      # timeouts: a dead connection (e.g. a Wi-Fi
    fs = fsspec.filesystem("https", skip_instance_cache=True,           # change) raises and is retried, never hangs
                           client_kwargs=dict(timeout=aiohttp.ClientTimeout(total=900, sock_connect=30, sock_read=60)))
    n_chunks = math.ceil(n_waves / CHUNK)
    cols, info = {}, {}
    with fs.open(url, "rb", block_size=64 * 1024, cache_type="blockcache") as f, h5py.File(f, "r") as h:
        attrs = {k: (h.attrs[k].decode() if isinstance(h.attrs[k], bytes) else str(h.attrs[k]))
                 for k in ("processing_version", "postprocessing", "time_coverage_start", "time_coverage_end")}
        maps, refs = {}, {}
        for key, name in COLS.items():
            d = h[name]
            assert d.shape == (1, n_waves) and d.chunks == (1, CHUNK) and d.shuffle and d.compression == "gzip", name
            cmap = chunk_map(d)
            if len(cmap) != n_chunks or [c[0] for c in cmap] != list(range(0, n_chunks * CHUNK, CHUNK)):
                raise IOError(f"{name}: chunk index has {len(cmap)} chunks, expected {n_chunks}")
            maps[key] = cmap
            refs[key] = d[0, :min(CHUNK, n_waves)].astype("f4")                # self-check reference: chunk 0

    def one(key):
        t0 = time.time()
        parts, nb, nruns = fetch_runs(url, maps[key], requests.Session())
        a = np.concatenate([inflate(c) for c in parts])
        del parts
        if not np.array_equal(a[:len(refs[key])], refs[key], equal_nan=True):
            raise IOError(f"{COLS[key]}: our reader and h5py disagree on chunk 0")
        return key, a[:n_waves], dict(how="index+ranges", chunks=n_chunks, requests=nruns, bytes=nb,
                                      seconds=round(time.time() - t0, 1))

    with cf.ThreadPoolExecutor(len(COLS)) as ex:
        for key, a, inf in ex.map(one, COLS):
            cols[key], info[key] = a, inf
    return cols, info, attrs


def count(cols):
    H, Hs, C, Tp = cols["H"], cols["Hs"], cols["C"], cols["Tp"]
    ok = np.isfinite(H) & np.isfinite(Hs) & (Hs > 0)
    ai = H[ok] / Hs[ok]
    cr = C[ok] / Hs[ok]
    band = np.digitize(np.where(np.isfinite(Tp[ok]), Tp[ok], 0.0), TP_EDGES)          # 0..5 -> 6..16 s
    srt = np.sort(ai)
    over = len(srt) - np.searchsorted(srt, KS, side="right")                          # count of AI > k
    by_band = []
    for b in range(6):
        sb = np.sort(ai[band == b])
        by_band.append(dict(tp_centre=6 + 2 * b, n=int(len(sb)),
                            over=[int(x) for x in len(sb) - np.searchsorted(sb, KS, side="right")]))
    top = np.sort(ai)[-5:][::-1]
    return dict(n_total=int(len(H)), n_valid=int(ok.sum()), n_missing=int((~ok).sum()), k=KS.tolist(),
                over=[int(x) for x in over], over_2_2=int((ai > 2.2).sum()),
                crest_over_1_25=int((cr > 1.25).sum()), by_tp_band=by_band, top5_ai=[round(float(x), 4) for x in top],
                hs_max=round(float(Hs[ok].max()), 2), tp_missing=int((~np.isfinite(Tp[ok])).sum()))


def run_station(fname, n_waves):
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, fname.replace(".nc", ".json"))
    if os.path.exists(out):
        out = out.replace(".json", time.strftime("_%Y%m%d-%H%M%S.json"))
    t0 = time.time()
    for attempt in range(3):                                            # stop rule: 3 retries, then missing
        try:
            cols, info, attrs = read_columns(BASE + fname, n_waves)
            break
        except Exception as e:
            print(f"  {fname}: attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(10)
    else:
        rec = dict(file=fname, status="missing", error="3 failed attempts")
        json.dump(rec, open(out, "w"), indent=1)
        return rec
    rec = dict(file=fname, status="ok", read=info, attrs=attrs, started=time.strftime("%Y-%m-%d %H:%M:%S",
               time.localtime(t0)), seconds=round(time.time() - t0, 1), numpy=np.__version__, h5py=h5py.__version__,
               **count(cols))
    json.dump(rec, open(out, "w"), indent=1)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    import csv
    meta = {r["file"]: int(r["n_waves"]) for r in csv.DictReader(open(os.path.join(EP, "data", "cdip_meta", "fowd_stations.csv")))}
    files = [os.path.basename(f) for f in a.files]
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_station, f, meta[f]): f for f in files}
        for fu in cf.as_completed(futs):
            r = fu.result()
            if r["status"] == "ok":
                mb = sum(v["bytes"] for v in r["read"].values() if v["bytes"] > 0) / 1e6
                print(f"{r['file']}: {r['n_valid']:,} waves, missing {r['n_missing']:,}, {mb:.1f} MB in {r['seconds']} s",
                      flush=True)
            else:
                print(f"{r['file']}: MISSING", flush=True)
