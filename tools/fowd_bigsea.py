"""A5, EXPLORATORY: the count again, only for waves in big seas (Hs >= 5 m; also >= 8 m). PRODUCTION_LOG.md A5.

    py -3.13 rogue_waves/tools/fowd_bigsea.py [--workers 6] [--budget 80e6]

Per station: the whole Hs column, then H and Tp only for the 10,000-wave chunks holding at least one wave with
Hs >= 5 m (same reader and checks as fowd_count.py, A3). Writes runs/A5/stations/<file>.json; skips finished stations;
console runs/A5/run_A5_console.log. Stop rule as L3 (3 retries, then missing; > 10 % missing big-sea chunks -> stop).
"""
import argparse
import csv
import glob
import json
import os
import sys
import threading
import time

import aiohttp
import fsspec
import h5py
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fowd_count as fc                                                  # noqa: E402

EP = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(EP, "runs", "A5", "stations")
THRESH = (5.0, 8.0)
CH = fc.CHUNK


def read_big(url, n_waves):
    sess = requests.Session()
    info = {}
    with fc.HTTPFile(url) as f, h5py.File(f, "r") as h:                 # own reader, hard timeouts (2026-10-18)
        maps, refs = {}, {}
        for key in ("Hs", "H", "Tp"):
            d = h[fc.COLS[key]]
            assert d.shape == (1, n_waves) and d.chunks == (1, CH) and d.shuffle and d.compression == "gzip"
            maps[key] = fc.chunk_map(d)
            refs[key] = d[0, :min(CH, n_waves)].astype("f4")
    t0 = time.time()
    parts, nb, _ = fc.fetch_runs(url, maps["Hs"], sess)
    hs = np.concatenate([fc.inflate(c) for c in parts])
    del parts
    if not np.array_equal(hs[:len(refs["Hs"])], refs["Hs"], equal_nan=True):
        raise IOError("Hs: reader and h5py disagree on chunk 0")
    hs = hs[:n_waves]
    info["Hs"] = dict(bytes=nb, seconds=round(time.time() - t0, 1))
    big = np.isfinite(hs) & (hs >= THRESH[0])
    n_chunks = len(maps["Hs"])
    sel = [c for c in range(n_chunks) if big[c * CH:(c + 1) * CH].any()]
    out = dict(hs=None, H=None, Tp=None, n_chunks=n_chunks, sel_chunks=len(sel))
    if not sel:
        return out, info
    idx = np.concatenate([np.arange(c * CH, min((c + 1) * CH, n_waves)) for c in sel])
    out["hs"] = hs[idx]
    for key in ("H", "Tp"):
        t0 = time.time()
        sub = [maps[key][c] for c in sel]
        parts, nb, nruns = fc.fetch_runs(url, sub, sess)
        a = np.concatenate([fc.inflate(c) for c in parts])
        if sel[0] == 0 and not np.array_equal(a[:len(refs[key])], refs[key], equal_nan=True):
            raise IOError(f"{key}: reader and h5py disagree on chunk 0")
        # the last chunk may be partly past n_waves: keep the positions in idx only
        starts = np.array(sel) * CH
        lens = np.minimum(starts + CH, n_waves) - starts
        a = np.concatenate([a[i * CH:i * CH + l] for i, l in enumerate(lens)])
        out[key] = a
        info[key] = dict(bytes=nb, requests=nruns, seconds=round(time.time() - t0, 1))
    return out, info


def count_big(d):
    res = {}
    for th in THRESH:
        if d["hs"] is None:
            res[f"hs_ge_{th:g}"] = dict(n_valid=0, over=[0] * len(fc.KS), by_tp_band=[dict(n=0, over=[0] * len(fc.KS))] * 6)
            continue
        ok = np.isfinite(d["H"]) & np.isfinite(d["hs"]) & (d["hs"] >= th)
        ai = d["H"][ok] / d["hs"][ok]
        tp = d["Tp"][ok]
        band = np.digitize(np.where(np.isfinite(tp), tp, 0.0), fc.TP_EDGES)
        srt = np.sort(ai)
        by = []
        for b in range(6):
            sb = np.sort(ai[band == b])
            by.append(dict(tp_centre=6 + 2 * b, n=int(len(sb)),
                           over=[int(x) for x in len(sb) - np.searchsorted(sb, fc.KS, side="right")]))
        res[f"hs_ge_{th:g}"] = dict(n_valid=int(ok.sum()), over=[int(x) for x in len(srt) - np.searchsorted(srt, fc.KS, side="right")],
                                    by_tp_band=by, top3_ai=[round(float(x), 4) for x in srt[-3:][::-1]] if len(srt) else [])
    return res


def run_station(fname, n_waves):
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, fname.replace(".nc", ".json"))
    t0 = time.time()
    for attempt in range(3):
        try:
            d, info = read_big(fc.BASE + fname, n_waves)
            break
        except Exception as e:
            print(f"  {fname}: attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(10)
    else:
        rec = dict(file=fname, status="missing", error="3 failed attempts")
        json.dump(rec, open(out, "w"), indent=1)
        return rec
    rec = dict(file=fname, status="ok", n_waves=n_waves, n_chunks=d["n_chunks"], sel_chunks=d["sel_chunks"], read=info,
               k=fc.KS.tolist(), seconds=round(time.time() - t0, 1), when=time.strftime("%Y-%m-%d %H:%M:%S"), **count_big(d))
    json.dump(rec, open(out, "w"), indent=1)
    return rec


if __name__ == "__main__":
    try:                                                                # keep Windows from idle sleep while running
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget", type=float, default=80e6)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(EP, "data", "cdip_meta", "fowd_stations.csv"))))
    os.makedirs(OUT, exist_ok=True)
    done = {json.load(open(p))["file"] for p in glob.glob(os.path.join(OUT, "*.json"))}
    queue = sorted([r for r in rows if r["file"] not in done], key=lambda r: -int(r["n_waves"]))
    log = open(os.path.join(os.path.dirname(OUT), "run_A5_console.log"), "a", buffering=1)

    def say(s):
        line = time.strftime("%H:%M:%S ") + s
        print(line, flush=True)
        log.write(line + "\n")

    say(f"A5 start: {len(rows)} stations, {len(done)} done, {len(queue)} to read")
    cond = threading.Condition()
    st = dict(inflight=0, running=0, finished=len(done), missing=0, mb=0.0)
    t0 = time.time()

    def worker(r):
        n = int(r["n_waves"])
        try:
            rec = run_station(r["file"], n)
        except Exception as e:
            rec = dict(file=r["file"], status="missing", error=str(e)[:200])
            json.dump(rec, open(os.path.join(OUT, r["file"].replace(".nc", ".json")), "w"), indent=1)
        with cond:
            last["t"] = time.time()
            st["inflight"] -= n
            st["running"] -= 1
            st["finished"] += 1
            if rec["status"] == "ok":
                mb = sum(v["bytes"] for v in rec["read"].values()) / 1e6
                st["mb"] += mb
                msg = (f"{rec['file']}: big-sea waves {rec['hs_ge_5']['n_valid']:,} (chunks {rec['sel_chunks']}/{rec['n_chunks']}), "
                       f"{mb:.0f} MB, {rec['seconds']:.0f} s")
            else:
                st["missing"] += 1
                msg = f"{r['file']}: MISSING"
            say(f"[{st['finished']}/{len(rows)}] {msg} | {st['mb'] / 1e3:.2f} GB in {(time.time() - t0) / 60:.0f} min")
            cond.notify_all()

    last = dict(t=time.time())                                          # watchdog: no station for 15 min -> exit 3,

    def watchdog():                                                     # the wrapper loop restarts (finished kept)
        while True:
            time.sleep(30)
            with cond:
                idle = time.time() - last["t"]
                busy = st["running"] > 0
            if busy and idle > 900:
                say(f"WATCHDOG: no station finished for {idle / 60:.0f} min; exiting for a restart")
                log.flush()
                os._exit(3)

    threading.Thread(target=watchdog, daemon=True).start()
    threads = []
    while queue:
        with cond:
            while True:
                pick = None
                if st["running"] < a.workers:
                    for r in queue:
                        if st["running"] == 0 or st["inflight"] + int(r["n_waves"]) <= a.budget:
                            pick = r
                            break
                if pick is not None:
                    break
                cond.wait()
            queue.remove(pick)
            st["inflight"] += int(pick["n_waves"])
            st["running"] += 1
        th = threading.Thread(target=worker, args=(pick,))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    say(f"A5 end: {st['finished']}/{len(rows)}, missing {st['missing']}, {st['mb'] / 1e3:.2f} GB, {(time.time() - t0) / 60:.0f} min")
