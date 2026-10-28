"""L4 sea states by the frozen rule (pre-registration + A4 (f)): at each duel station, the 30-min sea state with the
largest Hs, and its peak period, Goda peakedness, directional spread at the peak band.

    py -3.13 rogue_waves/tools/duel_seas.py fowd_cdip_029p1_filtered.nc [...]

Reads the Hs column whole (index-based ranges, tools/fowd_count.py), then a few values at the argmax wave through
h5py. Appends to runs/L4/seas.json (never overwrites an entry).
"""
import json
import math
import os
import sys
import time

import aiohttp
import fsspec
import h5py
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fowd_count as fc                                                  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "runs", "L4", "seas.json"))


def find(fname):
    url = fc.BASE + fname
    fs = fsspec.filesystem("https", skip_instance_cache=True,
                           client_kwargs=dict(timeout=aiohttp.ClientTimeout(total=900, sock_connect=30, sock_read=60)))
    with fs.open(url, "rb", block_size=64 * 1024, cache_type="blockcache") as f, h5py.File(f, "r") as h:
        d = h["sea_state_30m_significant_wave_height_spectral"]
        n = d.shape[1]
        cmap = fc.chunk_map(d)
        parts, nb, _ = fc.fetch_runs(url, cmap, requests.Session())
        hs = np.concatenate([fc.inflate(c) for c in parts])[:n]
        hs = np.where(np.isfinite(hs), hs, -1)
        i = int(np.argmax(hs))
        g = lambda name: h[name][0, i]                                   # noqa: E731
        spread = h["direction_dominant_spread_in_frequency_interval"][0, i, :]
        lo, hi = h["meta_frequency_band_lower"][:], h["meta_frequency_band_upper"][:]
        tp = float(g("sea_state_30m_peak_wave_period"))
        fp = 1 / tp
        band = [b for b in range(len(lo)) if lo[b] <= fp < hi[b]]
        rec = dict(file=fname, station=h["meta_station_name"][0].decode() if isinstance(h["meta_station_name"][0], bytes)
                   else str(h["meta_station_name"][0]), wave_index=i, n_waves=n, hs_column_bytes=nb,
                   hs=float(hs[i]), tp=tp, peakedness_goda=float(g("sea_state_30m_bandwidth_peakedness")),
                   steepness=float(g("sea_state_30m_steepness")), bfi_peakedness=float(g("sea_state_30m_benjamin_feir_index_peakedness")),
                   start_time=int(g("sea_state_30m_start_time")), end_time=int(g("sea_state_30m_end_time")),
                   water_depth_m=float(g("meta_water_depth")), band_edges_hz=[[float(a), float(b)] for a, b in zip(lo, hi)],
                   spread_deg_by_band=[float(x) for x in spread], peak_band=band[0] if band else None,
                   spread_deg_at_peak=float(spread[band[0]]) if band else None,
                   read=time.strftime("%Y-%m-%d %H:%M:%S"))
    return rec


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    seas = json.load(open(OUT)) if os.path.exists(OUT) else []
    for fname in sys.argv[1:]:
        if any(s["file"] == fname for s in seas):
            print(fname, "already in seas.json")
            continue
        t0 = time.time()
        r = find(fname)
        seas.append(r)
        json.dump(seas, open(OUT, "w"), indent=1)
        print(json.dumps(r, indent=1), f"\n{time.time() - t0:.0f} s", flush=True)
