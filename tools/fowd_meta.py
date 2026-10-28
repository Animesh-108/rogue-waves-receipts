"""Station table from FOWD-CDIP file metadata only (no wave values are read): number of waves, water depth,
position, sampling rate, file size. Remote HTTP range reads (h5py + fsspec), so only small pieces are fetched.

    py -3.13 rogue_waves/tools/fowd_meta.py rogue_waves/data/cdip_meta/fowd_index_filtered.txt rogue_waves/data/cdip_meta/fowd_stations.csv
"""
import csv
import sys
import time

import fsspec
import h5py

index, out = sys.argv[1], sys.argv[2]
urls = [u.strip() for u in open(index) if u.strip()]
fs = fsspec.filesystem("https", block_size=256 * 1024)
rows, t0 = [], time.time()
for i, url in enumerate(urls, 1):
    try:
        with fs.open(url, "rb") as f, h5py.File(f, "r") as h:
            n = h["wave_height"].shape[1]
            row = dict(file=url.rsplit("/", 1)[1], station=h["meta_station_name"][0].decode() if isinstance(
                h["meta_station_name"][0], bytes) else str(h["meta_station_name"][0]), n_waves=n,
                water_depth_m=float(h["meta_water_depth"][0, 0]), lat=float(h["meta_deploy_latitude"][0, 0]),
                lon=float(h["meta_deploy_longitude"][0, 0]), sampling_hz=float(h["meta_sampling_rate"][0, 0]),
                size_bytes=fs.size(url))
    except Exception as e:                                     # keep going; the failure is listed in the table
        row = dict(file=url.rsplit("/", 1)[1], station="", n_waves="", water_depth_m="", lat="", lon="",
                   sampling_hz="", size_bytes="", error=str(e)[:120])
    rows.append(row)
    print(f"{i}/{len(urls)} {row['file']} n={row['n_waves']} depth={row['water_depth_m']} {row.get('error', '')}",
          flush=True)
keys = ["file", "station", "n_waves", "water_depth_m", "lat", "lon", "sampling_hz", "size_bytes", "error"]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in keys})
print(f"done in {time.time() - t0:.0f} s -> {out}")
