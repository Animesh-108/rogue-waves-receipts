"""Level 3, the full count (pre-registration + A3): every FOWD-CDIP station, our rule.

    py -3.13 rogue_waves/runs/L3/run_L3.py [--workers 6] [--budget 80e6]

Runs tools/fowd_count.py's run_station on all 152 stations, largest first, at most --workers at a time and at most
--budget waves in memory at once (8 GB PC); the largest station that fits starts next (fixed 2026-10-17
12:50 after the first run waited behind the biggest station). A station with a finished file in stations/ is skipped (the small test,
208p1, counts as is: A3 (e)). Stop rule: if waves in missing stations + invalid waves pass 10 % of all waves, no new
station starts and the run reports. Console: run_L3_console.log next to this file.
"""
import argparse
import csv
import glob
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EP = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(EP, "tools"))
import fowd_count as fc                                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--budget", type=float, default=80e6)
a = ap.parse_args()

rows = list(csv.DictReader(open(os.path.join(EP, "data", "cdip_meta", "fowd_stations.csv"))))
total = sum(int(r["n_waves"]) for r in rows)
done = {os.path.basename(p).split("_filtered")[0] for p in glob.glob(os.path.join(fc.OUT, "*.json"))}
todo = sorted([r for r in rows if r["file"].split("_filtered")[0] not in done], key=lambda r: -int(r["n_waves"]))
log = open(os.path.join(HERE, "run_L3_console.log"), "a", buffering=1)


def say(s):
    line = time.strftime("%H:%M:%S ") + s
    print(line, flush=True)
    log.write(line + "\n")


say(f"start: {len(rows)} stations, {total:,} waves; {len(done)} already done; {len(todo)} to read; workers {a.workers}, "
    f"budget {a.budget:,.0f} waves")
lock = threading.Lock()
cond = threading.Condition(lock)
state = dict(inflight=0, running=0, missing=0, bytes=0, finished=len(done), stop=False)
t0 = time.time()


def worker(r):
    n = int(r["n_waves"])
    try:
        rec = fc.run_station(r["file"], n)
    except Exception as e:                                              # never silent: listed as missing
        rec = dict(file=r["file"], status="missing", error=str(e)[:200])
        json.dump(rec, open(os.path.join(fc.OUT, r["file"].replace(".nc", ".json")), "w"), indent=1)
    with cond:
        state["inflight"] -= n
        state["running"] -= 1
        state["finished"] += 1
        if rec["status"] == "ok":
            state["missing"] += rec["n_missing"]
            mb = sum(v["bytes"] for v in rec["read"].values()) / 1e6
            state["bytes"] += mb
            msg = f"{rec['file']}: {rec['n_valid']:,} waves, {mb:.1f} MB, {rec['seconds']:.0f} s"
        else:
            state["missing"] += n
            msg = f"{r['file']}: MISSING ({rec.get('error', '')})"
        el = time.time() - t0
        say(f"[{state['finished']}/{len(rows)}] {msg} | total {state['bytes'] / 1e3:.2f} GB in {el / 60:.0f} min")
        if state["missing"] > 0.10 * total:
            state["stop"] = True
            say(f"STOP RULE: missing waves {state['missing']:,} > 10 % of {total:,}")
        cond.notify_all()


threads = []
queue = list(todo)                                                       # largest first
while queue:
    with cond:
        while True:                                                     # start the largest station that fits
            if state["stop"]:
                break
            pick = None
            if state["running"] < a.workers:
                for r in queue:
                    if state["running"] == 0 or state["inflight"] + int(r["n_waves"]) <= a.budget:
                        pick = r
                        break
            if pick is not None:
                break
            cond.wait()
        if state["stop"]:
            break
        queue.remove(pick)
        state["inflight"] += int(pick["n_waves"])
        state["running"] += 1
    th = threading.Thread(target=worker, args=(pick,), daemon=False)
    th.start()
    threads.append(th)
for th in threads:
    th.join()
say(f"end: {state['finished']}/{len(rows)} stations, {state['bytes'] / 1e3:.2f} GB, {(time.time() - t0) / 60:.0f} min, "
    f"stop={state['stop']}")
