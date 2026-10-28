"""Level 4, the full duel (pre-registration + A4): 3 sea states x 50 seeds x arms A, B, C, C0.

    py -3.13 rogue_waves/runs/L4/run_L4.py

First finds seas 2 and 3 by the frozen rule (tools/duel_seas.py: 094p1, 157p1), then runs every (sea, seed, arm)
not yet in runs.jsonl (the small test, sea 1 seed 0, counts as is). Console: run_L4_console.log next to this file.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EP = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(EP, "tools"))
import duel                                                              # noqa: E402

log = open(os.path.join(HERE, "run_L4_console.log"), "a", buffering=1)


def say(s):
    line = time.strftime("%H:%M:%S ") + s
    print(line, flush=True)
    log.write(line + "\n")


seas_path = os.path.join(HERE, "seas.json")
for fname in ("fowd_cdip_094p1_filtered.nc", "fowd_cdip_157p1_filtered.nc"):
    seas = json.load(open(seas_path))
    if not any(s["file"] == fname for s in seas):
        say(f"finding the sea state at {fname}")
        for attempt in range(3):
            r = subprocess.run([sys.executable, os.path.join(EP, "tools", "duel_seas.py"), fname], capture_output=True, text=True)
            if r.returncode == 0:
                break
            say(f"  attempt {attempt + 1} failed: {r.stderr[-300:]}")
            time.sleep(10)
seas = json.load(open(seas_path))
for i, s in enumerate(seas):
    p = duel.sea_params(s)
    say(f"sea {i + 1}: {s['station']} Hs {s['hs']:.2f} m, Tp {s['tp']:.1f} s, gamma {p['gamma']:.2f}, s {p['s']:.1f}, "
        f"depth {s['water_depth_m']:.0f} m, kp*depth {p['k0'] * s['water_depth_m']:.1f}")
done = set()
runs_path = os.path.join(HERE, "runs.jsonl")
if os.path.exists(runs_path):
    for line in open(runs_path):
        r = json.loads(line)
        done.add((r["sea"], r["seed"], r["arm"]))
t0 = time.time()
todo = [(i, seed, arm) for seed in range(50) for i in range(len(seas)) for arm in ("A", "C0", "C", "B")
        if (seas[i]["station"], seed, arm) not in done]
say(f"duel: {len(todo)} runs to do ({len(done)} done)")
for n, (i, seed, arm) in enumerate(todo, 1):
    r = duel.run(seas[i], seed, arm)
    k2 = r["k"].index(2.0)
    if arm == "B" or n % 20 == 0:
        el = time.time() - t0
        say(f"[{n}/{len(todo)}] sea {i + 1} seed {seed} {arm}: {r['n_waves']:,} waves, over 2x {r['over'][k2]}, "
            f"drift {r['energy_drift']:.0e}, {r['seconds']} s | {el / 60:.0f} min, ~{el / n * (len(todo) - n) / 60:.0f} min left")
say(f"duel done in {(time.time() - t0) / 60:.0f} min")
