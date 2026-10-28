#!/bin/sh
# A5 under a watchdog: rerun the driver until all 152 station files exist (finished stations are skipped).
cd "d:/New folder (5)"
for i in 1 2 3 4 5 6 7 8 9 10; do
  n=$(ls rogue_waves/runs/A5/stations | wc -l)
  [ "$n" -ge 152 ] && break
  py -3.13 rogue_waves/tools/fowd_bigsea.py --workers 6 --budget 80e6 2>&1 | grep -v Warning
  sleep 20
done
echo "loop done: $(ls rogue_waves/runs/A5/stations | wc -l) station files"
