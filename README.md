# Rogue waves: the receipts

Everything behind the Tensor108 video **"I Made a Monster Wave Out of Pure Chance"**: the rules and the bet frozen
before a single real wave was read, every run (the failed one too), the code, and every number the video shows.

**The question:** are monster waves (more than twice the sea's typical big wave) more common than chance alone
predicts?

**The answer, in this count:** 1,383,570,225 real waves from 152 CDIP buoys around the US coasts and territories
(1993 to November 2020) hold 82,058 monsters: one in every 16,861 waves. Chance worked out for real seas (the "fair
line") predicts one in 16,684 (ratio 0.99). The simplest odds (the "dice line", exp(-8)) predict one in 2,981, almost
six times more. My bet, frozen before the count, was one in 10,000 to 25,000: it held.

## Check the bet was made before the count

`runs/bet/` holds the pre-registration and the bet video exactly as frozen on **2026-10-17 at 01:36**, with their
SHA-256 sums. To check nothing changed since:

```
cd runs/bet
sha256sum -c SHA256SUMS.txt
```

All four lines should read `OK` (the pre-registration, the bet video, and the script and stamp that drew it).
Changes after freezing were made only as dated amendments below the frozen text.

## What is where

| Folder / file | What |
|---|---|
| `runs/bet/` | the frozen pre-registration (method, thresholds, stop rules, the bet) and the frozen bet video |
| `runs/L1/` | Level 1: tool checks on simulated seas (the dice line, a broad sea, the solver). The first check **failed** (1,208 monsters counted where 1,407 were expected, outside the margin set before the run); its diagnosis and the re-test are here |
| `runs/L3/` | Level 3: the count over every CDIP buoy (per-station results in `stations/`) |
| `runs/L4/` | Level 4: the storm duel, the stormiest half hour at three deep-water buoys run with and without the "trick" (nonlinear focusing), and its best case (every wave in one direction) |
| `runs/A5/` | an exploratory check added **after** the main result: only big seas (labelled exploratory in the video) |
| `tools/` | the code: the wave simulator, the nonlinear Schrödinger solver, the FOWD counter, the duel, and the renderers of the video's sea and dice table |
| `visuals/manim/` | the script that drew the frozen bet video |
| `numbers.json` | every number shown or said in the video, each traceable to a run file or a source (`_sources`) |
| `script/script.facts.md` | the facts lock: every claim in the script, its source, and the review notes |

## The data

The count runs on **FOWD-CDIP v0.5.2, filtered** (Häfner, Gemmrich & Jochum 2021, *FOWD: A Free Ocean Wave Dataset
for Data Mining and Machine Learning*, https://arxiv.org/abs/2011.12071; ERDA archive, free for public use). It is
not copied here: download it from its archive to re-run `runs/L3/run_L3.py`. Buoy data courtesy of CDIP
(DOI 10.18437/C7WC72). This is a re-count of a public catalogue with our own rule, not an independent processing of
the raw buoy records.

## Limits (what this does not show)

These are US buoys, many close to the coast; each sees a single point of the sea; the waves were cut out by someone
else's careful method (FOWD); and the fair line and the storm runs are models, not the ocean.

## Sources

- Haver, S. (2004). A possible freak wave event measured at the Draupner Jacket January 1 1995. Rogue Waves 2004, Brest (Ifremer, 2005).
- ESA (2004). Ship-sinking monster waves revealed by ESA satellites. https://www.esa.int/Applications/Observing_the_Earth/Ship-sinking_monster_waves_revealed_by_ESA_satellites
- Häfner, Gemmrich & Jochum (2021). Real-world rogue wave probabilities. Scientific Reports. https://doi.org/10.1038/s41598-021-89359-1
- Häfner, Gemmrich & Jochum (2023). Machine-guided discovery of a real-world rogue wave model. PNAS. https://arxiv.org/abs/2311.12579
- Gemmrich & Cicon (2022). Generation mechanism and prediction of an observed extreme rogue wave. Scientific Reports. https://doi.org/10.1038/s41598-022-05671-4
- Dudley et al. (2019). Rogue waves and analogies in optics and oceanography. Nature Reviews Physics. https://doi.org/10.1038/s42254-019-0100-0
- Onorato et al. (2001). Freak waves in random oceanic sea states. Physical Review Letters. https://arxiv.org/abs/nlin/0104055
