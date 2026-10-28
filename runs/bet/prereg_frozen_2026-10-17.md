## Pre-registration (experiment episodes; written BEFORE any run; date every change, never edit silently)

Written: 2026-10-17. **FROZEN 2026-10-17 01:36** (your "yes" at the P4 gate). From now on, changes only as dated
amendments below the text; the frozen copy and its SHA-256 are in `runs/bet/`. No wave value of the data below has been read: only file names, sizes, station metadata (depth,
position, number of waves) and the column names (`data/cdip_meta/`).

**A change from the P1 plan, said first.** `RESEARCH.md` planned to count raw CDIP buoy records ourselves. CDIP's
server answers 403 Forbidden to this PC and to an outside fetcher, on every page including its homepage
(2026-10-17). The count therefore runs on **FOWD-CDIP v0.5.2, filtered** (Häfner, Gemmrich & Jochum, *J. Atmos.
Oceanic Technol.* 2021; ERDA archive, "free for public use", cite the paper and acknowledge CDIP): every wave the
CDIP buoys recorded up to November 2020, already cut into waves and quality-controlled by FOWD's published rules.
That makes our count a **re-count of a public catalogue with our own rule**, not an independent processing of raw
records; the video says so. If CDIP answers again before P7, an extra labelled run counts raw records at 3
stations (the ones in Level 4) to check FOWD's processing; it never replaces the main count.

- **Tool validation (Level 1; no filming until all pass; max 3 documented fix attempts, then I report to you):**
  1. *Dice line.* Our wave simulator + zero-crossing counter on a simulated narrow-band linear sea (Gaussian
     spectrum, relative width 2 %, ≥ 3 million waves, seeds 0-9): P(H > k·Hs) within ±5 % of exp(−2k²) at
     k = 1.0 and 1.5, and the count at k = 2.0 inside the 95 % Poisson interval of exp(−8) = 3.35 × 10⁻⁴ (formula:
     Häfner et al. 2023, p. 7; Gemmrich & Cicon 2022, p. 1).
  2. *Broad sea.* The same on a JONSWAP sea (γ = 3.3): P(H > 1.5·Hs) within ±10 % of the Naess (1985)
     approximation exp(−4k²/(1+ρ)), ρ = the autocorrelation minimum of the simulated spectrum (☐ get Naess 1985).
  3. *The special mechanism's solver.* Our 1D nonlinear Schrödinger (NLS) solver reproduces the Peregrine
     breather's peak at 3.00× its background within 1 %, and the Benjamin-Feir growth rate of the most unstable
     side-band within 5 % of linear theory; the 2D solver conserves wave energy within 0.1 % over a run.
- **Data:** FOWD-CDIP v0.5.2 filtered, all 152 station files (`data/cdip_meta/fowd_index_filtered.txt`; station
  table from file metadata only, `data/cdip_meta/fowd_stations.csv`: all 152 readable, 1,383,570,225 waves in total,
  depths 7-4,253 m, sampling 1.28-2.56 Hz; 93 stations with ≥ 10⁶ waves; 42 stations ≥ 200 m deep), used as
  published (FOWD's QC flags a-g and its sea-state filters: Hs ≥ 1 m, mean period ≥ 5 s, low-frequency energy
  ≤ 10 %; FOWD paper pp. 6-9). No station is added or dropped after reading any wave value. Read remotely: only
  the columns named below. **Data check (Level 2):** pooled number of waves within ±10 % of "about 1.4 billion"
  (FOWD paper, p. 9) (already known from the file sizes: 1,383,570,225, inside the band; said openly, it is not
  a wave value); pooled share over 2 Hs within ±50 % of about 100,000 in about 1.5 billion = 6.7 × 10⁻⁵
  (Häfner et al. 2021, p. 1; that used v0.4.4, so the band is wide). Outside either band: one documented look for
  the reason, no tuning, reported.
- **Levels:**
  - L1 tool validation (above).
  - L2 data check (above).
  - **L3 the count (the headline).** For every wave: AI = H / Hs, H = `wave_height` (zero-crossing), Hs =
    `sea_state_30m_significant_wave_height_spectral` (Häfner et al. 2021, p. 2: the same definition). Counts of
    AI > k for k = 1.0 to 2.5 in steps of 0.05, per station and pooled; the crest version (crest / Hs > 1.25,
    `wave_crest_height`) reported beside it, not as the headline.
  - **L4 the duel.** Three real sea states, chosen by a rule fixed now: the 3 stations with water depth ≥ 200 m (depth
    of the file's first record) and the most waves; at each, the one 30-min sea state with the largest Hs. From
    the station table: CDIP 029p1 (571 m), 094p1 (319 m), 157p1 (366 m). Each becomes a model sea with the same
    Hs, peak period (`sea_state_30m_peak_wave_period`), peakedness (`sea_state_30m_bandwidth_peakedness`, matched
    by the JONSWAP γ that gives the same Goda peakedness) and directional spread at the peak band (`direction_
    dominant_spread_in_frequency_interval`, matched by cos^2s spreading). Three runs on each:
    **A** waves adding up (linear, random phases, directional);
    **B** the special mechanism on (2D NLS from the same start, 50 peak periods);
    **C** the special mechanism's best case (1D NLS, all energy in one direction, 50 peak periods).
    50 seeds each (seeds 0-49); 64 virtual buoys per run record the last 20 peak periods; same zero-crossing
    counter and AI as L3.
- **Measure:** the share of waves with H > 2·Hs (the rogue rate, per wave, at one point), with a 95 % Wilson
  interval; set against the dice line exp(−8) = 3.35 × 10⁻⁴.
- **What counts as a result:**
  - L3 **below the dice line**: the pooled rate's upper 95 % bound < 3.35 × 10⁻⁴ AND at least two thirds of the
    stations with ≥ 10⁶ waves sit below 3.35 × 10⁻⁴. **On the line**: the interval includes it. **Above**: the
    lower bound > 3.35 × 10⁻⁴. The twist ("the sea makes fewer monsters than dice") may be said only for "below".
  - L4, per sea state: **the special mechanism matters** if B/A (or C/A) ≥ 2 with non-overlapping 95 % intervals;
    **a small correction** if the ratio < 1.5 with overlapping intervals; otherwise **unclear**, said as such.
- **Runs:** L1 seeds 0-9; L4 seeds 0-49 per run; numpy default_rng(seed); code, versions and every output kept in
  `runs/` (failures included). Small test first: L3 on one station, L4 on one seed.
- **Stop rules:** a station that fails to read after 3 retries is listed as missing, never replaced; if more than
  10 % of waves are missing, stop and report. Network budget: stop and ask you if the transfer passes 60 GB
  (estimate ☐ from the size check). Compute budget for L4: 12 hours in total; I ask before any run over a few
  minutes (small test first).
- **Not tuned:** the thresholds (2.0, and 2.2 as a secondary), the Hs definition, FOWD's QC as published, the
  station list (all 152), the L4 sea-state rule, the spectrum-matching rules, the NLS settings (taken from the sea
  state, never adjusted to make waves), the seeds.
- **Wording rules for screen:** "in 152 CDIP buoy locations around the US coasts and territories, up to November
  2020" (☐ the exact first year from the data at P7), "about 1.4 billion waves" (1,383,570,225), "per wave, at one point"; never "the ocean" alone for our count. "Chance" =
  waves adding up, with their real shape as measured. The NLS runs are a model of the special mechanism, not the
  ocean. The count was cut into waves by FOWD (credit Häfner, Gemmrich & Jochum); our rule, our count. Never
  "proves"; "in this count", "in these runs".
- **The bet (C4, drawn on the chart before any data value is read; your choice "do it with manim": the Manim scene
  `visuals/manim/bet.py` draws it in pen style; at the freeze it renders with the time and this pre-registration's
  SHA-256, and the video's own SHA-256 goes in `runs/bet/`):**
  the pooled rate between 4 × 10⁻⁵ and 1 × 10⁻⁴ (under the dice line by 3 to 8 times); at least 80 % of stations
  below the line; in the duel, B/A < 1.5 (a small correction) and C/A ≥ 2 (the special mechanism only wins in its
  best case). My bet leans on the published numbers (Häfner et al.); it can still be wrong.
