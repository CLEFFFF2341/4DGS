# Reserved real clips, declared before candidate selection

2026-10-26. Download preparation may run during development. Do not inspect
candidate metrics on these clips before freezing a confirmation candidate.

- DSEC zurich_city_05_a, frames 120--143.
- DSEC zurich_city_08_a, frames 120--143.
- DSEC zurich_city_09_a, frames 260--283 (new window of an already used sequence;
  not an independent capture).

Use seeds 3,4,5, the same 360x270 train-only stereo initialization and fixed
timestamp split as v2, 32,768 Gaussians and identical per-seed warmup for all arms.
Download original stereo PNGs, calibration and measured exposures with the
existing CRC/SHA256 downloader. No generated blurry targets.

These names and their measured-exposure files are listed by the official DSEC
download page: https://dsec.ifi.uzh.ch/dsec-datasets/download/ . The existing
downloader resolves the corresponding training archive URLs.

Any acquisition failure is recorded. Do not replace a scene based on candidate
metrics. Freeze the exact candidate/control/budget protocol separately before
running these final comparisons. If confirmation fails and prompts revisions,
these clips become development and a later confirmation needs new clips.
