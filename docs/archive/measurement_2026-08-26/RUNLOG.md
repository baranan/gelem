# Run log -- §6.0 measurement pass

**Date:** 26 Aug 2026
**Machine:** Y B's Windows workstation (author's machine, named per §11 instructions)

## Environment

- OS: Microsoft Windows 11 Pro, version 10.0.26200, build 26200, 64-bit
- CPU: Intel(R) Core(TM) i9-14900K, 24 cores / 32 logical processors, 3200 MHz base
- RAM: 127.7 GB total
- Python: 3.13.2 (tags/v3.13.2:4f8bb39, Feb 4 2026) [MSC v.1942 64 bit (AMD64)]
- PyAV: 18.1.0
- `av.library_versions`: libavutil 60.26.102, libavcodec 62.28.102, libavformat 62.12.102,
  libavdevice 62.3.102, libavfilter 11.14.102, libswscale 9.5.102, libswresample 6.3.102
- Fixtures located via `GELEM_FIXTURES` = `C:\Users\baranan\Documents\gelem.videos`
- venv: `C:\Users\baranan\venvs\gelem` (the project's own `(gelem)` venv, per `setup.ps1`).
  `av` and `psutil` were not present in this venv and were installed for this
  measurement pass (`pip install av`, `pip install psutil`); nothing else was
  changed in it, and nothing in the Gelem repository was touched.

All figures reported are **warm-cache**: every measurement unit (defined per
measurement below) was run twice on each file, the first run discarded
entirely, the second reported. The first operation after every container open
was also discarded, separately, per the method rules (decoder warm-up).

## What ran, in order

1. `measure.py` -- full run across all eight fixtures. Wall clock 12:18:08 to
   12:43:23 (~25 minutes). Produced the first `results_raw.json`.
2. **Bug found and fixed: Measurement 3 timing.** The per-frame timer in
   `run_sequential` originally wrapped only the post-decode step (JPEG encode
   or `to_ndarray`), not the `next()` call that triggers the actual libav
   decode. This measured post-processing cost alone, not "cost per sequential
   frame" as the task defines it. Symptom that caught it: the legacy
   3200x1200 file's "analysis" (full-resolution, no-encode) median came back
   at 2.2 ms/frame, indistinguishable from the 1080p files, when a >1.8x
   larger frame should cost visibly more just to memcpy into an ndarray.
   Fixed by moving `t0` before `next(it)`. Reran measurement 3 in full
   (`rerun_m3.py`); `results_raw.json` was updated in place, and
   `measurement_3_rerun_note` records the fix inside it. After the fix, the
   legacy file's analysis cost is clearly separated from the 1080p files
   (~6.6-6.8 ms vs ~5.3-6.7 ms) -- present but smaller than expected; see the
   note in the §11 table about the phone file instead being the outlier.
3. **Bug found and fixed: Measurement 5 load balance.** The
   `different_files` condition built its 30 targets by cycling through the
   four CFR fixtures with period 4 (`cfr_files[i % 4]`), then chunked them for
   the thread pool with `targets[i::n_workers]`. For `n_workers` in `{2, 4}`
   this stride divides the cycle period evenly, so every worker's chunk
   landed on only one or two files instead of a mix -- at 4 workers, one
   thread got *all* of `h264_cfr_gop10s.mp4` (the slowest seek profile in the
   set) while another got all of the cheapest file, and wall clock was
   dominated by the unlucky thread. This is a load-imbalance artifact, not a
   parallelism measurement. Fixed by shuffling the target list before
   chunking (`rng.shuffle`). Reran (`rerun_m5.py`).
4. **Noise found in measurement 5, addressed by repetition.** A single-pass
   rerun of the (now-fixed) measurement 5 gave a same-file 4-worker speedup of
   W=2.14, markedly different from the W=3.31 seen in the original (buggy)
   full run, even though the `same_file` condition was never buggy -- nothing
   about it changed between the two runs. The whole batch is only ~1-3 s of
   work, short enough that thread pool startup, the i9-14900K's P/E-core
   scheduling, and turbo ramp-up plausibly dominate run-to-run variance at
   that timescale. Rather than trust either single number, `rerun_m5_repeats.py`
   fixed the 30 target positions once (same seed) and repeated the *timed*
   part 5 times, reporting median and \[min, max\] of W. Results were tight
   across the 5 repeats (same-file W at 4 workers: 2.00-2.11; different-files:
   1.61-1.75) -- this is the number reported in the final table, not the
   single noisy run. `results_raw.json`'s `measurement_5` was replaced with
   this repeated-run summary; `measurement_5_rerun_note` records both fixes.
5. `analyze.py` -- computed every derived quantity (ratios, k per codec, F0,
   T, G*, K, proxy disk, the P1.7a crossover, the §4.1a consistency check)
   from the final `results_raw.json`. Output transcribed into
   `section11_table.md`.

No other file behaved oddly. All eight open costs were single-digit
milliseconds (page cache already warm from repeated access across the run;
not separately cold-cache tested, which the method rules don't ask for).

## Deviations from the method as written, and why

- **"Run each measurement twice... discard the first" was applied at the
  finest defensible grain**, not literally once for "measurement 1" as a
  monolith: for measurement 1 that's the whole (keyframe-pass +
  full-decode-pass) unit per file; for measurement 2, the whole (scattered +
  sorted, 40 targets each) unit per file; for measurement 3, each
  (start-point x mode) unit. This was a judgement call to keep the run
  finishing in a reasonable time while still discarding a full cold pass
  before every reported number. Flagging it so the choice is visible rather
  than silently baked into the numbers.
- **Measurement 5's repeat-5x correction (step 4 above) goes beyond what the
  task specified** (which asked for total wall clock and W from one pass per
  condition/worker-count). It was necessary because the first two attempts
  disagreed with each other by a wide margin on a claim (does PyAV
  parallelize) that the whole measurement exists to settle. Reporting either
  single number without the repeat would have been reporting noise as
  signal.
- **Measurement 3's "analysis" timing and Measurement 2's `encode_s` field**
  were reused together to compute the P1.7a crossover (not explicitly
  specified how to combine them, since the task only asked for the number).
  The reasoning is in `section11_table.md`.
- The phone file's window for measurement 1 used the **whole file** (0 to
  179.3 s) rather than a 60 s-offset 3-minute slice, per the task's explicit
  instruction for files shorter than that.
- VFR-file "frames per hour" used in the P1.7a crossover for the phone
  recording is approximate (nominal ~29.9 fps average), noted inline where
  used; the two generated VFR/long-GOP fixtures were not part of that
  crossover calculation since it only concerns the three real recordings.

## Files produced

- `results_raw.json` -- every individual timing (final, corrected data; see
  the two `*_rerun_note` fields inside it for what changed and why).
- `measure.py`, `rerun_m3.py`, `rerun_m5.py`, `rerun_m5_repeats.py`,
  `analyze.py`, `smoke_test.py` -- the scripts themselves, kept for
  reproducibility. None of these import Gelem or write into the Gelem
  repository.
- `stdout.log`, `run_log_raw.txt`, `run_log_rerun_m3.txt`,
  `run_log_rerun_m5.txt`, `run_log_m5_repeats.txt` -- raw console logs from
  each run, kept for reference.
- `analysis_summary.json` -- the computed derived quantities, machine-
  readable, backing `section11_table.md`.
- `section11_table.md` -- the deliverable.

All of the above live in `%USERPROFILE%\Documents\gelem.measure\`. Nothing
was written inside the Gelem repository, and no measured number was made
into a constant anywhere -- this run log and the table are prose/data for a
document, not code.
