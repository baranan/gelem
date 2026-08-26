## 11. Numbers, and which of them generalise

*Status 26 Aug 2026, measurement pass.* All rows below are **measured 26 Aug
2026 on Y B's machine** (Intel Core i9-14900K, 24C/32T, 128 GB RAM, Windows 11
Pro 10.0.26200, PyAV 18.1.0 / ffmpeg libs per `RUNLOG.md`), replacing the
estimate rows below with measured ones. Every timing is warm-cache: each
measurement unit ran twice per file, first discarded, second reported;
the first operation after every container open was separately discarded.
Full method, raw numbers, and two implementation bugs found and fixed during
this pass are in `RUNLOG.md` and `results_raw.json`.

**Pre-registered prediction, and what actually happened.** The prediction was
that the keyframe-only/full-decode ratio would be roughly `gap_frames / k`
with k between 2 and 5. The measured ratio does track `gap_frames`
closely and consistently across all eight files and three codec families --
**the mechanism is confirmed** -- but the fitted k is **1.1 to 1.35**, well
below the predicted range, meaning the achieved speedup sits much closer to
the theoretical ceiling (proportional to `gap_frames`) than expected. Two
causes were checked, per the method rules, before accepting this:

- **`skip_frame` not taking effect?** No. Keyframe counts recovered in the
  3-minute window match `docs/fixtures.md`'s stated values almost exactly
  (e.g. the phone file: 179 keyframes measured against 179 stated; the
  1 s-GOP pair: 181 measured against an expected 180; the 10 s-GOP pair: 19
  measured against an expected 18).
- **JPEG encoding dominating and swamping the decode difference?** Yes, in
  part, and this is the actual explanation for the low k. Measurement 3's
  display/analysis split shows resize+JPEG-encode is **59-73% of total
  per-frame cost** for the constant-frame-rate files (e.g. h265 1 s-GOP:
  22.4 ms display vs 6.0 ms analysis-only). Because this pass produces
  *identical output* on both the keyframe-only and full-decode sides (as the
  method requires, so the ratio isn't an artifact of unequal work), the
  shared encode cost is paid once per frame on both sides and compresses the
  ratio toward the frame-count ceiling. This is not a measurement error --
  it is the actual cost of the actual pipeline Gelem would run to build a
  proxy -- but it means k is a property of *this* pipeline (decode -> scale
  150 px -> JPEG q80), not of decode alone. A decode-only k would likely sit
  closer to the originally-guessed 2-5.

### Structural, generalises

| Quantity | Measured 26 Aug 2026 (Y B's machine) | Why it generalises |
|---|---|---|
| Keyframe-only vs full decode | ratio ≈ `gap_frames / k`. k = 1.15-1.35 (h264, n=5), 1.10-1.22 (hevc, n=2), 1.17 (mpeg4, n=1). Tracks `gap_frames` consistently across all 8 files. | Skipping inter-frame decode is inherent to how any decoder implements a predictive codec -- the mechanism holds on any machine. The *specific* k value does not generalise: it is bound to this pipeline's encode-dominated cost structure (see above), not to decode cost alone. |
| Seek fixed cost F0 (as a share of total scattered-seek time) | h264: F0 = 23.4 ms = **64%** of seek time on the dense (1 s-GOP) file, **15%** on the sparse (10 s-GOP) file. hevc: F0 = 14.5 ms, same 64%/15% split. | F0 is index lookup + decoder flush/reinit, a structural cost of any seek in any container/decoder. That it dominates for dense-keyframe files and shrinks proportionally for sparse ones is a property of the mechanism. The absolute F0 value (14-23 ms) is machine- and library-specific and will not generalise numerically. |
| Does PyAV release the GIL during decode? | **Yes, confirmed.** W > 1 at every worker count tested (never ≈1.0, which would mean no parallelism). Sub-linear: same-file W = 1.60 (2 workers) / 2.06 (4 workers, of ideal 4.0); different-files W = 1.36 / 1.72 (5-repeat median, tight range -- see `RUNLOG.md`). | Whether the C library releases the GIL during its C-level decode call is a fact about the binding, true on any machine. That scaling is sub-linear, and further reduced when threads decode different files/codecs simultaneously rather than one shared file, is a real, reproduced effect -- but the exact multipliers are this CPU's core count and scheduler, and will not generalise numerically. |
| Sorted vs scattered batching benefit | **0.5-2.9% faster sorted**, one file (h265 1 s-GOP) measured 7.4% *slower* sorted -- within noise. No meaningful benefit observed. | **Does not generalise, and cannot be read as "batching doesn't help."** The method's mandatory warm-cache protocol (discard first pass, report second) removes exactly the disk-I/O locality effect that sequential access is supposed to exploit. This result says the batching benefit is not visible *once the OS page cache is already warm* -- it is silent on the cold-cache / Google-Drive-Streaming case §4.3 is actually written for. A cold-cache variant would be needed to test the claim as stated. |

### Machine and file specific, must not become a constant

| File | Codec | Open cost (median, ms) | Scattered seek (median / IQR / max, ms) | Sorted seek (median, ms) | Sequential display (median / IQR, ms/frame) | Sequential analysis (median / IQR, ms/frame) | Proxy build cost (s/hour) | Proxy size (MB/hour) | T (s) | G\* (frames) |
|---|---|---|---|---|---|---|---|---|---|---|
| sid89_video.mp4 (Zoom) | h264 | 10.0 | 28.4 / [18.4, 31.7] / 57.3 | 27.8 | 20.64 / [18.89, 23.81] | 5.50 / [5.22, 6.42] | 23.8 | 3.95 | 0.41 | 1.4 |
| PID_031...mp4 (legacy) | mpeg4 | 3.2 | 25.2 / [23.8, 27.0] / 33.9 | 24.7 | 30.66 / [29.01, 33.18] | 6.65 / [6.42, 7.18] | 128.3 | 12.0 | 0.37 | 0.8 |
| VID_20260826...mp4 (phone) | h264 | 12.7 | 113.3 / [84.1, 147.5] / 203.5 | 112.7 | 27.32 / [25.72, 30.18] | 15.85 / [13.96, 18.82] | 75.4 | 9.40 | 1.65 | 4.1 |
| h264_cfr_gop1s.mp4 | h264 | 6.5 | 36.8 / [30.6, 40.8] / 67.2 | 36.9 | 22.09 / [20.33, 25.24] | 9.13 / [7.54, 12.01] | 53.4 | 7.06 | 0.54 | 1.7 |
| h264_cfr_gop10s.mp4 | h264 | 7.1 | 157.5 / [94.6, 220.4] / 347.8 | 153.0 | 21.81 / [20.21, 24.96] | 8.06 / [7.10, 10.50] | 6.3 | 0.72 | 2.29 | 7.2 |
| h265_cfr_gop1s.mp4 | hevc | 1.6 | 22.6 / [17.7, 26.7] / 34.2 | 24.3 | 22.40 / [20.30, 26.28] | 6.00 / [5.57, 6.81] | 47.0 | 7.06 | 0.33 | 1.0 |
| h265_cfr_gop10s.mp4 | hevc | 1.7 | 96.1 / [58.4, 130.9] / 186.6 | 95.1 | 22.54 / [20.32, 25.89] | 5.90 / [5.54, 6.54] | 5.2 | 0.72 | 1.40 | 4.3 |
| h264_vfr_longgop.mp4 | h264 (VFR) | 7.4 | 199.3 / [126.0, 266.0] / 386.7 | 195.5 | 21.23 / [19.89, 23.41] | 7.71 / [7.04, 9.34] | 4.3 | 0.47 | 2.90 | 9.4 |

Each depends on: keyframe density and codec (seek and build cost), frame
resolution and bitrate/entropy (sequential decode cost -- see the phone-file
note below), and this machine's single-thread decode speed and disk cache
state (all of it). None of it should become a setting default or a hardcoded
threshold without re-measuring on the machine and files it will actually run
against.

**Unexpected: the phone file, not the legacy file, is the outlier.** The
prediction was that the legacy 3200x1200 recording would differ sharply from
the 1080p files at full resolution. It doesn't -- its analysis-mode cost
(6.65 ms) is close to the Zoom file's (5.50 ms) despite ~1.85x the pixels.
The phone file's analysis-mode cost (15.85 ms) is the actual outlier, roughly
2.5x every other file including the higher-resolution legacy one. The phone
file is also the only one recorded at high bitrate (20 Mbps vs 0.86-2.98 Mbps
for the others) with real, irregular VFR -- entropy/bitrate, not resolution,
looks like the dominant driver of decode cost here, at least on this small
set. Worth another look before anything is built on the resolution-scales-cost
assumption.

### Arithmetic, not measurement

| Quantity | Value |
|---|---|
| Proxy disk, 240 h corpus, if entirely Zoom-like (sid89 density) | 0.93 GB |
| Proxy disk, 240 h corpus, if entirely legacy-like (PID_031 density) | **2.81 GB** |
| Proxy disk, 240 h corpus, if entirely phone-like (VID_20260826 density) | **2.20 GB** |
| Proxy disk, 240 h corpus, if entirely h264 1 s-GOP density | 1.65 GB |
| Proxy disk, 240 h corpus, if entirely h264/h265 10 s-GOP density | 0.17 GB |
| Proxy disk, 240 h corpus, if entirely VFR-longgop density | 0.11 GB |
| K (build cost per hour ÷ scattered seek time), sid89 | 837 |
| K, legacy (PID_031) | 5093 |
| K, phone | 665 |
| K, h264 1 s-GOP | 1451 |
| K, h264 10 s-GOP | 40 |
| K, h265 1 s-GOP | 2079 |
| K, h265 10 s-GOP | 54 |
| K, VFR-longgop | 22 |

K is "how many thumbnails must be drawn from one hour of video before a
prepared cache repays its own construction." For every real recording, K is
in the hundreds to low thousands -- a proxy would need to be reused hundreds
to thousands of times per hour of source video before it pays for itself.
Under the T-driven verdict below this doesn't matter (P1.3 isn't being built
regardless), but if that verdict is revisited later, K is the number that
says the payback bar is high.

The real-corpus proxy-disk figure is not a single number: Y B's actual 240 h
corpus is presumably a mix of these three densities, not uniformly one type,
and its composition wasn't measured. **Two of the three real recording types
(legacy and phone) individually exceed the 2 GB gate on their own; only the
Zoom-density type does not.** A realistic mixed corpus is not comfortably
under the gate -- see the verdict below.

---

## Verdict, stated mechanically

**Rule:** T < 1 s -> not built, comfortable. T in [1, 6] s -> not built,
accepted trade-off. T > 6 s -> reconsider (more workers / reduced-res decode
for mpeg4 / prefetch). Two gates defer P1.3 regardless of T: build cost > 180
s/hour, or 240 h-corpus proxy disk > 2 GB.

| Real recording | T | Band | Build-cost gate (>180 s/h)? | Proxy-disk gate (>2 GB @240h)? |
|---|---|---|---|---|
| sid89_video.mp4 (Zoom) | 0.41 s | **under 1 s** | No (23.8 s/h) | No (0.93 GB) |
| PID_031...mp4 (legacy, mpeg4) | 0.37 s | **under 1 s** | No (128.3 s/h) | **Yes (2.81 GB)** |
| VID_20260826...mp4 (phone) | 1.65 s | **1-6 s** | No (75.4 s/h) | **Yes (2.20 GB)** |

**P1.3 is not built.** All three real recordings land in the "not built"
bands on T alone (two comfortably under 1 s, the phone recording in the
accepted 1-6 s trade-off band). The disk gate independently confirms this for
two of the three recording types, and is ambiguous rather than clearly clear
for a realistic mixed corpus (see above) -- it does not change the verdict,
since T already settles it, but it removes any temptation to treat "T is
fine" as the last word if T is revisited later on a different machine.

No band is close enough to a boundary to flag as fragile except the phone
file's T = 1.65 s, which sits inside the 1-6 s band regardless of the exact
value -- moving W (the parallelism speedup used in T's denominator) between
the two measured values in this pass (2.06 vs the earlier noisy 3.31) moved T
between 1.03 s and 1.65 s without changing the band. The verdict is robust to
that noise; a specific point estimate of T for the phone file is not.

---

## Contradictions with the plan, reported as instructed

**P1.7a (trial-thumbnail batch job) is contradicted, sharply, for all three
real recordings.** The crossover -- how many trials per hour of video before
a full sequential decode pass beats sorted per-trial seeking -- comes out at:

| File | Full sequential pass (s/hour, decode-only) | Sorted seek, seek-only component (ms) | Crossover N\* (trials/hour) |
|---|---|---|---|
| sid89_video.mp4 | 494.7 | 17.55 | **~28,200** |
| PID_031...mp4 (legacy) | 478.7 | 8.16 | **~58,600** |
| VID_20260826...mp4 (phone) | 1705.8 | 101.68 | **~16,800** |

(Method: sorted-seek total time was split into its seek-only and JPEG-encode
components using measurement 2's own per-operation fields, since a
sequential pass only has to pay the encode cost at the N representative
frames it actually captures, not at every frame it decodes through. N\* =
full-pass decode-only cost ÷ (sorted-seek time − encode-only time). Full
detail in `analyze.py`.)

No real study has 16,000-59,000 trials in one hour of video. **For any
realistic trial count, sorted per-trial seeking beats a full sequential
decode by two to three orders of magnitude on this corpus.** P1.7a's design
-- decode the whole video once, in trial order, rather than seek to each
trial -- rests on the same "seeking is expensive" assumption that motivated
the proxy layer, and the same measurement that weakens the case for P1.3
weakens it here too, more directly: this isn't a case for reconsidering
P1.7a, it's numbers loud enough to ask whether it should be built as
currently specified at all.

**§4.1a's dense-keyframes-get-a-proxy rule does not track which files
actually have expensive seeks.** The rule sorts files by keyframe gap and
expects seek cost to follow. It doesn't, for the three real recordings:

| File | Keyframe gap | Scattered seek median |
|---|---|---|
| PID_031 (legacy) | 12 frames (0.6 s) -- densest | 25.2 ms -- cheapest |
| sid89 (Zoom) | 50 frames (2.0 s) -- sparsest of the three | 28.4 ms -- second cheapest |
| VID_20260826 (phone) | 30 frames (~1.0 s) -- in between | 113.3 ms -- **4x more expensive than either** |

The phone file has a *smaller* keyframe gap than the Zoom recording (denser,
by the rule's own logic it should seek cheaper) yet costs four times as much
per seek. Bitrate looks like the better predictor on this evidence: the phone
file is encoded at 20 Mbps against 0.86-2.98 Mbps for the other two, meaning
substantially more entropy-coded data to parse per frame regardless of how
close the nearest keyframe is. **§4.1a's binary dense/sparse-by-keyframe-gap
rule would classify the phone file as cheap to seek (it's reasonably dense)
when it is in fact the most expensive file in the entire eight-file set.**
This doesn't necessarily mean §4.1a's per-video runtime measurement approach
is wrong -- it already avoids hardcoding a global assumption -- but the
specific signal it proposes measuring (keyframe interval alone) missed the
dominant cost driver on this evidence. Three real files is not enough to
replace "keyframe gap" with "bitrate" as the decision signal with confidence;
it is enough to say gap alone is not sufficient, and that this wasn't
tested before now.
