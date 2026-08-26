# Gelem video fixture set

**Status:** built and verified, 26 Aug 2026
**Built for:** the §6.0 measurement pass of `media_architecture.md`
**Reused by:** P0.3 (address semantics) and the §7 guardrail tests

---

## Where the media lives, and why not here

The three real recordings contain identifiable participants, and the set runs to
about 2 GB. It is therefore kept on local disk **outside the repository**, and is
not synchronised to Google Drive -- a Drive Streaming path would turn a first read
into a network fetch, which makes any timing measurement meaningless, and its
shifting modification times would spuriously invalidate the artifact cache key
described in §4.5.

Tests locate the folder through a `GELEM_FIXTURES` environment variable and
**skip rather than fail** when it is unset, so a checkout without the media stays
green.

The dividing line for anything added later: **generated-and-small may live in the
repo; real-and-large does not.** The synthetic lossless fixture that §7 requires is
reproducible from a command and contains no participants, so it should be generated
on demand by the test fixture itself rather than committed as a file.

Current location on the author's machine: `C:\Users\baranan\Documents\gelem.videos`.
That path is machine-specific and should never be hardcoded anywhere.

---

## Real recordings

Unmodified. Original filenames are kept because they carry provenance
(participant id, date). Nothing in this group was trimmed or re-encoded: a
re-encode would replace the keyframe structure that is the whole reason these
files are here.

| File | Source | Codec | Resolution | Frame rate | Duration | Bitrate | Keyframe gap |
|---|---|---|---|---|---|---|---|
| `sid89_video.mp4` | Zoom study recording | h264 (Baseline) | 1920x1080 | 25, constant | 44:30.6 | 0.86 Mbps | mean 2.000 s, min 0.760, max 2.000, n=1338 |
| `PID_031_2026-01-08_16-17-54.mp4` | Legacy study encoder | **mpeg4** (MPEG-4 Part 2) | 3200x1200 | 20, constant | 35:49 | 2.98 Mbps | 0.600 s, perfectly uniform, n=3582 |
| `VID_20260826_100749315.mp4` | Moto G phone, stock camera app | h264 | 1920x1080 stored | ~29.9, **variable** (98 distinct frame durations) | 2:59.3 | 20 Mbps | mean 1.002 s, min 0.995, max 1.011, n=179 |

**Notes.**

- `sid89_video.mp4` is also the source for the generated set below. ffmpeg only
  reads its input, so this file is untouched by that use.
- The legacy file is **mpeg4**, a third codec family alongside h264 and hevc. The
  resolver must not assume the corpus is H.264/H.265. Its 3200x1200 frame is almost
  certainly two cameras side by side -- exactly the "splitting a side-by-side
  recording into two rows" scenario §3.6 decision 5 names as a likely future
  source of a region address, though as of 26 Aug 2026 nothing in Gelem produces
  one and none is scheduled.
- The phone file carries `rotation=90`: recorded portrait, stored landscape. Its
  video stream has **no** edit list. Its **audio** stream has an edit list whose
  first entry is empty (`media time: -1`), so audio and video begin at different
  times -- a real instance of §3.6 item 10. `start_time=0`.
- Transferred by USB in file-transfer mode, verified untranscoded by bitrate
  (20 Mbps) and by the survival of the rotation metadata.

**All three are dense, 0.6 s to 2.0 s. No real recording in this corpus has a long
GOP.**

---

## Generated fixtures

All five derive from `_master_cfr25.mp4`, which is 20 minutes of `sid89_video.mp4`
starting at 00:05:00, normalised to 1080p at 25 fps constant. Audio stripped
(`-an`). All are closed-GOP with scene detection disabled, so a stated GOP length
is the real one.

The four constant-frame-rate files form a 2x2 of codec against GOP length with
everything else held equal, including duration. That is what allows a timing
difference between two of them to be attributed to the one property that differs.
**If any of them is ever regenerated, all four must be.**

| File | Codec | Frame rate | Frames | Keyframes | Gap min / max / mean | Distinct frame durations | Size |
|---|---|---|---|---|---|---|---|
| `_master_cfr25.mp4` | h264 | 25, constant | 30000 | 1200 | 1.000 s | 1 | 337 MB |
| `h264_cfr_gop1s.mp4` | h264 | 25, constant | 30000 | 1200 | 1.000 / 1.000 / 1.000 | 1 | 214 MB |
| `h264_cfr_gop10s.mp4` | h264 | 25, constant | 30000 | 120 | 10.000 / 10.000 / 10.000 | 1 | 168 MB |
| `h265_cfr_gop1s.mp4` | hevc | 25, constant | 30000 | 1200 | 1.000 / 1.000 / 1.000 | 1 | 97 MB |
| `h265_cfr_gop10s.mp4` | hevc | 25, constant | 30000 | 120 | 10.000 / 10.000 / 10.000 | 1 | 76 MB |
| `h264_vfr_longgop.mp4` | h264 | **variable** | 20280 | 82 | 10.320 / **49.360** / 14.788 | 17 | 164 MB |

**About the variable-frame-rate file.** `mpdecimate` dropped 32.4% of frames, so
the average rate is 16.9 fps against a 25 fps nominal. `keyint=250` counts
**frames, not seconds**, so once frames were removed those 250 surviving frames
came to span between 10 and 49 seconds of wall time. The file is named
`longgop` rather than `gop10s` for that reason: a name asserting 10 s on a file
with 49 s gaps is the exact failure this set exists to prevent. The 49 s extreme is
useful, not a defect -- it is the harshest seek and sparsest-proxy case available
here.

**A synthetic VFR file is not a phone's VFR.** A phone varies its frame interval
because exposure time tracks the light. `mpdecimate` only produces irregular
timestamps. The phone recording is the authoritative variable-frame-rate fixture;
this one is a controlled companion.

---

## Lossless known-frame fixture (P0.3)

Required by `media_architecture.md` §7 for an equivalence reference that a lossy
codec cannot serve: every frame's pixels report their own frame index, exactly,
with no colour conversion in the way. It is small (a few KB), synthetic, and
contains no participants, so per the dividing line above it is **generated on
demand by the test fixture itself** (`tests/test_media_address.py`,
`_generate_known_frame_video`) rather than committed.

64x64, grayscale (`-pix_fmt gray`, so there is no RGB/YUV conversion to
introduce doubt), 25 fps, 2 s -- 50 frames, each comfortably under 256 so no
wraparound is needed. FFV1 is lossless. `geq=lum='N'` sets every pixel of frame
`N` to the value `N`.

```powershell
ffmpeg -hide_banner -y -f lavfi -i "color=c=black:s=64x64:r=25:d=2" -vf "format=gray,geq=lum='N'" -pix_fmt gray -c:v ffv1 known_frames.mkv
```

The self-check decodes it back with a plain `ffmpeg ... -f rawvideo -pix_fmt gray -`
pipe (no PyAV, no OpenCV -- nothing beyond ffmpeg itself, which fixture
generation already requires) and asserts every byte of frame `N` equals `N`.
Frame-selection tests that resolve addresses *against* this fixture belong to
P1.2, which has a resolver to decode with; P0.3's own test of it is only this
self-check.

## Video-stream edit list attempt (P0.3, decision 12)

No file in the set above has an edit list on its *video* stream (see Known
gaps, historically). `media_architecture.md` §3.6 item 12 needed one to test
against. Attempted by stream-copying a non-keyframe-aligned cut of the real
Zoom recording -- `-ss` lands 0.3 s past the nearest keyframe (keyframes are at
even seconds in that file), so a stream-copy trim cannot start exactly there
and must either drop the pre-roll or hide it with an edit list:

```powershell
ffmpeg -hide_banner -y -ss 10.3 -i "$env:GELEM_FIXTURES\sid89_video.mp4" -t 3 -c copy -map 0:v:0 elst_attempt.mp4
```

**Outcome: it worked, verified 26 Aug 2026.** `ffprobe -v debug` on the result
reports a genuine edit list on stream 0 (the mapped, and therefore only,
stream -- video): `Processing st: 0, edit list 0 - media time: 9000, duration:
90600` at the file's 1/30000 time base, i.e. a 0.3 s pre-roll -- matching the
0.3 s offset requested, and reproduced on a second independent run of the same
command. This is not assumed from the command succeeding; it is read back from
the container the way `media_architecture.md` §3.6 warns to. The command and
check are automated in `tests/test_media_address.py`
(`test_decision12_edit_list_on_video_stream_attempt`), which skips rather than
fails if `GELEM_FIXTURES` is unset or a future ffmpeg version stops
reproducing this. The output file is generated into the test's own `tmp_path`
and is not committed -- it is a three-second cut of a real, identifiable
participant recording, not a synthetic fixture, so it belongs outside the repo
like the recording it comes from.

---

## Known gaps

Recorded so that nobody mistakes absence of a test for a passing one.

| Gap | Consequence |
|---|---|
| No real H.265 file | H.265 is represented only by the two synthetic fixtures. One 30-second iPhone clip would close it, and would likely bring a video-stream edit list with it. |
| Source is soft, 0.86 Mbps for 1080p | Absolute decode times from the generated set will be optimistic. Acceptable: §11 already holds that absolute per-frame times do not generalise and must not drive design. Ratios are unaffected, since all five share the source. |

**Closed 26 Aug 2026 (P0.3):**

- No lossless known-frame fixture -- built above, generated on demand from
  `ffmpeg` alone. Robustly closed: `test_lossless_known_frame_fixture_self_check`
  runs (and can fail) in any environment with `ffmpeg` on PATH, including CI.

**Closed on the author's machine only, 26 Aug 2026 (P0.3) -- not the same
strength of closure as the row above:**

- No **video-stream** edit list anywhere in the committed set -- §3.6 item 12
  is verified against the generated attempt above, but
  `test_decision12_edit_list_on_video_stream_attempt` is double-gated: it
  needs both `ffmpeg`/`ffprobe` on PATH and `GELEM_FIXTURES` pointing at the
  real, uncommitted recordings, which per this document's own opening section
  exist only on Y B's machine and are never synced to the repo or Drive. On
  any other checkout -- CI included -- this test unconditionally **skips**,
  which looks identical to "not yet run" rather than "regressed". A future
  ffmpeg version could stop reproducing the edit list and nothing would flag
  it; re-run the test on `GELEM_FIXTURES` before trusting this line, rather
  than trusting the line itself.

---

## Generation commands

Run from the fixture folder. `scenecut=0` is the flag that matters most: without it
the encoder inserts extra keyframes at scene changes and a "10 s GOP" file quietly
gets keyframes every second or two.

```powershell
# master: 20 min from the Zoom recording, normalised to 1080p25 CFR
ffmpeg -hide_banner -ss 00:05:00 -i sid89_video.mp4 -t 00:20:00 -an -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -r 25 -fps_mode cfr -x264-params "keyint=25:min-keyint=25:scenecut=0:open-gop=0" _master_cfr25.mp4

# H.264, 1 s GOP  (25 frames at 25 fps)
ffmpeg -hide_banner -i _master_cfr25.mp4 -an -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -r 25 -fps_mode cfr -x264-params "keyint=25:min-keyint=25:scenecut=0:open-gop=0" h264_cfr_gop1s.mp4

# H.264, 10 s GOP  (250 frames at 25 fps)
ffmpeg -hide_banner -i _master_cfr25.mp4 -an -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -r 25 -fps_mode cfr -x264-params "keyint=250:min-keyint=250:scenecut=0:open-gop=0" h264_cfr_gop10s.mp4

# H.265, 1 s GOP
ffmpeg -hide_banner -i _master_cfr25.mp4 -an -c:v libx265 -preset fast -crf 24 -pix_fmt yuv420p -r 25 -fps_mode cfr -x265-params "keyint=25:min-keyint=25:scenecut=0:open-gop=0" -tag:v hvc1 h265_cfr_gop1s.mp4

# H.265, 10 s GOP
ffmpeg -hide_banner -i _master_cfr25.mp4 -an -c:v libx265 -preset fast -crf 24 -pix_fmt yuv420p -r 25 -fps_mode cfr -x265-params "keyint=250:min-keyint=250:scenecut=0:open-gop=0" -tag:v hvc1 h265_cfr_gop10s.mp4

# H.264, variable frame rate. There is deliberately no -r here:
# specifying an output frame rate forces CFR and would defeat the file.
ffmpeg -hide_banner -i _master_cfr25.mp4 -an -vf "mpdecimate=hi=64*12:lo=64*5:frac=0.33" -fps_mode vfr -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -x264-params "keyint=250:min-keyint=250:scenecut=0:open-gop=0" h264_vfr_longgop.mp4
```

Requires ffmpeg with libx264 and libx265. On Windows: `winget install --id
Gyan.FFmpeg -e`, then open a new shell so PATH is picked up.

Keep `_master_cfr25.mp4` as long as the set is in use, so a single variant can be
regenerated without starting over.

---

## Verification

The encoder log says what ffmpeg intended. This reads what is actually in the
container, which is the only thing that counts. One paste, one line per file.

```powershell
$files = "h264_cfr_gop1s","h264_cfr_gop10s","h265_cfr_gop1s","h265_cfr_gop10s","h264_vfr_longgop"
foreach ($n in $files) {
  $f = "$env:GELEM_FIXTURES\$n.mp4"
  $k = ffprobe -v error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=p=0 $f | Where-Object { $_ -match ',K' } | ForEach-Object { [double]($_ -split ',')[0] }
  $g = 1..($k.Count-1) | ForEach-Object { $k[$_] - $k[$_-1] }
  $s = $g | Measure-Object -Minimum -Maximum -Average
  $u = (ffprobe -v error -select_streams v:0 -show_entries packet=duration_time -of csv=p=0 $f | Sort-Object -Unique).Count
  $c = ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 $f
  "{0,-20} codec={1,-5} kf={2,-5} min={3:N3} max={4:N3} mean={5:N3} uniqDur={6}" -f $n,$c,$k.Count,$s.Minimum,$s.Maximum,$s.Average,$u
}
```

**Pass condition.** `uniqDur=1` on the four constant-frame-rate files; minimum gap
equal to the mean on all four; many distinct durations on the variable file. A
minimum gap below the mean on a CFR file means scene detection leaked in and that
file must be regenerated.

Recorded output, 26 Aug 2026:

```
h264_cfr_gop1s       codec=h264  kf=1200  min=1.000  max=1.000  mean=1.000  uniqDur=1
h264_cfr_gop10s      codec=h264  kf=120   min=10.000 max=10.000 mean=10.000 uniqDur=1
h265_cfr_gop1s       codec=hevc  kf=1200  min=1.000  max=1.000  mean=1.000  uniqDur=1
h265_cfr_gop10s      codec=hevc  kf=120   min=10.000 max=10.000 mean=10.000 uniqDur=1
h264_vfr_longgop     codec=h264  kf=82    min=10.320 max=49.360 mean=14.788 uniqDur=17
```

### Inspecting a real recording

Keyframe spacing of any file, which is also how §6.0's question about real data was
answered:

```powershell
$f = "<path to file>"
$k = ffprobe -v error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=p=0 $f | Where-Object { $_ -match ',K' } | ForEach-Object { [double]($_ -split ',')[0] }
$gaps = 1..($k.Count-1) | ForEach-Object { $k[$_] - $k[$_-1] }
"keyframes=$($k.Count)"; $gaps | Measure-Object -Minimum -Maximum -Average
```

Rotation, start time and container structure:

```powershell
ffprobe -hide_banner -v error -select_streams v:0 -show_streams $f | Select-String -Pattern "rotation|rotate|^width=|^height=|start_time|time_base"
ffprobe -hide_banner -v debug $f 2>&1 | Select-String -Pattern "edit list|elst"
```

Note that video packets are stored in decode order, so with B-frames present
`pts_time` is not monotonic and naive consecutive differences can go negative. The
`duration_time` test above avoids the problem entirely, which is why it is
preferred for deciding constant versus variable frame rate.
