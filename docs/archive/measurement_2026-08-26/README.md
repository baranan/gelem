# Video measurement pass -- 26 Aug 2026

**This folder is historical record, not guidance.** Like everything under
`docs/archive/`, nothing here is authoritative. If a number below and a number
in `docs/media_architecture.md` disagree, the current document is right and
this folder is stale by definition -- it is kept so a future reader can see
what was actually run and measured, not to be cited as a source of truth.

## What this run was

The §6.0 measurement pass for the video and decoder questions in
`docs/media_architecture.md`: keyframe-only extraction versus full decode,
scattered-seek cost, sequential-frame cost, and whether PyAV parallelises
across threads. Run against the eight fixtures described in
`docs/fixtures.md`, using PyAV (not OpenCV, per §4.3).

## When, and on what machine

26 Aug 2026. Y B's Windows workstation: Intel Core i9-14900K (24C/32T),
128 GB RAM, Windows 11 Pro 10.0.26200, PyAV 18.1.0. Full environment detail,
including exact library versions, is in `RUNLOG.md`.

## Which document section this supports

`docs/media_architecture.md` §11 ("Numbers, and which of them generalise") is
populated from this run. §10 ("Considered and rejected") cites it directly as
the basis for rejecting P1.3 (the proxy layer) and folding §4.1a in. §4.1b
(segment thumbnails) and §4.3 (batching) were rewritten because of findings
here. The authoritative, current statement of all of this is in those
sections of `docs/media_architecture.md` -- read this folder only for the raw
numbers and method behind them.

## What's in this folder

- **`section11_table.md`** -- the deliverable this run produced: the
  measured tables later pasted into §11, plus the verdict and the two
  contradictions-with-the-plan findings (P1.7a, §4.1a).
- **`RUNLOG.md`** -- machine and library versions, what ran in what order,
  two implementation bugs found and fixed mid-run (and why), and every
  deviation from the method as originally specified.
- **`results_raw.json`** -- every individual timing behind the tables above,
  for re-examining the distributions without rerunning anything.

**Left out deliberately:** the measurement scripts (`measure.py` and its
reruns, `analyze.py`) and the raw console logs. They add nothing once the
results and run log exist, and `docs/fixtures.md`'s generated-and-small /
real-and-large rule is about data, not about re-runnable tooling. The scripts
still live outside the repo, at
`%USERPROFILE%\Documents\gelem.measure\` on the machine above, if anyone
needs to rerun or extend the measurement.

## Where the original lives

`%USERPROFILE%\Documents\gelem.measure\` on the machine named above, outside
this repository. That copy is the working copy and may still change if the
measurement is extended; this one is a fixed snapshot from 26 Aug 2026 and
will not be updated to match.
