"""
tests/test_request_queue.py

P0.5b-2i -- ArtifactStore's request queue (docs/media_architecture.md
section 4.4).

The thread-per-request model becomes a bounded, coalescing, cancellable
worker queue. Nothing a researcher sees changes; these tests are the
only evidence the mechanism is there at all.

Written from the work-item spec, not from the implementation. Each test
is designed so that removing the one mechanism it names makes it fail --
in particular a purely synchronous implementation passes every test that
existed before this file and must fail `test_request_thumbnail_runs_the
_decode_off_the_caller_thread`.

Run with:
    python -m pytest tests/test_request_queue.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from PIL import Image

from artifacts.artifact_store import ArtifactStore
from artifacts.worker_pool import WorkerPool
from media.media_address import resolve_source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_png(path: Path, colour: tuple[int, int, int]) -> None:
    Image.new("RGB", (64, 64), colour).save(path)


def _address_of(path: Path, root: Path) -> tuple[str, str]:
    """(canonical address, absolute source path) -- the same pair the
    controller hands request_thumbnail."""
    return resolve_source(str(path), str(root))


def _spin_until(predicate, timeout: float = 5.0) -> None:
    """Poll `predicate` until it is truthy or the timeout expires. Used
    only to wait for a background worker to reach a known point before
    the test does the next thing."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not met within the timeout")


# ===========================================================================
# a. The pool is bounded -- at most worker_count callables run at once.
#
#    Without the bound (a thread per submitted callable), all five jobs
#    below would run concurrently and `peak` would reach 5.
# ===========================================================================

def test_worker_pool_runs_no_more_than_worker_count_at_once():
    pool = WorkerPool(worker_count=2)

    lock = threading.Lock()
    active = 0
    peak = 0
    started = threading.Semaphore(0)
    release = threading.Event()

    def job():
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        started.release()
        assert release.wait(timeout=10)
        with lock:
            active -= 1

    for _ in range(5):
        pool.submit(job)

    # Two workers pick up two jobs; a third must not start while they run.
    assert started.acquire(timeout=5)
    assert started.acquire(timeout=5)
    assert not started.acquire(timeout=0.5), "a third job ran past the bound"
    with lock:
        assert peak == 2

    release.set()
    pool.shutdown()


def test_clear_pending_drops_queued_but_not_running_callables():
    pool = WorkerPool(worker_count=1)

    started = threading.Semaphore(0)
    release = threading.Event()
    ran: list[str] = []

    def blocker():
        ran.append("blocker")
        started.release()
        assert release.wait(timeout=10)

    def queued():
        ran.append("queued")

    pool.submit(blocker)
    assert started.acquire(timeout=5)   # blocker occupies the one worker
    pool.submit(queued)                  # this one waits in the queue
    pool.clear_pending()                 # ... and is dropped
    release.set()

    time.sleep(0.2)
    assert ran == ["blocker"], "clear_pending() did not drop the queued callable"
    pool.shutdown()


def test_submit_after_shutdown_raises():
    pool = WorkerPool(worker_count=1)
    pool.submit(lambda: None)
    pool.shutdown()
    with pytest.raises(RuntimeError):
        pool.submit(lambda: None)


# ===========================================================================
# a / f. The store honours its own worker_count.
#
#    worker_count=1 means the store decodes one source at a time however
#    many requests are outstanding. If worker_count were ignored (a
#    hardcoded pool, or a thread per request) `peak` would exceed 1.
# ===========================================================================

def test_store_honours_its_worker_count_bound(tmp_path):
    addresses = []
    for index, colour in enumerate([(200, 0, 0), (0, 200, 0), (0, 0, 200)]):
        path = tmp_path / f"c{index}.png"
        _solid_png(path, colour)
        addresses.append(_address_of(path, tmp_path))

    lock = threading.Lock()
    active = 0
    peak = 0
    release = threading.Event()

    class Gated(ArtifactStore):
        def _decode_source(self, source_path):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            assert release.wait(timeout=10)
            with lock:
                active -= 1
            return super()._decode_source(source_path)

    store = Gated(tmp_path / "artifacts", worker_count=1)
    done = threading.Semaphore(0)
    store.on_thumbnail_ready = lambda table_name, row_id: done.release()

    for index, (address, source) in enumerate(addresses):
        store.request_thumbnail(f"r{index}", address, Path(source), "frames")

    # Let every worker that is going to start have started.
    time.sleep(0.3)
    with lock:
        assert peak == 1, f"more than one concurrent decode with worker_count=1: {peak}"

    release.set()
    for _ in addresses:
        assert done.acquire(timeout=10)
    assert peak == 1


# ===========================================================================
# b. request_thumbnail returns without doing the work -- the decode runs
#    on a worker thread, never the caller's.
#
#    This is the test a synchronous implementation fails. Every test that
#    existed before P0.5b-2i passes against a synchronous request_thumbnail.
# ===========================================================================

def test_request_thumbnail_runs_the_decode_off_the_caller_thread(tmp_path):
    source_file = tmp_path / "red.png"
    _solid_png(source_file, (220, 0, 0))
    address, source = _address_of(source_file, tmp_path)

    decoded_on: list[int] = []

    class Recording(ArtifactStore):
        def _decode_source(self, source_path):
            decoded_on.append(threading.get_ident())
            return super()._decode_source(source_path)

    store = Recording(tmp_path / "artifacts")
    done = threading.Event()
    store.on_thumbnail_ready = lambda table_name, row_id: done.set()

    store.request_thumbnail("r", address, Path(source), "frames")

    assert done.wait(timeout=30), "no ready callback -- the job never ran"
    assert decoded_on, "the source was never decoded"
    assert decoded_on[0] != threading.get_ident(), (
        "the decode ran on the calling thread -- request_thumbnail is synchronous"
    )


# ===========================================================================
# c. Coalescing: two requests for the SAME address, from two different
#    (table, row) subscribers, while a worker is held inside the decode.
#    Exactly one decode happens; both subscribers are notified once each.
#
#    Without coalescing the second request starts a second job, so
#    `decode_calls` reaches 2.
# ===========================================================================

def test_requests_for_one_address_coalesce_to_one_decode(tmp_path):
    source_file = tmp_path / "green.png"
    _solid_png(source_file, (0, 180, 0))
    address, source = _address_of(source_file, tmp_path)

    decode_calls: list[str] = []
    release = threading.Event()

    class Gated(ArtifactStore):
        def _decode_source(self, source_path):
            decode_calls.append(source_path.name)
            assert release.wait(timeout=10)
            return super()._decode_source(source_path)

    store = Gated(tmp_path / "artifacts", worker_count=2)

    notified: list[tuple[str, str]] = []
    notify_lock = threading.Lock()
    both = threading.Event()

    def on_ready(table_name, row_id):
        with notify_lock:
            notified.append((table_name, row_id))
            if len(notified) == 2:
                both.set()

    store.on_thumbnail_ready = on_ready

    store.request_thumbnail("row1", address, Path(source), "tableA")
    # Make sure the worker is actually inside the decode before the
    # second request, so it genuinely joins an in-flight job.
    _spin_until(lambda: decode_calls, timeout=5)

    store.request_thumbnail("row2", address, Path(source), "tableB")
    # Give a wrongly-spawned second job time to reach _decode_source too
    # (worker_count is 2, so it would not be queued behind the first).
    time.sleep(0.2)
    assert len(decode_calls) == 1, "the second request started its own decode"

    release.set()
    assert both.wait(timeout=10), "both subscribers were not notified"
    assert len(decode_calls) == 1, "a second decode ran after release"
    assert sorted(notified) == [("tableA", "row1"), ("tableB", "row2")]


# ===========================================================================
# d. Cancellation via reset(): a job enqueued, reset() called, worker
#    released -- no JPEG written, no index entry, no notification. Covered
#    for a job still in the queue and for a job already running.
# ===========================================================================

def test_reset_drops_a_still_queued_job(tmp_path):
    blocker_file = tmp_path / "blocker.png"
    target_file = tmp_path / "target.png"
    _solid_png(blocker_file, (0, 0, 0))
    _solid_png(target_file, (200, 0, 0))
    blocker_addr, blocker_src = _address_of(blocker_file, tmp_path)
    target_addr, target_src = _address_of(target_file, tmp_path)

    entered: list[str] = []
    hold = threading.Event()

    class Gated(ArtifactStore):
        def _decode_source(self, source_path):
            entered.append(source_path.name)
            assert hold.wait(timeout=10)
            return super()._decode_source(source_path)

    # One worker: the blocker occupies it while the target job waits in
    # the queue.
    store = Gated(tmp_path / "artifacts", worker_count=1)
    notified: list[tuple[str, str]] = []
    store.on_thumbnail_ready = lambda t, r: notified.append((t, r))

    store.request_thumbnail("b", blocker_addr, Path(blocker_src), "frames")
    _spin_until(lambda: "blocker.png" in entered, timeout=5)

    store.request_thumbnail("t", target_addr, Path(target_src), "frames")
    store.reset()          # bump the generation while the target is queued
    hold.set()             # let the blocker finish and the queue drain

    time.sleep(0.3)
    assert "target.png" not in entered, "the queued job decoded despite reset()"
    assert list((tmp_path / "artifacts").glob("*.jpg")) == [], "a dropped job wrote a JPEG"
    assert store._index == {}, "a dropped job left an index entry"
    assert store._fingerprints == {}, "a dropped job polluted the fingerprint memo"
    assert notified == [], "a dropped job sent a notification"


def test_reset_drops_an_already_running_job(tmp_path):
    source_file = tmp_path / "red.png"
    _solid_png(source_file, (200, 0, 0))
    address, source = _address_of(source_file, tmp_path)

    in_decode = threading.Event()
    proceed = threading.Event()

    class Gated(ArtifactStore):
        def _decode_source(self, source_path):
            in_decode.set()
            assert proceed.wait(timeout=10)
            return super()._decode_source(source_path)

    store = Gated(tmp_path / "artifacts", worker_count=1)
    notified: list[tuple[str, str]] = []
    store.on_thumbnail_ready = lambda t, r: notified.append((t, r))

    store.request_thumbnail("t", address, Path(source), "frames")
    assert in_decode.wait(timeout=5), "the job never started"

    store.reset()          # cancel the job while it is mid-decode
    proceed.set()          # let the decode return

    time.sleep(0.3)
    assert list((tmp_path / "artifacts").glob("*.jpg")) == [], (
        "a cancelled running job wrote a JPEG"
    )
    # The job committed nothing: not an index entry, and not a
    # fingerprint-memo entry (which would wrongly mark the address
    # 'verified' for whatever project loads next).
    assert store._index == {}, "a cancelled running job left an index entry"
    assert store._fingerprints == {}, "a cancelled running job polluted the memo"
    assert store._verified == set(), "a cancelled running job marked an address verified"
    assert notified == [], "a cancelled running job sent a notification"


# ===========================================================================
# e. The verified short-circuit performs no work and no stat.
#
#    Item 4's decision made into a guardrail: a source file overwritten
#    while the project is open is NOT re-noticed, because a re-stat on the
#    main thread per request is the cost item 4 refuses to pay.
# ===========================================================================

def test_verified_short_circuit_does_no_work_and_no_stat(tmp_path, monkeypatch):
    source_file = tmp_path / "blue.png"
    _solid_png(source_file, (0, 0, 200))
    address, source = _address_of(source_file, tmp_path)

    store = ArtifactStore(tmp_path / "artifacts")
    first_ready = threading.Event()
    store.on_thumbnail_ready = lambda t, r: first_ready.set()

    store.request_thumbnail("r1", address, Path(source), "frames")
    assert first_ready.wait(timeout=30), "first request never completed"

    # The address is now verified this session. Spy on the two things the
    # short-circuit must not do: stat any file, and queue a worker job.
    import pathlib

    real_stat = pathlib.Path.stat
    armed = {"on": False}
    stats_seen: list[str] = []

    def counting_stat(self, *args, **kwargs):
        if armed["on"]:
            stats_seen.append(str(self))
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "stat", counting_stat)

    submit_calls: list[int] = []
    real_submit = store._pool.submit
    monkeypatch.setattr(
        store._pool,
        "submit",
        lambda fn, key=None: (submit_calls.append(1), real_submit(fn, key)),
    )

    second_ready = threading.Event()
    store.on_thumbnail_ready = lambda t, r: second_ready.set()

    armed["on"] = True
    store.request_thumbnail("r2", address, Path(source), "frames")
    armed["on"] = False

    assert second_ready.is_set(), "the short-circuit did not notify synchronously"
    assert stats_seen == [], f"the verified short-circuit performed a stat(): {stats_seen}"
    assert submit_calls == [], "the verified short-circuit queued a worker job"


# ===========================================================================
# f. Worker count is a constructor parameter, keyword-only, and the
#    positional single-argument construction still works.
# ===========================================================================

def test_worker_count_is_a_keyword_only_constructor_parameter(tmp_path):
    # Positional single-argument construction -- as in main.py and every
    # existing test -- still works and gets the default.
    default_store = ArtifactStore(tmp_path / "default")
    assert default_store._pool._worker_count == 2

    # The parameter is honoured.
    custom_store = ArtifactStore(tmp_path / "custom", worker_count=3)
    assert custom_store._pool._worker_count == 3

    # It is keyword-only: a second positional argument is a TypeError,
    # not a silent worker count.
    with pytest.raises(TypeError):
        ArtifactStore(tmp_path / "positional", 3)


# ===========================================================================
# g. WorkerPool job identity -- promote() and drop_pending() (P0.5b-3ii-a).
#
#    The pool never interprets a key; it only matches by membership. Each
#    test parks the single worker inside a blocker job, stacks keyed jobs
#    behind it, reorders or prunes the queue, then releases the worker and
#    reads back the order the jobs actually ran in. Remove promote() and
#    `test_promote_moves_matching_keys_to_the_front` fails; remove
#    drop_pending()'s effect and `test_drop_pending_removes_unkept_jobs...`
#    fails.
# ===========================================================================

def _pool_with_blocked_worker():
    """A 1-worker pool whose worker is parked inside a blocker job.

    Returns (pool, release_event, order, order_lock). Recorder jobs
    appended with `_recorder` write their tag into `order` when they run,
    which only happens after `release_event` is set.
    """
    pool = WorkerPool(worker_count=1)
    started = threading.Semaphore(0)
    release = threading.Event()
    order: list = []
    order_lock = threading.Lock()

    def blocker():
        started.release()
        assert release.wait(timeout=10)

    pool.submit(blocker, key="__blocker__")
    assert started.acquire(timeout=5), "the blocker job never started"
    return pool, release, order, order_lock


def _recorder(order, order_lock, tag):
    def job():
        with order_lock:
            order.append(tag)
    return job


def test_promote_moves_matching_keys_to_the_front():
    pool, release, order, order_lock = _pool_with_blocked_worker()

    pool.submit(_recorder(order, order_lock, "a"), key="a")
    pool.submit(_recorder(order, order_lock, "b"), key="b")
    pool.submit(_recorder(order, order_lock, "c"), key="c")

    pool.promote(["b", "c"])

    release.set()
    _spin_until(lambda: len(order) == 3, timeout=5)
    # b and c jump ahead of a; among themselves, and relative to a, every
    # job keeps its original order.
    assert order == ["b", "c", "a"]
    pool.shutdown()


def test_promote_uses_queue_order_not_the_order_of_the_keys_argument():
    pool, release, order, order_lock = _pool_with_blocked_worker()

    pool.submit(_recorder(order, order_lock, "a"), key="a")
    pool.submit(_recorder(order, order_lock, "b"), key="b")
    pool.submit(_recorder(order, order_lock, "c"), key="c")

    # Keys handed in c-before-b; the queue has b before c, and the queue
    # order is what survives.
    pool.promote(["c", "b"])

    release.set()
    _spin_until(lambda: len(order) == 3, timeout=5)
    assert order == ["b", "c", "a"]
    pool.shutdown()


def test_drop_pending_removes_unkept_jobs_and_returns_their_keys():
    pool, release, order, order_lock = _pool_with_blocked_worker()

    pool.submit(_recorder(order, order_lock, "a"), key="a")
    pool.submit(_recorder(order, order_lock, "b"), key="b")
    pool.submit(_recorder(order, order_lock, "c"), key="c")

    dropped = pool.drop_pending(keep=["b"])
    assert sorted(dropped) == ["a", "c"]

    release.set()
    _spin_until(lambda: order == ["b"], timeout=5)
    time.sleep(0.1)
    assert order == ["b"], "a dropped job still ran"
    pool.shutdown()


def test_drop_pending_returns_nothing_and_changes_nothing_when_all_kept():
    pool, release, order, order_lock = _pool_with_blocked_worker()

    pool.submit(_recorder(order, order_lock, "a"), key="a")
    pool.submit(_recorder(order, order_lock, "b"), key="b")

    assert pool.drop_pending(keep=["a", "b"]) == []

    release.set()
    _spin_until(lambda: len(order) == 2, timeout=5)
    assert order == ["a", "b"]
    pool.shutdown()


def test_promote_and_drop_pending_never_touch_a_running_job():
    pool = WorkerPool(worker_count=1)
    started = threading.Semaphore(0)
    release = threading.Event()
    finished = threading.Event()

    def running_job():
        started.release()
        assert release.wait(timeout=10)
        finished.set()

    pool.submit(running_job, key="running")
    assert started.acquire(timeout=5)

    # "running" is neither kept by drop_pending nor named to promote, but
    # it has already left the queue -- both calls must leave it alone.
    assert pool.drop_pending(keep=[]) == []
    pool.promote(["something-else"])

    release.set()
    assert finished.wait(timeout=5), "a running job was disturbed"
    pool.shutdown()


def test_submit_without_a_key_still_runs_and_drops_as_key_none():
    pool, release, order, order_lock = _pool_with_blocked_worker()

    pool.submit(_recorder(order, order_lock, "keyless"))   # key defaults to None

    # None is not in keep, so the keyless job is dropped and reported as
    # None.
    assert pool.drop_pending(keep=["x"]) == [None]

    release.set()
    time.sleep(0.15)
    assert order == [], "a keyless job survived a drop that did not keep None"
    pool.shutdown()


# ===========================================================================
# h. ArtifactStore.set_wanted_addresses (P0.5b-3ii-a).
#
#    Declares the canonical addresses the display currently wants. Among
#    jobs still QUEUED in the current generation: wanted ones move to the
#    front, every other one is dropped -- no callback, nothing committed,
#    and its _inflight entry removed so a later request starts a fresh
#    job. A job already running is untouched.
# ===========================================================================

def test_set_wanted_addresses_drops_a_queued_request_for_an_unwanted_address(tmp_path):
    blocker_file = tmp_path / "blocker.png"
    wanted_file = tmp_path / "wanted.png"
    unwanted_file = tmp_path / "unwanted.png"
    _solid_png(blocker_file, (0, 0, 0))
    _solid_png(wanted_file, (0, 200, 0))
    _solid_png(unwanted_file, (200, 0, 0))
    blocker_addr, blocker_src = _address_of(blocker_file, tmp_path)
    wanted_addr, wanted_src = _address_of(wanted_file, tmp_path)
    unwanted_addr, unwanted_src = _address_of(unwanted_file, tmp_path)

    entered: list = []
    hold = threading.Event()

    class Gated(ArtifactStore):
        def _decode_source(self, source_path):
            entered.append(source_path.name)
            assert hold.wait(timeout=10)
            return super()._decode_source(source_path)

    # One worker: the blocker holds it while the next two sit in the queue.
    store = Gated(tmp_path / "artifacts", worker_count=1)
    notified: list = []
    store.on_thumbnail_ready = lambda t, r: notified.append((t, r))

    store.request_thumbnail("b", blocker_addr, Path(blocker_src), "frames")
    _spin_until(lambda: "blocker.png" in entered, timeout=5)
    store.request_thumbnail("u", unwanted_addr, Path(unwanted_src), "frames")
    store.request_thumbnail("w", wanted_addr, Path(wanted_src), "frames")

    # Only the 'wanted' address is on screen now.
    store.set_wanted_addresses([wanted_addr])

    hold.set()   # release the blocker; the queue drains

    _spin_until(lambda: ("frames", "w") in notified, timeout=5)
    time.sleep(0.2)
    assert "unwanted.png" not in entered, "a dropped request still decoded"
    assert ("frames", "u") not in notified, "a dropped request still notified"
    assert store.get(unwanted_addr, "thumbnail") is None, "a dropped request committed an index entry"


def test_a_request_for_a_dropped_address_starts_a_new_job(tmp_path):
    blocker_file = tmp_path / "blocker.png"
    target_file = tmp_path / "target.png"
    _solid_png(blocker_file, (0, 0, 0))
    _solid_png(target_file, (10, 20, 30))
    blocker_addr, blocker_src = _address_of(blocker_file, tmp_path)
    target_addr, target_src = _address_of(target_file, tmp_path)

    entered: list = []
    hold = threading.Event()

    class Gated(ArtifactStore):
        def _decode_source(self, source_path):
            entered.append(source_path.name)
            assert hold.wait(timeout=10)
            return super()._decode_source(source_path)

    store = Gated(tmp_path / "artifacts", worker_count=1)
    notified: list = []
    store.on_thumbnail_ready = lambda t, r: notified.append((t, r))

    store.request_thumbnail("b", blocker_addr, Path(blocker_src), "frames")
    _spin_until(lambda: "blocker.png" in entered, timeout=5)

    store.request_thumbnail("t1", target_addr, Path(target_src), "frames")
    store.set_wanted_addresses([])          # drop the queued target job
    time.sleep(0.1)

    # A later request for the same address must NOT silently join the job
    # that was just dropped -- it must start its own.
    store.request_thumbnail("t2", target_addr, Path(target_src), "frames")

    hold.set()
    _spin_until(lambda: ("frames", "t2") in notified, timeout=5)
    assert "target.png" in entered, "the new request never decoded"
    assert ("frames", "t1") not in notified, "the dropped request still notified"


def test_set_wanted_addresses_keeps_wanted_jobs_in_queue_order(tmp_path):
    files = {}
    for tag, colour in [("blk", (0, 0, 0)), ("a", (200, 0, 0)),
                        ("b", (0, 200, 0)), ("c", (0, 0, 200))]:
        path = tmp_path / f"{tag}.png"
        _solid_png(path, colour)
        files[tag] = _address_of(path, tmp_path)

    entered: list = []
    hold = threading.Event()

    class Gated(ArtifactStore):
        def _decode_source(self, source_path):
            entered.append(source_path.name)
            if source_path.name == "blk.png":
                assert hold.wait(timeout=10)
            return super()._decode_source(source_path)

    store = Gated(tmp_path / "artifacts", worker_count=1)
    store.on_thumbnail_ready = lambda t, r: None

    blk_addr, blk_src = files["blk"]
    store.request_thumbnail("blk", blk_addr, Path(blk_src), "frames")
    _spin_until(lambda: entered == ["blk.png"], timeout=5)

    for tag in ("a", "b", "c"):
        addr, src = files[tag]
        store.request_thumbnail(tag, addr, Path(src), "frames")

    # b and c are wanted, a is not. Keys handed c-first, but the two
    # survivors must decode in queue order (b then c) and a must not
    # decode at all.
    store.set_wanted_addresses([files["c"][0], files["b"][0]])

    hold.set()
    _spin_until(lambda: len(entered) == 3, timeout=5)
    time.sleep(0.2)
    assert entered == ["blk.png", "b.png", "c.png"]


def test_set_wanted_addresses_does_not_disturb_a_running_job(tmp_path):
    source_file = tmp_path / "run.png"
    _solid_png(source_file, (123, 45, 67))
    address, source = _address_of(source_file, tmp_path)

    in_decode = threading.Event()
    proceed = threading.Event()

    class Gated(ArtifactStore):
        def _decode_source(self, source_path):
            in_decode.set()
            assert proceed.wait(timeout=10)
            return super()._decode_source(source_path)

    store = Gated(tmp_path / "artifacts", worker_count=1)
    done = threading.Event()
    store.on_thumbnail_ready = lambda t, r: done.set()

    store.request_thumbnail("r", address, Path(source), "frames")
    assert in_decode.wait(timeout=5), "the job never started"

    # The running job's address is not in the wanted set, but the job has
    # already started -- it must finish, commit and notify.
    store.set_wanted_addresses(["some/other/address.png"])
    proceed.set()

    assert done.wait(timeout=10), "a running job was cancelled by set_wanted_addresses"
    assert store.get(address, "thumbnail") is not None, "the running job did not commit"
