"""
artifacts/worker_pool.py

WorkerPool -- a small bounded FIFO executor of plain callables.

P0.5b-2i split this out of ArtifactStore so the scheduling primitive can
be tested on its own. It deliberately knows nothing about media
addresses, ArtifactKeys, fingerprints or Qt: it takes zero-argument
callables and runs them, at most `worker_count` at a time, in the order
they were submitted.

P0.5b-3ii-a gave each job an optional `key` -- an opaque handle the pool
never inspects. `promote(keys)` pulls the jobs carrying those keys to the
front of the pending queue; `drop_pending(keep)` deletes the pending jobs
whose keys are not in `keep` and reports what it deleted. Both act only
on jobs still waiting; a job already running on a worker is never
touched. The pool still does not know what a key means -- deciding which
addresses are wanted, and turning that into a key set, is ArtifactStore's
job one layer up.

Everything address-shaped -- request coalescing by canonical address,
generation-based cancellation tied to ArtifactStore.reset(), and the
choice of which jobs are worth promoting or dropping -- lives one layer
up, in ArtifactStore. This module is just the bound and the ordering
primitive.

Threads are daemon threads and are started lazily on the first submit, so
importing this module (or building a pool that is never used) starts
nothing, and a worker blocked on a slow read never delays interpreter
exit. concurrent.futures.ThreadPoolExecutor is deliberately not used:
its worker threads are non-daemon and its atexit hook joins them, which
would hang a Qt app quitting while a thumbnail read is stuck on a network
path.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Iterable


# Posted to the queue by shutdown() to wake a worker and tell it to exit.
_SENTINEL = object()


class WorkerPool:
    """Runs submitted callables on a fixed number of daemon threads,
    FIFO, at most `worker_count` at once."""

    def __init__(self, worker_count: int = 2):
        # Low default on purpose: worker count is machine dependent, so
        # it is a parameter, never a hardcoded constant (CLAUDE.md's
        # generality rule). The caller -- ArtifactStore -- surfaces it as
        # its own constructor parameter.
        if worker_count < 1:
            raise ValueError("worker_count must be at least 1")
        self._worker_count = worker_count

        # _queue holds pending jobs. Each entry is a (fn, key) tuple --
        # `key` is the opaque handle promote()/drop_pending() match on,
        # or None when submit() was called without one. shutdown() also
        # appends the bare _SENTINEL object, which the loops below skip.
        # _lock guards the queue; _not_empty is how idle workers wait for
        # the next job without spinning.
        self._queue: deque = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

        # Threads are created on the first submit(), not here.
        self._started = False
        self._shutdown = False
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, fn: Callable[[], None], key: object = None) -> None:
        """Queue `fn` to run on a worker thread. Returns immediately --
        the pool never runs the callable on the caller's thread.

        `fn` takes no arguments and its return value is ignored. An
        exception it raises is caught and printed, never propagated.

        `key` is an opaque handle for this job. The pool never inspects
        or compares it except by membership in the collections handed to
        promote() and drop_pending(). Two jobs may share a key, and None
        (the default) is a valid key like any other -- a job submitted
        without a key is dropped by any drop_pending() call whose `keep`
        does not contain None.
        """
        with self._not_empty:
            if self._shutdown:
                raise RuntimeError("WorkerPool.submit() after shutdown()")
            if not self._started:
                self._start_workers()
            self._queue.append((fn, key))
            self._not_empty.notify()

    def promote(self, keys: Iterable[object]) -> None:
        """Move every pending job whose key is in `keys` to the front of
        the queue. The promoted jobs keep their existing order relative
        to each other, and the jobs left behind keep theirs -- only the
        two groups swap places. A job already running on a worker is not
        in the queue and is unaffected. Keys that match no pending job
        are ignored.
        """
        # A plain list, not a set: a key need not be hashable, and the
        # pending queue is short enough that membership cost does not
        # matter.
        wanted = list(keys)
        with self._lock:
            promoted: list = []
            rest: list = []
            for entry in self._queue:
                if entry is _SENTINEL:
                    # Only present during shutdown(), which does not call
                    # this method -- but keep it in place if it is.
                    rest.append(entry)
                    continue
                _fn, key = entry
                if key in wanted:
                    promoted.append(entry)
                else:
                    rest.append(entry)
            self._queue.clear()
            self._queue.extend(promoted)
            self._queue.extend(rest)

    def drop_pending(self, keep: Iterable[object]) -> list:
        """Remove every pending job whose key is NOT in `keep`, and
        return the removed keys in the order the jobs sat in the queue.
        A job already running on a worker is not in the queue and is
        never dropped. This is clear_pending() with a survivor set, plus
        a report of what went.
        """
        survivors_keys = list(keep)
        with self._lock:
            survivors: list = []
            dropped: list = []
            for entry in self._queue:
                if entry is _SENTINEL:
                    survivors.append(entry)
                    continue
                _fn, key = entry
                if key in survivors_keys:
                    survivors.append(entry)
                else:
                    dropped.append(key)
            self._queue.clear()
            self._queue.extend(survivors)
        return dropped

    def clear_pending(self) -> None:
        """Drop every callable that has been queued but not yet started.
        Callables already running on a worker are unaffected.

        ArtifactStore.reset() calls this: a generation bump makes every
        outstanding job a no-op, but without this the workers still have
        to pull each stale closure off the queue before reaching the new
        project's work.
        """
        with self._lock:
            self._queue.clear()

    def shutdown(self) -> None:
        """Stop all workers once they finish their current job. Pending
        (not yet started) callables are dropped. Mainly for tests -- a
        running application relies on the threads being daemon threads
        and exiting with the process.
        """
        with self._not_empty:
            if not self._started:
                self._shutdown = True
                return
            self._shutdown = True
            # Drop anything not yet picked up, then wake every worker
            # with one sentinel each.
            self._queue.clear()
            for _ in self._threads:
                self._queue.append(_SENTINEL)
            self._not_empty.notify_all()

        for thread in self._threads:
            thread.join(timeout=2)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _start_workers(self) -> None:
        """Create the daemon worker threads. Caller holds _lock."""
        self._started = True
        for index in range(self._worker_count):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"artifact-worker-{index}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

    def _worker_loop(self) -> None:
        """Pull one job at a time and run it, forever (or until a
        sentinel from shutdown())."""
        while True:
            with self._not_empty:
                while not self._queue:
                    self._not_empty.wait()
                entry = self._queue.popleft()

            if entry is _SENTINEL:
                return

            fn, _key = entry
            try:
                fn()
            except Exception as error:
                # A job must never take a worker down. ArtifactStore's
                # own job body already catches its decode errors; this is
                # the backstop for anything it misses.
                print(f"[WorkerPool] job raised: {error}")
