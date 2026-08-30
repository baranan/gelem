"""
artifacts/worker_pool.py

WorkerPool -- a small bounded FIFO executor of plain callables.

P0.5b-2i split this out of ArtifactStore so the scheduling primitive can
be tested on its own. It deliberately knows nothing about media
addresses, ArtifactKeys, fingerprints or Qt: it takes zero-argument
callables and runs them, at most `worker_count` at a time, in the order
they were submitted.

Everything address-shaped -- request coalescing by canonical address,
and generation-based cancellation tied to ArtifactStore.reset() -- lives
one layer up, in ArtifactStore. This module is just the bound.

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
from typing import Callable


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

        # _queue holds pending callables. _lock guards it; _not_empty is
        # how idle workers wait for the next job without spinning.
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

    def submit(self, fn: Callable[[], None]) -> None:
        """Queue `fn` to run on a worker thread. Returns immediately --
        the pool never runs the callable on the caller's thread.

        `fn` takes no arguments and its return value is ignored. An
        exception it raises is caught and printed, never propagated.
        """
        with self._not_empty:
            if self._shutdown:
                raise RuntimeError("WorkerPool.submit() after shutdown()")
            if not self._started:
                self._start_workers()
            self._queue.append(fn)
            self._not_empty.notify()

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
                job = self._queue.popleft()

            if job is _SENTINEL:
                return

            try:
                job()
            except Exception as error:
                # A job must never take a worker down. ArtifactStore's
                # own job body already catches its decode errors; this is
                # the backstop for anything it misses.
                print(f"[WorkerPool] job raised: {error}")
