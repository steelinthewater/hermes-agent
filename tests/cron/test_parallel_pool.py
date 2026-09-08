"""Tests for the persistent parallel pool and running-job guard in cron/scheduler.py.

These verify the fix for the tick-blocking issue where as_completed(timeout=600)
prevented the ticker thread from firing, causing all other jobs to be fast-forwarded.
"""

import concurrent.futures
import threading
import time
from unittest.mock import patch

import pytest


class TestPersistentPool:
    """_get_parallel_pool returns a persistent ThreadPoolExecutor."""

    def test_pool_is_reused(self, monkeypatch):
        """Same pool instance returned when max_workers doesn't change."""
        import cron.scheduler as sched

        # Reset module state.
        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None

        pool1 = sched._get_parallel_pool(4)
        pool2 = sched._get_parallel_pool(4)
        assert pool1 is pool2

        # Cleanup.
        sched._shutdown_parallel_pool()


    def test_shutdown_clears_pool(self, monkeypatch):
        """_shutdown_parallel_pool resets state."""
        import cron.scheduler as sched

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._get_parallel_pool(2)

        sched._shutdown_parallel_pool()
        assert sched._parallel_pool is None
        assert sched._parallel_pool_max_workers is None


class TestRunningJobGuard:
    """_running_job_ids prevents double-dispatch of active jobs."""

    def test_running_set_prevents_double_dispatch(self, tmp_path, monkeypatch):
        """A job already in _running_job_ids is skipped on the next tick."""
        import cron.scheduler as sched

        # Reset state.
        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._running_job_ids.clear()

        job = {
            "id": "guard-job",
            "name": "guard-test",
            "prompt": "test",
            "schedule": "every 5m",
            "enabled": True,
            "next_run_at": "2020-01-01T00:00:00",
            "deliver": "local",
        }

        # Simulate the job already running.
        sched._running_job_ids.add("guard-job")

        dispatched = []
        monkeypatch.setattr(sched, "get_due_jobs", lambda: [job])
        monkeypatch.setattr(sched, "advance_next_runs", lambda *_a, **_kw: 0)
        monkeypatch.setattr(sched, "run_job", lambda j, **_kw: dispatched.append(j["id"]) or (True, "out", "resp", None))
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)

        n = sched.tick(verbose=False)
        assert n == 0  # skipped, not dispatched
        assert dispatched == []

        sched._running_job_ids.discard("guard-job")
        sched._shutdown_parallel_pool()


    def test_create_execution_failure_stays_due_and_later_job_runs(self, monkeypatch):
        """A failed execution claim must not advance or block later jobs."""
        import cron.scheduler as sched

        sched._running_job_ids.clear()
        jobs = [
            {"id": job_id, "name": job_id, "prompt": "test",
             "schedule": "every 5m", "enabled": True,
             "next_run_at": "2020-01-01T00:00:00", "deliver": "local"}
            for job_id in ("claim-fails", "still-runs")
        ]
        advance_calls = []
        called = []

        class InlinePool:
            def submit(self, callback):
                future = concurrent.futures.Future()
                future.set_result(callback())
                return future

        def create_execution(job_id, **_kwargs):
            if job_id == "claim-fails":
                raise RuntimeError("execution ledger unavailable")
            return {"id": f"{job_id}-execution"}

        monkeypatch.setattr(sched, "get_due_jobs", lambda: jobs)
        monkeypatch.setattr(sched, "_get_parallel_pool", lambda _workers: InlinePool())
        monkeypatch.setattr(sched, "create_execution", create_execution)
        monkeypatch.setattr(
            sched, "advance_next_runs",
            lambda ids: advance_calls.append(list(ids)) or len(list(ids)),
        )
        monkeypatch.setattr(
            sched, "run_one_job",
            lambda job, **_kwargs: called.append(job["id"]) or True,
        )

        assert sched.tick(verbose=False) == 1
        assert called == ["still-runs"]
        assert advance_calls == [["still-runs"]]
        assert "claim-fails" not in sched.get_running_job_ids()
        assert "still-runs" not in sched.get_running_job_ids()


    def test_advance_failure_releases_reserved_execution(self, monkeypatch):
        """No worker or running guard survives a failed schedule write."""
        import cron.scheduler as sched

        sched._running_job_ids.clear()
        job = {
            "id": "advance-fails", "name": "advance-fails", "prompt": "test",
            "schedule": "every 5m", "enabled": True,
            "next_run_at": "2020-01-01T00:00:00", "deliver": "local",
        }
        submitted = []
        finished = []

        class RecordingPool:
            def submit(self, callback):
                submitted.append(callback)

        monkeypatch.setattr(sched, "get_due_jobs", lambda: [job])
        monkeypatch.setattr(sched, "_get_parallel_pool", lambda _workers: RecordingPool())
        monkeypatch.setattr(
            sched, "create_execution", lambda *_args, **_kwargs: {"id": "execution-1"},
        )
        monkeypatch.setattr(
            sched, "advance_next_runs",
            lambda _ids: (_ for _ in ()).throw(RuntimeError("jobs store unavailable")),
        )
        monkeypatch.setattr(
            sched, "finish_execution",
            lambda execution_id, **kwargs: finished.append((execution_id, kwargs)),
        )

        with pytest.raises(RuntimeError, match="jobs store unavailable"):
            sched.tick(verbose=False)

        assert submitted == []
        assert finished == [
            ("execution-1", {
                "success": False,
                "error": "Schedule advance failed: jobs store unavailable",
            })
        ]
        assert "advance-fails" not in sched.get_running_job_ids()


    def test_submit_and_finish_failures_do_not_wedge_later_job(self, monkeypatch):
        """A failed cleanup write must not block later reserved jobs."""
        import cron.scheduler as sched

        sched._running_job_ids.clear()
        jobs = [
            {"id": job_id, "name": job_id, "prompt": "test",
             "schedule": "every 5m", "enabled": True,
             "next_run_at": "2020-01-01T00:00:00", "deliver": "local"}
            for job_id in ("submit-fails", "still-runs")
        ]
        called = []

        class FirstSubmitFailsPool:
            def __init__(self):
                self.calls = 0

            def submit(self, callback):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("executor rejected")
                future = concurrent.futures.Future()
                future.set_result(callback())
                return future

        def finish_execution(execution_id, **_kwargs):
            if execution_id == "submit-fails-execution":
                raise RuntimeError("execution ledger unavailable")

        pool = FirstSubmitFailsPool()
        monkeypatch.setattr(sched, "get_due_jobs", lambda: jobs)
        monkeypatch.setattr(sched, "_get_parallel_pool", lambda _workers: pool)
        monkeypatch.setattr(
            sched, "create_execution",
            lambda job_id, **_kwargs: {"id": f"{job_id}-execution"},
        )
        monkeypatch.setattr(sched, "advance_next_runs", lambda ids: len(list(ids)))
        monkeypatch.setattr(sched, "finish_execution", finish_execution)
        monkeypatch.setattr(
            sched, "run_one_job",
            lambda job, **_kwargs: called.append(job["id"]) or True,
        )

        assert sched.tick(verbose=False) == 1
        assert called == ["still-runs"]
        assert "submit-fails" not in sched.get_running_job_ids()
        assert "still-runs" not in sched.get_running_job_ids()


class TestSyncMode:
    """tick() blocks by default (sync=True); tick(sync=False) returns immediately."""

    def test_sync_true_blocks_and_returns_correct_count(self, tmp_path, monkeypatch):
        """sync=True waits for jobs and returns actual results."""
        import cron.scheduler as sched

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._running_job_ids.clear()

        jobs = [
            {"id": f"job-{i}", "name": f"Job {i}", "prompt": "test",
             "schedule": "every 5m", "enabled": True,
             "next_run_at": "2020-01-01T00:00:00", "deliver": "local"}
            for i in range(3)
        ]

        monkeypatch.setattr(sched, "get_due_jobs", lambda: jobs)
        monkeypatch.setattr(sched, "advance_next_runs", lambda *_a, **_kw: 0)
        monkeypatch.setattr(sched, "run_job", lambda j, **_kw: (True, "out", "resp", None))
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_kw: "/tmp/out")
        monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)

        n = sched.tick(verbose=False)
        assert n == 3

        sched._shutdown_parallel_pool()


class TestSequentialPool:
    """Sequential (workdir) jobs use the persistent cron-seq pool.

    Verifies the follow-up fix: env-mutating jobs no longer run inline
    in the ticker thread, so a long workdir job can't starve the
    schedule the same way the parallel path used to.
    """

    def test_sequential_job_does_not_block_ticker(self, tmp_path, monkeypatch):
        """sync=False returns immediately even when a workdir job is slow."""
        import cron.scheduler as sched

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._sequential_pool = None
        sched._running_job_ids.clear()

        job = {
            "id": "slow-workdir",
            "name": "slow-workdir",
            "prompt": "test",
            "schedule": "every 5m",
            "enabled": True,
            "next_run_at": "2020-01-01T00:00:00",
            "deliver": "local",
            "workdir": str(tmp_path),  # makes it sequential
        }

        barrier = threading.Barrier(2, timeout=5)

        def slow_run(j, *, defer_agent_teardown=None, **_kw):
            barrier.wait()
            return True, "out", "resp", None

        monkeypatch.setattr(sched, "get_due_jobs", lambda: [job])
        monkeypatch.setattr(sched, "advance_next_runs", lambda *_a, **_kw: 0)
        monkeypatch.setattr(sched, "run_job", slow_run)
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_kw: "/tmp/out")
        monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)

        start = time.monotonic()
        n = sched.tick(verbose=False, sync=False)
        elapsed = time.monotonic() - start

        assert n == 1  # optimistic count
        assert elapsed < 1.0  # did NOT block on the slow workdir job

        barrier.wait()
        time.sleep(0.1)
        sched._shutdown_parallel_pool()


    def test_get_sequential_pool_is_persistent(self):
        """_get_sequential_pool returns the same single-thread pool."""
        import cron.scheduler as sched

        sched._sequential_pool = None
        pool1 = sched._get_sequential_pool()
        pool2 = sched._get_sequential_pool()
        assert pool1 is pool2

        sched._shutdown_parallel_pool()
        assert sched._sequential_pool is None


class TestTickBatchAdvance:
    """The tick's pre-dispatch advance must go through advance_next_runs
    exactly once with the whole due set — a revert to the per-job loop
    (or back to advance_next_run) must fail this test, not slip past the
    helper-level I/O pin."""

    def test_tick_calls_advance_next_runs_once_with_all_due_ids(self, tmp_path, monkeypatch):
        import cron.scheduler as sched

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._running_job_ids.clear()

        jobs = [
            {"id": f"job-{i}", "name": f"Job {i}", "prompt": "test",
             "schedule": "every 5m", "enabled": True,
             "next_run_at": "2020-01-01T00:00:00", "deliver": "local"}
            for i in range(4)
        ]

        advance_calls = []
        monkeypatch.setattr(sched, "get_due_jobs", lambda: jobs)
        monkeypatch.setattr(
            sched, "advance_next_runs",
            lambda ids: advance_calls.append(list(ids)) or len(list(ids)))
        monkeypatch.setattr(sched, "run_job", lambda j, **_kw: (True, "out", "resp", None))
        monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_kw: "/tmp/out")
        monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)

        n = sched.tick(verbose=False)

        assert n == 4
        assert advance_calls == [["job-0", "job-1", "job-2", "job-3"]], (
            f"tick must batch-advance the due set in ONE call; got {advance_calls}")

        sched._shutdown_parallel_pool()
