"""Concurrency hardening for the quota tracker (pre-live punch-list item #5).

`record_call` does an unlocked load -> mutate -> save. The OddsPapi counter is
a read-modify-write (`calls_this_month += 1`), so concurrent callers lose
increments -> the count under-reports -> the quota guardrail can let you blow
the monthly cap. This stress test asserts no increments are lost under heavy
contention.

QUOTA_FILE (and the lock, once added) are monkeypatched onto a temp path.
"""
import threading

from filelock import FileLock

import api.quota_tracker as quota_tracker


def test_concurrent_record_call_loses_no_increments(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        quota_tracker, "QUOTA_FILE", str(tmp_path / "api_quota.json")
    )
    # Isolate the lock file too, once the fix introduces it.
    if hasattr(quota_tracker, "_QUOTA_LOCK"):
        monkeypatch.setattr(
            quota_tracker, "_QUOTA_LOCK",
            FileLock(str(tmp_path / "api_quota.json.lock")),
        )

    n_threads, per_thread = 12, 40
    expected = n_threads * per_thread

    def worker():
        for _ in range(per_thread):
            quota_tracker.record_call("oddspapi")

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    used = quota_tracker.read_quota()["oddspapi"]["used"]
    assert used == expected, f"lost increments: {used} != {expected}"
