from __future__ import annotations

from typing import Optional

from ray.tune import Stopper


class ConjunctiveStopper(Stopper):
    """Combine several stoppers via 'AND'."""
    def __init__(self, *stoppers: Stopper):
        self._stoppers = stoppers

    def __call__(self, trial_id, result):
        return all(s(trial_id, result) for s in self._stoppers)

    def stop_all(self):
        return all(s.stop_all() for s in self._stoppers)


class TotalTimeStopper(Stopper):
    def __init__(self, total_time_s: Optional[int] = None):
        self._total_time_s = total_time_s

    def __call__(self, trial_id, result):
        return result["time_total_s"] > self._total_time_s

    def stop_all(self):
        return False
