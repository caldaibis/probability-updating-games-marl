from __future__ import annotations

from typing import Optional

from ray.tune import Stopper


class TotalTimeStopper(Stopper):
    def __init__(self, total_time_s: Optional[int] = None):
        self._total_time_s = total_time_s

    def __call__(self, trial_id, result):
        return result["time_total_s"] > self._total_time_s

    def stop_all(self):
        return False
