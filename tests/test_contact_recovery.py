"""A side-label collapse must not erase foot timing or invent anatomical labels."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai-module"))
from contact_review import METHOD, SEPARATION_METHOD, recover_rhythm


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.frames = []
        for i in range(106):
            width = .14 * math.sin(math.pi * i / 15) ** 2
            points = [[.5, .8, 0., 1., None] for _ in range(33)]
            for j in range(27, 33, 2):
                points[j][0] -= width / 2
                points[j + 1][0] += width / 2
            self.frames.append(dict(index=i, source_time_ms=i * 1000 / 30, landmarks=points))
        self.tracking = dict(transitions=[dict(source_time_ms=400), dict(source_time_ms=900)],
                             checked_transitions=105, possible_transitions=105,
                             leg_scale_px=150, contact_radius_ms=75)
        self.gait = dict(method_id=METHOD, quality_reasons=[], steps=[], rejected_pairs={}, metrics={},
                         events=[dict(source_time_ms=f["source_time_ms"]) for f in self.frames[7::15]])

    def run_measurement(self):
        recover_rhythm(self.gait, self.frames, {"width": 640}, self.tracking)

    def test_timing_survives_label_collapse(self):
        self.run_measurement()
        self.assertEqual(self.gait["method_id"], SEPARATION_METHOD)
        self.assertEqual(len(self.gait["steps"]), 6)
        self.assertAlmostEqual(self.gait["metrics"]["cadence"]["value"], 120, delta=2)
        self.assertEqual({e["side"] for e in self.gait["events"]}, {"unknown"})

    def test_occluded_interval_is_not_bridged(self):
        self.frames[42]["landmarks"][27][3] = .1
        self.run_measurement()
        self.assertEqual(self.gait["method_id"], METHOD)

    def test_clock_gap_is_not_bridged(self):
        for f in self.frames[42:]:
            f["source_time_ms"] += 500
        self.run_measurement()
        self.assertEqual(self.gait["method_id"], METHOD)

    def test_reliable_alternating_steps_keep_the_original_method(self):
        self.gait["steps"] = [dict(start_ms=1200, stop_ms=1700, landing_side="left"),
                              dict(start_ms=1700, stop_ms=2200, landing_side="right")]
        self.run_measurement()
        self.assertEqual(self.gait["method_id"], METHOD)


if __name__ == "__main__":
    unittest.main()
