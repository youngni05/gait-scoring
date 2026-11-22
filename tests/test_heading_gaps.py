"""Exercise the automatic segmenter on native frames with known direction changes."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai-module"))
from segmentation import detect


def recording():
    frames = []
    for i in range(70):
        pts = [[.5, .5, 0., 1.] for _ in range(33)]
        pts[0][0] = .55
        for side in (0, 1):
            ankle = .5 + (-1)**side * .1 * math.sin(i / 4)
            pts[27 + side][:2] = [ankle, .9]
            pts[29 + side][:2] = [ankle, .9]
            pts[31 + side][:2] = [ankle + .05, .9]
        frames.append(dict(index=i, source_time_ms=i*1000/30, landmarks=pts))
    return dict(image=dict(width=640, height=480), frames=frames)


class HeadingGaps(unittest.TestCase):
    def test_one_or_two_disagreements_but_not_three(self):
        for n in (1, 2, 3):
            pose = recording()
            for f in pose["frames"][30:30+n]:
                f["landmarks"][0][0] = .45
            result = detect(pose)
            self.assertEqual([v["frame_index"] for v in result["interpolated_heading_frames"]],
                             list(range(30, 30+n)) if n <= 2 else [])
            self.assertEqual(len(result["segments"]), 1 if n <= 2 else 2)

    def test_edges_pose_loss_timestamp_gap_and_turn(self):
        pose = recording()
        frames = pose["frames"]
        frames[0]["landmarks"][0][0] = .45
        frames[16]["landmarks"] = None
        for f in frames[30:]:
            f["source_time_ms"] += 250
        for f in frames[47:]:
            pts = f["landmarks"]
            pts[0][0] = .45
            for side in (0, 1):
                pts[31 + side][0] = pts[29 + side][0] - .05
        result = detect(pose)
        self.assertFalse(result["interpolated_heading_frames"])
        self.assertEqual(result["boundaries"][0]["before_frame_index"], 30)
        self.assertTrue({"direction_cues_disagree", "pose_absent", "direction_transition_margin"}
                        <= {v["reason"] for v in result["excluded"]})
        self.assertEqual(result["segments"][-1]["direction"], "image_left")


if __name__ == "__main__":
    unittest.main()
