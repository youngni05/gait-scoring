"""Optional local QA: synthetic rectangles only, no person or clinical data.
Requires the review environment's PyAV and NumPy. Not an app dependency.
Outputs remain in the Git-ignored local-test-data directory.
"""
from pathlib import Path
from fractions import Fraction
import json
import av
import numpy as np

root = Path(__file__).resolve().parents[1] / 'local-test-data/stage3-synthetic'
root.mkdir(parents=True, exist_ok=True)
for name, width, height, count, variable, rate in [
    ('landscape', 320, 180, 180, False, 30),
    ('portrait', 180, 320, 90, False, 30),
    ('variable', 320, 180, 90, True, 30),
    ('high-fps', 320, 180, 300, False, 120),
]:
    path = root / (name + '.mp4')
    with av.open(str(path), 'w') as output:
        stream = output.add_stream('libx264', rate=rate)
        stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
        stream.time_base = stream.codec_context.time_base = Fraction(1, 30000)
        pts = 0
        for index in range(count):
            pixels = np.zeros((height, width, 3), dtype=np.uint8)
            pixels[:, :, 1] = 60
            left = (index * 3) % (width - 60)
            pixels[20:80, left:left + 60, 0] = 230
            frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            frame.pts, frame.time_base = pts, Fraction(1, 30000)
            for packet in stream.encode(frame):
                output.mux(packet)
            pts += 2000 if variable and index % 2 else 30000 // rate
        for packet in stream.encode():
            output.mux(packet)
    with av.open(str(path)) as source:
        times = [float(frame.time) for frame in source.decode(video=0)]
    (root / (name + '.expected.json')).write_text(json.dumps(times), encoding='utf-8')
    print(name, len(times), times[:5], times[-1])
(root / 'broken.mp4').write_text('Intentionally invalid synthetic test video.', encoding='utf-8')
