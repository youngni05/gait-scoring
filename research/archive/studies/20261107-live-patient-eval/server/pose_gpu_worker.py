"""GPU forward pass only; lossless float32 heatmaps for official local decoding.

Same network, preprocessing and chunking as the earlier typically developing run.
Completed videos are skipped on restart. PILOT=1 processes the first video only.
"""
from pathlib import Path
import json, hashlib, time, os
import numpy as np
import cv2, torch
from pose_torch import Body25, probe

HERE = Path(__file__).resolve().parent
PILOT = os.environ.get('PILOT') == '1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, obj):
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(obj, indent=2))
    t.replace(p)


def blob(im):
    h, w = im.shape[:2]
    nh = 368
    nw = 16 * int(nh * w / h / 16 + .5)
    scale = min((nw - 1) / (w - 1), (nh - 1) / (h - 1))
    mat = np.array([[scale, 0, 0], [0, scale, 0]], dtype=np.float64)
    resized = cv2.warpAffine(im, mat, (nw, nh), flags=cv2.INTER_LINEAR)
    return np.ascontiguousarray((resized.astype(np.float32) / 256 - .5).transpose(2, 0, 1)[None])


def main():
    started = time.time()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    probe(HERE / 'pose_probe', 'cuda')
    im = cv2.imread(str(HERE / 'pose_probe/input/frame.png'))
    original = np.load(HERE / 'pose_probe/native_arrays.npz')['blob']
    assert np.array_equal(blob(im), original), 'Image preprocessing mismatch'
    net = Body25(HERE / 'pose_probe').cuda().eval()
    rows = json.loads((HERE / 'download.json').read_text())['videos']
    metadata = {r['file']: r for r in json.loads((HERE / 'video_metadata.json').read_text())}
    rows = [r for r in rows if metadata[r['file']]['fps'] <= 40]
    assert len(rows) == 138, len(rows)
    (HERE / 'heatmaps').mkdir(exist_ok=True)
    indexpath = HERE / 'heatmaps/index.json'
    index = json.loads(indexpath.read_text()) if indexpath.exists() else []
    donepath = HERE / 'videos_done.json'
    complete = json.loads(donepath.read_text()) if donepath.exists() else []
    finished = {c['video'] for c in complete}
    index = [r for r in index if r['video'] in finished]
    for row in sorted(rows, key=lambda r: r['file']):
        if row['file'] in finished:
            continue
        path = HERE / 'videos' / row['file']
        assert sha(path) == row['sha256']
        stem = path.stem
        cap = cv2.VideoCapture(str(path))
        chunk = []
        frame = 0
        with torch.inference_mode():
            while True:
                ok, im = cap.read()
                if not ok:
                    break
                y = net(torch.from_numpy(blob(im)).cuda()).cpu().numpy()[0]
                assert np.isfinite(y).all()
                chunk.append(y)
                frame += 1
                if len(chunk) == 50 or frame == metadata[row['file']]['frames']:
                    start = frame - len(chunk)
                    name = '%s_%06d_%06d.npz' % (stem, start, frame - 1)
                    dest = HERE / 'heatmaps' / name
                    tmp = dest.with_suffix('.tmp')
                    with tmp.open('wb') as f:
                        np.savez_compressed(f, heatmaps=np.stack(chunk))
                    tmp.replace(dest)
                    index.append(dict(file=name, video=row['file'], first=start, last=frame - 1,
                                      sha256=sha(dest), bytes=dest.stat().st_size))
                    save(indexpath, index)
                    chunk = []
                    save(HERE / 'progress.json', dict(video=row['file'], frame=frame, completed_videos=len(complete),
                                                     elapsed=round(time.time() - started)))
        cap.release()
        assert frame == metadata[row['file']]['frames'], (row['file'], frame)
        assert not chunk
        complete.append(dict(video=row['file'], frames=frame))
        save(donepath, complete)
        print(row['file'], frame, 'elapsed', round(time.time() - started), flush=True)
        if PILOT:
            break
    if not PILOT:
        save(HERE / 'complete.json', dict(videos=complete, frames=sum(r['frames'] for r in complete), chunks=len(index),
                                          seconds=time.time() - started))
    print('PILOT_DONE' if PILOT else 'POSE_HEATMAPS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
