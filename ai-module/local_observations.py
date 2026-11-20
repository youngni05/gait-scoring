"""Separate local detail using the independently reviewed, unchanged candidate."""
from pathlib import Path
import hashlib
from local_knee import recover as reviewed_recover

CANDIDATE=Path(__file__).with_name('local_knee.py')
CANDIDATE_SHA256='5f05acdaa7e0e7b119d3dcc896bb2533d892cf1ee5ee5b3eda7a3d2d232b602d'
if hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()!=CANDIDATE_SHA256:
    raise RuntimeError('독립 검산한 국소 무릎 관측 코드와 파일이 다르다.')


def build_local_observations(analysis,frames,recover):
    """Leave the original analysis unchanged; do not feed local rows into history."""
    by_index={f['index']:f for f in frames}
    output={}
    for segment in analysis['segments']:
        selected=[by_index[i] for i in segment['observation']['source_indices']]
        attachment=recover(segment,selected)
        output[segment['segment_id']]=[row for row in attachment['observations']
            if row['status']=='limited_local_observation']
    return output
