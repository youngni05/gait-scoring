"""Embed computed values into the local comparison viewer."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'output/older-gait-viewer-20261106/older-gait-viewer.html'
d = json.loads((HERE / 'motion-data.json').read_text(encoding='utf-8'))
P = d['people']


def rank(v):
    s = sorted(v)
    return [sum(i for i, q in enumerate(s) if q == x) / s.count(x) for x in v]


def corr(a, b):
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** .5


def rho(a, b):
    return corr(rank(a), rank(b))


speed = [p['gait']['speed'] for p in P]
score = [p['added'] for p in P]
base = [p['baseline'] for p in P]
stroke = [p for p in P if p['cohort'] == 'stroke']
healthy = [p for p in P if p['cohort'] == 'healthy']
t = next(p for p in P if p['id'] == 'TVC4')
s = next(p for p in P if p['id'] == 'SUBJ5')
stats = dict(rho_added=rho(score, speed), rho_baseline=rho(base, speed),
             tvc4_stroke_speed_rank=1 + sum(p['gait']['speed'] > t['gait']['speed'] for p in stroke),
             subj5_healthy_speed_rank=1 + sum(p['gait']['speed'] > s['gait']['speed'] for p in healthy),
             stroke_faster_than_subj5=sum(p['gait']['speed'] > s['gait']['speed'] for p in stroke),
             subj5_age_rank=1 + sum(p['age'] > s['age'] for p in healthy),
             stroke_POMA_ge_27=sum((p['POMA'] or 0) >= 27 for p in stroke))
print(stats)
d['reading'] = (f"두 사람의 순서는 진단명보다 걸음 능력과 더 잘 맞는다. TVC4는 뇌졸중 25명 중 가장 빠르고 "
                f"POMA도 28점 만점에 27점이다. SUBJ5는 건강 11명 중 {stats['subj5_healthy_speed_rank']}번째로 빠르며 84세다. "
                f"36명 전체에서 점수와 속도의 순위상관은 추가 후 {stats['rho_added']:.2f}, 추가 전 {stats['rho_baseline']:.2f}이다.")
(HERE / 'viewer-stats.json').write_text(json.dumps(stats, indent=1), encoding='utf-8')
html = (HERE / 'viewer-template.html').read_text(encoding='utf-8').replace('/*DATA*/null', json.dumps(d, ensure_ascii=False, separators=(',', ':')))
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(html, encoding='utf-8')
print(OUT, len(html))
