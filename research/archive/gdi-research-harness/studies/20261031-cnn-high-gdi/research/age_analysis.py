"""Age-only analysis: approved development cohort and public TD metadata."""
from pathlib import Path
import json
import hashlib
import collections
import numpy as np
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
def describe(values):
    a=np.array(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),sd=float(a.std(ddof=1)),median=float(np.median(a)),min=float(a.min()),max=float(a.max()))
def main():
    path=HERE/'PARTICIPANT_CHARACTERISTICS.xlsx'
    assert hashlib.md5(path.read_bytes()).hexdigest()=='3f70a6b32ae4abe6488a5b3e1d6de7b7'
    wb=openpyxl.load_workbook(path,read_only=True,data_only=True)
    rows=list(wb['TD'].values);assert rows[0][4]=='AGE (years)'
    td={r[1]:float(r[4]) for r in rows[1:] if r[1] and isinstance(r[4],(int,float))}
    inventory=json.loads((HERE/'video_inventory.json').read_text())['files']
    video_people={r['name'].split('/')[2] for r in inventory if r['name'].startswith('VIDEO/TD/') and 'GAIT' in r['name'] and 'SAGITTAL.MOV' in r['name']}
    matched={p:a for p,a in td.items() if p in video_people}
    with np.load(HERE.parents[1]/'20261031-ljmu-normal-cnn/diagnostics/development_age_metadata.npz') as d:
        ages=d['age'];people=d['person_id']
        first=np.array([ages[people==p].min() for p in np.unique(people)])
    edges=np.array([0,5,8,11,14,18,25,60])
    summary={'public_td_metadata':describe(list(td.values())),
             'age_and_video_matched_td':describe(list(matched.values())),
             'public_td_age_counts':dict(sorted(collections.Counter(td.values()).items())),
             'public_td_missing_expected_code':'S18', 'paper_reported_td_n':25,
             'development_earliest_available_visit_per_person':describe(first),
             'development_all_record_age':describe(ages),
             'bin_edges_years':edges.tolist(),
             'development_person_counts':np.histogram(first,edges)[0].tolist(),
             'td_person_counts':np.histogram(list(matched.values()),edges)[0].tolist(),
             'original_2020_cnn_paper_age':{'people':1026,'mean':11,'sd':5.9,'full_histogram_not_available_here':True},
             'original_2008_gdi_normative_age_histogram':'not verified; not the same population as 2020 CNN patients'}
    (HERE/'age_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(10.5,4.6),layout='constrained')
    x=np.arange(len(edges)-1);w=.36
    for delta,values,label,color in [(-w/2,first,'현재 CNN 개발자료 692명 (사람별 첫 가용 방문)','#3975ad'),(w/2,list(matched.values()),'나이와 영상이 연결된 정상 아동 23명','#d48a35')]:
        hist=np.histogram(values,edges)[0]
        bars=ax.bar(x+delta,hist/len(values)*100,w,label=label,color=color)
        for b,n in zip(bars,hist):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.5,str(n)+'명',ha='center',fontsize=9)
    ax.set_xticks(x,['5세 미만','5–7세','8–10세','11–13세','14–17세','18–24세','25세 이상'])
    ax.set_ylabel('각 집단 내 비율 (%)');ax.set_ylim(0,55);ax.legend(frameon=False)
    ax.set_title('나이대는 겹치지만, 정상 아동 후보는 더 어린 쪽에 집중된다',loc='left',pad=15)
    fig.savefig(HERE/'age_distribution.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
