"""Display all prespecified conditions, including adverse subgroup outcomes."""
from pathlib import Path
import sys,importlib.util
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
if importlib.util.find_spec('matplotlib') is None:
    sys.path.insert(0,str(HERE.parents[2]/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

def run():
    font=Path('C:/Windows/Fonts/malgun.ttf')
    if font.exists():plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    effects=pd.read_csv(HERE/'aggregate-factorial-effects.csv');effects=effects[effects.draw.eq(0)]
    cells=[('short',1),('short',3),('short',5),('long',1),('long',3),('long',5)]
    contrasts=['input_mean','structure_core','combined'];names=['입력 추가(B)','구조 변경(C)','둘 다(D)']
    rows=['4–10초 · 1구간','4–10초 · 3구간','4–10초 · 5구간','30–60초 · 1구간','30–60초 · 3구간','30–60초 · 5구간']
    fig,axes=plt.subplots(2,3,figsize=(16,8.4),layout='constrained')
    for i,target in enumerate(['sppb','tug']):
        unit='점' if target=='sppb' else '초';tail='낮은 SPPB' if target=='sppb' else '긴 TUG'
        for j,(group,metric,title) in enumerate([('all','mae','전체 MAE'),('tail','mae',tail+' MAE'),('tail','error_p90',tail+' 큰 오차(P90)')]):
            matrix=np.empty((6,3))
            for y,(band,k) in enumerate(cells):
                for x,c in enumerate(contrasts):
                    q=effects[effects.target.eq(target)&effects.band.eq(band)&effects.k.eq(k)&effects.contrast.eq(c)&effects.group.eq(group)]
                    assert len(q)==1;matrix[y,x]=float(q.iloc[0][metric+'_change_mean'])
            ax=axes[i,j];limit=max(float(abs(matrix).max()),1e-6)
            im=ax.imshow(matrix,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
            ax.set_xticks(range(3),names,fontsize=10);ax.set_yticks(range(6),rows,fontsize=10)
            ax.set_title(f'{target.upper()} · {title} 변화 ({unit})',fontsize=12,pad=12)
            ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,6,1),minor=True);ax.grid(which='minor',color='white',linewidth=1.5);ax.tick_params(which='minor',bottom=False,left=False)
            for y in range(6):
                for x in range(3):ax.text(x,y,f'{matrix[y,x]:+.3f}',ha='center',va='center',fontsize=11,color='white' if abs(matrix[y,x])>.58*limit else 'black')
            ax.axhline(2.5,color='#333333',linewidth=1.4)
            fig.colorbar(im,ax=ax,shrink=.8,pad=.025)
    fig.suptitle('같은 사람·구간에서 기존 Deep Sets(A) 대비 오차 변화\n음수·파란색: 개선 / 양수·붉은색: 악화 · 초기값 5개의 평균 · 구간 선택 draw 0',fontsize=15)
    fig.savefig(HERE/'factorial-effects.png',dpi=180)
    plt.close(fig)
    print('SAVED factorial-effects.png',flush=True)
if __name__=='__main__':run()
