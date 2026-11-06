import sys,json
from pathlib import Path
sys.path.insert(0,'D:/gait-research-data/D_P/movi-gdi-cycles-20261105/plot-deps')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent
data=np.load(HERE/'angle-curves.npz');c=data['curves'];ref=data['reference_curves'];rows=json.loads((HERE/'cycle-diagnostics.json').read_text());kept=[r['curve_index'] for r in rows if not r['body_orientation_flag']]
names=['골반 전후 기울기','골반 좌우 기울기','골반 수평 회전','고관절 굽힘·폄','고관절 모음·벌림','고관절 안쪽·바깥쪽 회전','무릎 굽힘·폄','발목 굽힘·폄','발 진행각']
t=np.linspace(0,100,51);fig,axes=plt.subplots(3,3,figsize=(12,9),constrained_layout=True)
for j,ax in enumerate(axes.flat):
    low,high=np.percentile(ref[:,:,j],[5,95],axis=0);ax.fill_between(t,low,high,color='#a7b2bf',alpha=.4,label='원 정상 기준 5–95백분위')
    for i in kept:ax.plot(t,c[i,:,j],color='#2787b9',alpha=.18,lw=.8)
    ax.plot(t,ref[:,:,j].mean(0),color='#333333',lw=2,label='원 정상 기준 평균');ax.plot(t,c[kept,:,j].mean(0),color='#0074b8',lw=2,label='MoVi 근사 재구성 평균')
    ax.set(title=names[j],xlabel='보행 주기 (%)',ylabel='각도 (도)');ax.grid(alpha=.2)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3);fig.suptitle('MoVi 9개 각도 재구성 — 몸 방향 보조 조건을 통과한 22명·30주기\n정상 기준과의 차이는 측정 오차를 의미하지 않는다',fontsize=14)
fig.savefig(HERE/'nine-angle-curves.png',dpi=150);fig.savefig(HERE/'nine-angle-curves.pdf');plt.close(fig)

# A whole-cycle example selected by source ID order, not its GDI value.
from analyze import source_frame
cy=json.loads((HERE/'cycles.json').read_text());idx=next(r['curve_index'] for r in rows if r['id']==27 and r['side']=='L');r=cy[idx];a=r['start_video_nearest'];b=r['end_video_nearest'];frames=np.rint(np.linspace(a,b,6)).astype(int)
sheet=Image.new('RGB',(960,530),'white');draw=ImageDraw.Draw(sheet)
for i,fr in enumerate(frames):
    im=source_frame(27,int(fr)).resize((320,240));x=(i%3)*320;y=(i//3)*265;sheet.paste(im,(x,y));draw.text((x+5,y+243),f'Subject 27 / left cycle / frame {fr}',fill='black')
sheet.save(HERE/'cycle-example.jpg',quality=90)
