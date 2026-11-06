"""Plot the identified direction-reversal window; does not alter inputs or scores."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import prepare as p
H=p.H;B=p.B

def main():
    d=np.load(B/'input.npz');audit=p.read(H/'input-audit.json');i=d['ids'].tolist().index('sub-275');leg=audit['records'][i]['leg_length_m'];S=d['sequence'][i]
    rel=S[:32].T.reshape(91,16,2)*leg;root=S[-2:].T*leg;points=rel+root[:,None];times=np.arange(91)/30
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].plot(times,root[:,0],color='#2563eb');axes[0].set(xlabel='구간 시간 (초)',ylabel='투영 진행 방향 위치 (m)',title='순이동 방향과 반대인 변화가 섞인 구간');axes[0].grid(alpha=.2)
    edges=[('Neck','RShoulder'),('Neck','LShoulder'),('RShoulder','RElbow'),('RElbow','RWrist'),('LShoulder','LElbow'),('LElbow','LWrist'),('Neck','midHip'),('midHip','RHip'),('midHip','LHip'),('RHip','RKnee'),('RKnee','RAnkle'),('RAnkle','RHeel'),('LHip','LKnee'),('LKnee','LAnkle'),('LAnkle','LHeel')]
    for frame,color in zip([0,30,60,90],['#93c5fd','#60a5fa','#2563eb','#1e3a8a']):
        xy=points[frame]
        for a,b in edges:
            u=p.JOINTS.index(a);v=p.JOINTS.index(b);axes[1].plot(xy[[u,v],0],xy[[u,v],1],color=color,alpha=.8)
        axes[1].scatter(xy[:,0],xy[:,1],s=7,color=color,label=f'{frame/30:.0f}초')
    axes[1].axis('equal');axes[1].set(title='같은 구간의 투영 관절',xlabel='진행 방향 (m)',ylabel='골반 평균 높이 기준 (m)');axes[1].legend(fontsize=8)
    fig.suptitle('sub-275 입력 점검: 평균 절대속도 0.872m/s / 순이동속도 0.520m/s',fontsize=11)
    fig.text(.5,.01,'관절 좌표 재구성. 실제 회전인지 좌표 오차인지는 RGB·단계 주석 없이 확정할 수 없음. 원래 분석에서 임의 제외하지 않음.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(H/'direction-reversal-QA.png',dpi=150);plt.close(fig);print('QA_PLOT_WRITTEN')

if __name__=='__main__':main()
