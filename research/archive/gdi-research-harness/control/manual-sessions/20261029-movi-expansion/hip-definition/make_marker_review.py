"""Read-only review of observable skin markers, not hidden anatomical hip centers."""
import io,json,sys,tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from run_ready_subjects import ROOT,OUT as BASE,np,loadmat,cv2,project,digest
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent;OUT=ROOT/'experiments'/'hip-definition'/'marker-review'
OUT.mkdir(exist_ok=True)
if (OUT/'manifest.json').exists():
    existing=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    if any(r.get('human_status')!='pending' for r in existing['records']):
        raise RuntimeError('Human reviews already exist. Do not regenerate or overwrite the reviewed frame set.')
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
titlefont=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',22)
markers=['LFWT','RFWT','LBWT','RBWT','LKNE','RKNE','LANK','RANK']
chosen=[1,2,31,49]
records=[]
with tarfile.open(ROOT/'raw'/'Camera Parameters.tar') as tar:
    def get(name):
        with np.load(io.BytesIO(tar.extractfile('Calib/'+name+'.npz').read())) as z:return dict(z)
    cal={c:(get('cameraParams_'+c),get('Extrinsics_'+c)) for c in ['PG1','PG2']}
for s in chosen:
    source=ROOT/'reference'/f'F_v3d_Subject_{s}.mat'
    m=loadmat(source,simplify_cells=True)[f'Subject_{s}_F']['move']
    names=[n if isinstance(n,str) else '' for n in m['markerName']];indices=[names.index(n) for n in markers]
    for camera in ['PG1','PG2']:
        with np.load(BASE/f'subject-{s:02d}'/'full'/f'{camera}_800_comparison.npz') as z:frames=z['frames']
        mocap=(frames-1)*4;physical=m['markerLocation'][mocap][:,indices]
        valid=~m['markerGaps'][mocap][:,indices].astype(bool)
        projected=np.full(physical.shape[:-1]+(2,),np.nan)
        projected[valid]=project(physical[valid],*cal[camera])
        inside=(np.isfinite(projected)&(projected>=0)&(projected<[800,600])).all(axis=-1)
        candidates=np.flatnonzero((inside&valid).all(axis=1))
        mode='nearest walking midpoint with 8 valid, in-frame reference skin markers'
        if not len(candidates):
            candidates=np.flatnonzero((inside[:,:4]&valid[:,:4]).all(axis=1))
            mode='fallback: nearest walking midpoint with 4 valid, in-frame pelvic skin markers'
        assert len(candidates)>0
        idx=int(candidates[np.argmin(abs(candidates-(len(frames)-1)/2))]);frame=int(frames[idx])
        video=ROOT/'raw'/f'F_{camera}_Subject_{s}_L.avi'
        cap=cv2.VideoCapture(str(video));assert cap.get(cv2.CAP_PROP_FPS)==30
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame-1);ok,im=cap.read();cap.release();assert ok
        xy=projected[idx];good=inside[idx]&valid[idx]
        lo=np.floor(xy[good].min(axis=0)-45).astype(int);hi=np.ceil(xy[good].max(axis=0)+45).astype(int)
        lo=np.maximum(lo,[0,0]);hi=np.minimum(hi,[800,600])
        crop=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).crop((int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])))
        scale=max(2.,500/crop.width);size=(round(crop.width*scale),round(crop.height*scale))
        original=crop.resize(size,Image.Resampling.NEAREST);overlay=original.copy();draw=ImageDraw.Draw(overlay)
        for j,p in enumerate(xy):
            if not good[j]:continue
            x,y=(p-lo)*scale;radius=5*scale
            draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline='#21ff64',width=2)
            draw.text((x+radius+2,y-16),str(j+1),font=titlefont,fill='#21ff64',stroke_width=2,stroke_fill='black')
        canvas=Image.new('RGB',(size[0]*2+24,size[1]+170),'white');draw=ImageDraw.Draw(canvas)
        rid=len(records)+1
        draw.text((12,6),f'{rid:02d} · 참여자 {s} · {camera} · 영상 {frame}프레임',font=titlefont,fill='#152533')
        draw.text((12,40),'원본',font=font,fill='black');draw.text((size[0]+24,40),'초록 고리: 마커 좌표의 투영',font=font,fill='black')
        canvas.paste(original,(0,70));canvas.paste(overlay,(size[0]+24,70));draw=ImageDraw.Draw(canvas)
        for line,text in enumerate(['1 왼골반앞  2 오른골반앞  3 왼골반뒤  4 오른골반뒤',
                                    '5 왼무릎바깥  6 오른무릎바깥  7 왼발목바깥  8 오른발목바깥',
                                    '사람 기준 좌우. 안 보이는 마커는 판독 불가로 남긴다.']):
            draw.text((12,size[1]+76+line*27),text,font=font,fill='#152533')
        target=OUT/f'{rid:02d}-subject-{s:02d}-{camera}.png';canvas.save(target)
        records.append({'id':rid,'subject':s,'camera':camera,'video_frame_one_based':frame,
          'mocap_frame_one_based':int(mocap[idx])+1,'selection':mode,'crop_xyxy':[int(v) for v in [*lo,*hi]],
          'marker_names':markers,'projected_marker_xy':xy.tolist(),'in_frame_and_valid':good.tolist(),
          'reference_sha256':digest(source),'image':str(target),'human_status':'pending',
          'question':'Do clearly visible skin markers coincide with projected green rings? Mark non-visible markers unobservable; do not infer internal hip center.'})
manifest={'purpose':'Qualitative video/projection alignment review of visible physical markers, not anatomical hip-center validation.',
          'case_selection':'subject 1 baseline; subjects 2/31 stored-origin exceptions; subject 49 static lcData unavailable. Purposive audit examples, not a representative accuracy sample.',
          'records':records,'human_status':'pending'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8')
cards=''.join(f'<details {"open" if r["id"]==1 else ""}><summary>{r["id"]}번 · 참여자 {r["subject"]} · {r["camera"]} · {r["video_frame_one_based"]}프레임</summary><img loading="lazy" src="{Path(r["image"]).as_uri()}" alt="원본과 실제 마커 좌표 투영 비교"></details>' for r in records)
html='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>실제 마커와 좌표의 영상 정렬 확인</title><style>body{font-family:"Malgun Gothic",sans-serif;max-width:1150px;margin:30px auto;padding:0 20px;line-height:1.7;background:#fafbfd;color:#18212b}img{max-width:100%;height:auto}details{padding:14px;margin:16px 0;background:white;border:1px solid #cfd9e1}summary{cursor:pointer;font-weight:bold}.note{background:#fff1d6;padding:16px}</style>
<h1>보이는 마커와 초록 고리가 맞는지만 확인한다</h1><p>왼쪽은 원본, 오른쪽은 같은 프레임에 정밀 장비의 물리적 마커 좌표를 겹친 것이다. 작은 흰 마커가 보이면 초록 고리 중심과 대체로 겹치는지 살핀다. 옷이나 반대쪽 몸에 가려져 안 보이면 <b>판독 불가</b>다.</p>
<p class="note">몸속 고관절 중심을 찾는 과제가 아니다. 가상 고관절 점과 근사 재구성 점은 이 화면에 넣지 않았다. 이 검토만으로 픽셀 단위 정확도나 해부학적 정답을 확정하지 않는다.</p>
<p>우선 1번 장면부터 확인하면 된다. 답은 이 대화에 <b>“1번: 1·2·5는 맞아 보임, 3·4는 안 보임, 7·8은 같은 방향으로 밀림”</b>처럼 남기면 된다. 전반적으로 맞는지, 한 방향으로 밀리는지, 구분하기 어려운지가 핵심이다. 파일로 저장해 줄 필요는 없다.</p>'''+cards+'</html>'
(HERE/'marker-review.html').write_text(html,encoding='utf-8')
print('Prepared',len(records),'paired marker-review images; human review pending.')
