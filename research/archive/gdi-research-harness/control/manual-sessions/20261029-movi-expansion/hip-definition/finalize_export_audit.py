"""Verify audit artifacts and add factual findings to the existing results page."""
import hashlib,json,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
DATA=Path('D:/gait-research-data/MoVi');OUT=DATA/'experiments'/'hip-definition'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=read(OUT/'export-fields-audit.json');old=read(OUT/'analysis.json')
assert a['summary']['people']==a['summary']['exact_global_reference_reads']==30
assert a['summary']['lcdata_unusable_subjects']==[49]
assert sha(DATA/'experiments'/'walking-expansion'/'results.json')==old['plan']['source_results_sha256']
for f in a['source_manifest']['files']:assert sha(HERE/'public-source'/f['path'])==f['sha256']
assert a['summary']['tutorial_inputs_inspected']
for f in a['supplementary_tutorial_inputs']['files']:
    assert sha(HERE/'public-source'/f['stored_source_file'])==f['stored_source_sha256']
usable=[r for r in a['rows'] if r['calibration_centroid_diagnostic']]
assert len(usable)==29
assert all(r['affine_translation_vs_translation_field']['max']==0 for r in a['rows'])
review=read(OUT/'marker-review'/'manifest.json')
assert len(review['records'])==8 and review['human_status'] in ['pending','partial','complete']
reviewed=[r for r in review['records'] if r['human_status']=='reviewed_qualitative']
assert (HERE/'marker-review.html').exists()
for r in review['records']:assert Path(r['image']).is_file()
inventory=[]
for p in sorted((DATA/'raw').glob('*.tar')):
    with tarfile.open(p) as tar:
        members=tar.getmembers();files=[m.name for m in members if m.isfile()]
    inventory.append({'name':p.name,'files':len(files),'non_mat_files':[n for n in files if not n.endswith('.mat')]})
verification={'export_fields_audit_sha256':sha(OUT/'export-fields-audit.json'),
 'baseline_results_sha256_unchanged':old['plan']['source_results_sha256'],
 'source_text_files_sha256_verified':len(a['source_manifest']['files']),
 'supplementary_tutorial_inputs_sha256_verified':len(a['supplementary_tutorial_inputs']['files']),
 'global_reference_reads':30,'calibration_centroid_checks':29,
 'forward_inverse_check':'Both coordinate directions checked during 30-subject audit; residual agreement stored per person.',
 'source_inventory':inventory,'human_review_images_prepared':8,'human_review_status':review['human_status'],
 'human_reviewed_case_ids':[r['id'] for r in reviewed],
 'passed':True,'interpretation_limit':'Verified data/field relationships and inspected source scope, not original processing reconstruction or anatomical accuracy.'}
(OUT/'export-audit-verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')

src=a['public_sources'];link=(HERE/'marker-review.html').as_uri()
review_note=f"현재 {len(reviewed)}/8장 사람 검토가 기록됐다."
if reviewed:
    for r in reviewed:
        h=r['human_review']
        overlap='·'.join(map(str,h['qualitative_overlap_marker_ids']))
        unseen='·'.join(map(str,h['unobservable_marker_ids']))
        parts=[]
        if overlap:parts.append(f'{overlap}번 마커는 겹쳐 보임')
        if unseen:parts.append(f'{unseen}번은 판독 불가')
        uncertain='·'.join(map(str,h.get('uncertain_displacement_marker_ids',[])))
        if uncertain:parts.append(f'{uncertain}번 고리가 실제 마커보다 약간 왼쪽이라는 의심(미확정)')
        if h.get('global_offset_hypothesis'):parts.append('전체 고리도 왼쪽으로 밀렸을 가능성(실제 마커 식별이 어려워 미확정)')
        review_note+=f" {r['id']}번 장면: "+', '.join(parts)+'이라는 사용자 판단이다.'
    review_note+=' 이는 보이는 피부 마커의 정성적 대조이며 고관절 중심의 해부학적 정답이나 픽셀 정확도를 확정하지 않는다.'
section=f'''<!-- export-source-audit-start -->
<section id="export-source-audit"><p class="lead"><b>원래 설정 추가 확인: 관절 정의 차이가 원인이라고 확정한 것은 아니다.</b></p>
<p>확보한 30명 MAT와 공식 저장소의 공개 텍스트 소스 11개, Python 노트북 전체 입력 셀, MATLAB Live Script의 입력 코드, 현재 파일 목록·59개 변경 메시지·공개 이슈를 조사했다. 노트북의 내장 영상·그림 출력은 저장하지 않았다. 새 모델을 실행하거나 기존 기준 좌표를 바꾸지 않았다.</p>
<table><tr><th>확인한 대상</th><th>실제 확인 결과</th><th>확인 범위의 한계</th></tr>
<tr><td>기존 실험의 기준 좌표 읽기</td><td>30명·두 시점에서 제공된 전역 관절 좌표를 그대로 읽었다. 누락 마스크를 적용한 값과 저장된 비교 입력이 정확히 일치했다.</td><td>이 일치는 해당 지점이 해부학적으로 맞다는 증명은 아니다.</td></tr>
<tr><td>MAT 내부의 골반 마커·관절 관계</td><td>정적 골반 마커 좌표를 사용할 수 있는 29명에서, 마커 중심과 저장 회전을 통해 원점을 대조하면 고관절이 Bell 계수 형태의 관계와 수치적으로 맞았다. 2·31번의 저장된 골반 원점에는 고정된 전역 이동 차이가 있었다.</td><td>골반 너비를 제공된 양 고관절 간격에서 역산한 관계 검산이다. 원래 골반 너비 설정을 독립적으로 복원한 것이 아니다. 49번의 정적 골반 마커 좌표는 0이어서 이 검산에 사용할 수 없었다.</td></tr>
<tr><td>생성·내보내기 소스</td><td>공식 예제는 이미 계산된 virtualMarkerLocation을 읽고 영상에 투영한다. utils.py와 Python 노트북의 인체 모델 계산·축 변환은 AMASS 경로다. 확인한 자료에서 원래 Visual3D 모델과 MAT 내보내기 코드를 찾지 못했다.</td><td>현재 공개 코드의 입력은 확인했지만 모든 과거 파일을 검토한 것은 아니다. 논문은 일부 전처리 코드를 요청하면 제공할 수 있다고 안내한다.</td></tr></table>
<p>PELVIS는 네 골반 마커의 평균, MHIP는 양 고관절의 평균과 수치적으로 일치했다. 하지만 이 이름이나 MAT의 segmentType=Visual_3D만으로 원래 CODA 모델의 원점·마커 반지름·개인별 골반 너비 설정을 확정할 수는 없다. 저장된 골반 변환 원점을 원래 해부학적 CODA 원점과 동일한 것으로 해석할 근거가 부족하다. 두 원점의 위치가 달라 보인다는 사실만으로 자료의 생성 오류를 판정할 수 없다. 원래 고관절 생성에 어떤 좌표계가 적용되고 내보내기에서 어떤 변환이 이루어졌는지는 별도 모델·처리 코드가 필요하다.</p>
<p class="note"><b>앞선 수치의 해석:</b> 17.9→8.1픽셀, 14.0→10.9°는 문서식으로 근사한 다른 참고 좌표를 사용했을 때의 변화다. 원래 MoVi 처리 과정을 복원하거나 MediaPipe와 MoVi가 서로 다른 해부학적 지점을 정의한다고 증명한 결과가 아니다. 정의 차이·기준 생성 과정·모델 추정 오류의 몫을 아직 분리하지 못했다.</p>
<p><b>사람이 확인할 부분:</b> 영상에 실제로 보이는 피부 마커와 투영한 초록 고리가 맞는지 <a href="{link}">읽기 전용 마커 비교 화면</a>을 준비했다. 몸속 고관절 중심을 추측하지 않으며, 안 보이는 마커는 판독 불가로 남긴다. {review_note}</p>
<p>근거: <a href="{src['tutorial']}">공식 MATLAB 예제</a> · <a href="{src['projector']}">공식 영상 투영 코드</a> · <a href="{src['amass_converter']}">AMASS 변환 코드</a> · <a href="{src['author_v3d_fk_reply']}">저자의 Visual3D 복원 코드 관련 답변</a> · <a href="{src['paper']}">원 논문 §2.7·§5</a> · <a href="{(OUT/'export-fields-audit.json').as_uri()}">30명 필드 대조 기록</a> · <a href="{(OUT/'export-audit-verification.json').as_uri()}">검증·확보 파일 범위</a></p></section>
<!-- export-source-audit-end -->'''
report=HERE.parents[1]/'20261028-movi-totalcapture-acquisition'/'experiment-results.html'
text=report.read_text(encoding='utf-8');start='<!-- export-source-audit-start -->';end='<!-- export-source-audit-end -->'
if start in text:
    lo=text.index(start);hi=text.index(end,lo)+len(end);text=text[:lo]+section+text[hi:]
else:text=text.replace('<!-- hip-definition-20261029-start -->',section+'\n<!-- hip-definition-20261029-start -->',1)
assert text.count(start)==1
report.write_text(text,encoding='utf-8')
review_html=HERE/'marker-review.html';review_text=review_html.read_text(encoding='utf-8')
status=f'<p id="human-review-status" class="note">{review_note}</p>'
import re
if 'id="human-review-status"' in review_text:
    review_text=re.sub(r'<p id="human-review-status"[^>]*>.*?</p>',status,review_text,count=1,flags=re.S)
else:review_text=review_text.replace('<details ',status+'<details ',1)
review_html.write_text(review_text,encoding='utf-8')
print(f'Verified: 30 exact reference reads; 29 calibration-centroid checks; 11 source hashes; 8 review images. Human review: {len(reviewed)}/8 recorded.')
