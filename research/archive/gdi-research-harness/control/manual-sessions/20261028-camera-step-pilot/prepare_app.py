from pathlib import Path
import json, hashlib, shutil
base=Path('C:/Users/jun/Desktop/D_P')
h=base/'gdi-research-harness'
s=h/'control/manual-sessions/20261028-camera-step-pilot'
d=h/'app-development/camera-preparation'
assert not d.exists(), 'Do not overwrite existing work'
d.mkdir(parents=True)
source=base/'2026-ITEC0401/app'
measure=base/'ai-measurement-review/app'
manifest=[]
def copy_file(p, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    data=p.read_bytes(); dest.write_bytes(data)
    manifest.append({'source':str(p),'destination':str(dest),'sha256':hashlib.sha256(data).hexdigest()})
for p in source.rglob('*'):
    rel=p.relative_to(source)
    if any(x in ('node_modules','dist','.git','supabase') for x in rel.parts): continue
    if not p.is_file() or p.name.startswith('.env'): continue
    if p.suffix not in ('.ts','.tsx','.css','.json','.html','.mjs','.svg') and p.name!='.gitignore': continue
    if p.name=='package-lock.json': continue
    copy_file(p,d/rel)
stage=d/'packages/web/src/prepared-camera'
for name in ('camera.ts','landmarker.ts','frameLoop.ts','frameLoop.test.ts'):
    copy_file(measure/'packages/web/src/measurement'/name,stage/name)
p=stage/'frameLoop.ts'
p.write_text(p.read_text(encoding='utf-8').replace("from '@sppb/core'", "from './poseTypes'"),encoding='utf-8')
pose=measure/'packages/core/src/pose.ts'
manifest.append({'source':str(pose),'destination':'extracted:poseTypes.ts','sha256':hashlib.sha256(pose.read_bytes()).hexdigest()})
text=pose.read_text(encoding='utf-8').split('export interface FrameSize')[0]
(stage/'poseTypes.ts').write_text(text,encoding='utf-8')
config=measure/'packages/web/src/measurement/config.ts'
manifest.append({'source':str(config),'destination':'extracted:config.ts','sha256':hashlib.sha256(config.read_bytes()).hexdigest()})
text=config.read_text(encoding='utf-8').split('/** [시작]')[0]
(stage/'config.ts').write_text(text,encoding='utf-8')
(stage/'index.ts').write_text("// 연결 준비용 내보내기다. 화면·저장·점수 계산과는 아직 연결하지 않았다.\nexport { startCamera, stopCamera, cameraErrorMessage } from './camera';\nexport { createLandmarker, closeLandmarker } from './landmarker';\nexport { startFrameLoop } from './frameLoop';\nexport type { Landmark, PoseFrame } from './poseTypes';\n",encoding='utf-8')
local={"extends":"./tsconfig.json","compilerOptions":{"paths":{"@sppb/core":["../core/src/index.ts"],"@mediapipe/tasks-vision":[str(measure/'node_modules/@mediapipe/tasks-vision/vision.d.ts')]}}}
(d/'packages/web/tsconfig.local.json').write_text(json.dumps(local,indent=2),encoding='utf-8')
(d/'vitest.local.config.mjs').write_text("import { defineConfig } from 'vitest/config';\nimport { fileURLToPath } from 'node:url';\nexport default defineConfig({resolve:{alias:{'@sppb/core':fileURLToPath(new URL('./packages/core/src/index.ts',import.meta.url))}},test:{include:['packages/**/*.test.ts'],environment:'node'}});\n",encoding='utf-8')
# Package/lock remain original declarations: the staged files are not connected to the runtime.
(d/'README.md').write_text('''# 카메라 연결 준비 작업본

기존 관리 앱의 현재 소스·설정을 복사하고 `packages/web/src/prepared-camera/`에 측정 앱의 카메라·관절 추정 모듈을 준비했다. 원본의 로컬 변경도 포함한다. Git 작업트리는 아니며 소스 사본이다.

카메라 → MediaPipe → 시각·관절 좌표 또는 미검출(null) 콜백까지의 기존 부품이다. 화면에서는 아직 사용하지 않는다. SPPB 판정·저장·임상 기준은 추가하지 않았다. 영상 녹화 기능도 아니다.

`index.ts`가 이후 화면에서 가져올 진입점이다. `frameLoop.ts`는 측정 앱의 전체 core 대신 추출한 두 자료형만 참조한다. `config.ts`에는 기존 모델 URL·WASM 경로·카메라 조건·추정 신뢰도만 남겼다. 임상 측정 실패 기준을 새 앱 정책으로 가져오지 않았다.

환경 파일·서버키·node_modules·git·빌드·DB 설정·잠금 파일은 복사하지 않았다. 이 사본은 독립 설치가 끝난 배포본이 아니다. `tsconfig.local.json`은 기존 MediaPipe 1.1.0 설치 위치와 이 사본의 core를 연결해 로컬 타입 검사한다. 새로 패키지를 설치하지 않았다. 실제 화면에 연결할 때는 MediaPipe 의존성 및 잠금 파일을 확정하고 WASM 자산을 같은 출처에 준비해야 한다.

관리 화면은 아직 모의 측정·SPPB 저장 구조다. 다음은 이 모의 흐름과 별개인 카메라 미리보기 화면을 연결하고 대상자 선택·중단·정리 동작을 실제 브라우저에서 확인하는 것이다. 서버 저장·영상 보관·걸음 계산은 별도 결정과 검증이 필요하다.
''',encoding='utf-8')
(s/'app_source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(s/'app_plan.md').write_text('''사용자 요청: 별도 사본에서 기존 카메라·관절 추정 연결 준비. 원본 소스와 현재 변경 보존. 새로운 저장/판정/화면 연결 설계는 적용하지 않는다. 기존 camera/landmarker/frameLoop와 필요한 타입만 준비하고 관리 앱 전체 타입 검사와 기존 테스트를 사본에서 실행한다. 합성 테스트 통과와 실제 카메라 정확도는 구분한다. 타입 검사 시 기존 설치의 1.1.0을 명시해 루트 1.0.1을 실수로 사용하지 않는다.
''',encoding='utf-8')
print(json.dumps({'source_files':len(manifest),'work_copy':str(d),'new_install':False}))
