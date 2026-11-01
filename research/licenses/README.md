# 출처·라이선스·배포 목적

**이 프로젝트의 모델 묶음은 대학 과목 제출을 위한 비상업적 연구 결과로 공개한다.**

This release shares non-commercial academic research results for university coursework.

이 표시는 이번 배포의 목적을 설명한다. 원자료·기존 코드·사전학습 모델의 라이선스를 대체하거나, 그 라이선스가 허용하는 권리를 축소하는 별도 조건은 아니다. 이 안내만으로 상업적 제품·서비스에 필요한 권리를 모두 확보했다고 해석해서도 안 된다. 원권리자의 이용 조건과 필요한 별도 권리를 각각 확인해야 한다.

| 사용 자료·구성 요소 | 원문에 명시된 라이선스 | 적용 범위·동봉 원문 |
|---|---|---|
| Stanford mobile-gaitlab 처리 관절 궤적 | CC BY-NC 2.0 | GDI 학습에 사용한 원자료의 조건. [원문](CC-BY-NC-2.0.html) |
| Stanford mobile-gaitlab 코드 | Apache 2.0 | 참고한 CNN 코드의 조건. [원문](Apache-2.0-mobile-gaitlab.txt) |
| MotionBERT 코드 | Apache 2.0 | 포함한 MotionBERT 코드의 기존 라이선스. [원문](Apache-2.0-MotionBERT-code.txt) |
| 공식 MotionBERT 모델 배포 저장소 | Apache 2.0 LICENSE 제공 | Hugging Face 배포 저장소에서 가져온 원문을 별도로 보존. [원문](Apache-2.0-MotionBERT-model.txt) |
| OlderAdults44 | CC BY 4.0 | SPPB 실험 원자료의 조건. [원문](CC-BY-4.0.txt) |
| Neuromuscular OpenCap Dataset v2.0 | CC BY 4.0 | TUG 실험 원자료의 조건. [원문](CC-BY-4.0.txt) |

데이터 라이선스와 코드 라이선스는 구분한다. 데이터에 붙은 라이선스가 모든 학습 가중치의 라이선스를 자동으로 결정한다고 선언하지 않는다. 이 저장소나 가중치 전체에 하나의 새 라이선스를 일괄 부여하는 문서도 아니다. 출처별 기존 고지와 비상업적 학술 배포 목적을 함께 보존한다.

## 모델 압축에 들어 있는 안내

각 압축은 모델 폴더 안에 다음 파일을 포함한다.

- `RESEARCH_USE.md`: 대학 과목 제출용 비상업적 연구라는 배포 목적.
- `SOURCES.md`: 자료·모델의 저자, 출처 URL, 논문·데이터 DOI와 라이선스 링크.
- `MODIFICATIONS.md`: 원자료 처리·학습·공개용 변환 및 2026-11-02 고지 추가 내역.
- `NOTICE.md`: 기존 권리 고지의 적용 범위와 포함 파일 안내.
- `LICENSES/`: 해당 묶음과 관련된 공식 라이선스 원문 사본.

2026-11-02 안내 추가에서는 모델 파일과 기존 `weights-manifest.json`의 바이트를 보존했다. 모델의 재학습·가중치 변경은 없으며 압축 파일의 SHA-256만 달라졌다. 이전·새 압축의 대응과 파일 대조 결과는 [package-notices-update-20261102.json](../package-notices-update-20261102.json)에 있다. 초기 재추론 검산 기록에는 이번 압축 갱신 정보를 덧붙여 현재 다운로드 파일의 지문도 확인할 수 있게 했다.

## 원문 출처

- Stanford mobile-gaitlab: https://github.com/stanfordnmbl/mobile-gaitlab — 처리 궤적과 코드의 조건이 다름을 README의 License 항목에서 확인했다. 데이터: https://doi.org/10.18735/j0rz-0k12
- MotionBERT 코드: https://github.com/Walter0807/MotionBERT ; 실제 사전학습 가중치 배포 저장소: https://huggingface.co/walterzhu/MotionBERT
- OlderAdults44: https://doi.org/10.17632/xgw6bg3g8h.1
- Neuromuscular OpenCap Dataset v2.0: https://doi.org/10.5281/zenodo.22309771 — 공식 API의 `metadata.license.id=cc-by-4.0`을 확인했다.
- CC BY 4.0 원문: https://creativecommons.org/licenses/by/4.0/legalcode.txt
- CC BY-NC 2.0 원문: https://creativecommons.org/licenses/by-nc/2.0/legalcode
- MotionBERT 공식 모델 배포 저장소의 LICENSE: https://huggingface.co/walterzhu/MotionBERT/raw/main/LICENSE

라이선스 원문은 번역하거나 수정하지 않았다. 코드 라이선스 사본도 기존 실행 묶음에 있던 원문을 그대로 보존했다. 출처 고지가 원 저자·기관의 보증이나 임상 사용 승인을 의미하지 않는다.
