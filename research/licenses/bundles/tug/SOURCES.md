# 출처와 적용 라이선스

## 학습 자료

- 자료명: Neuromuscular OpenCap Dataset, Version 2.0 (2026).
- 저자: Sydney Covitz, Parker Ruth, Shelby Vogt-Domke, Sarah Ismail, Lin Karman, Julie Muccini, Carmichael Ong, Shannon Li, Audrey Chun, Melina Rogers, Tian Tan, Jennifer Hicks, Scott Uhlrich, John Day, Constance de Monts, Tina Duong, Scott Delp.
- 공식 자료: https://zenodo.org/records/22309771 ; https://doi.org/10.5281/zenodo.22309771
- 라이선스: CC BY 4.0. https://creativecommons.org/licenses/by/4.0/ ; 공식 API https://zenodo.org/api/records/22309771 의 metadata.license.id=cc-by-4.0 확인.
- 관련 논문: Deep learning models detect neuromuscular disease and quantify functional impairment from video-derived biomechanics data. https://doi.org/10.64898/2026.09.30.754736
- 공개본의 298명에 대해 3초 보행 시계열과 같은 방문의 실제 TUG 시간을 연결했다. 관절 각도·골반 속도를 선택·보간하고 학습용 입력으로 변환했다. 원본 영상·개인별 시계열·임상 정답표는 포함하지 않았다.
- Ridge·RF·작은 CNN·TCN은 본 프로젝트에서 학습했다. 원 저자의 질환 탐지 모델 가중치를 재배포한 묶음이 아니다. OpenCap으로 미리 복원한 시계열을 사용했으며 OpenCap 실행 프로그램은 포함하지 않는다.

프로젝트 연구기록: https://github.com/youngni05/gait-scoring/tree/main/research
