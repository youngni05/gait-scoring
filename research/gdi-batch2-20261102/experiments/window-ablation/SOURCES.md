# 출처와 적용 라이선스

## 학습 자료

- 저자: Łukasz Kidziński, Bryan Yang, Jennifer L. Hicks, Apoorva Rajagopal, Scott L. Delp, Michael H. Schwartz.
- 논문: Deep neural networks enable quantitative movement analysis using single-camera videos, Nature Communications 11, 4054 (2020). https://doi.org/10.1038/s41467-020-17807-z
- 처리된 관절 궤적·임상값 배포: https://doi.org/10.18735/j0rz-0k12 ; https://simtk.org/projects/video-gaitlab
- 공식 코드·이용 조건: https://github.com/stanfordnmbl/mobile-gaitlab ; https://raw.githubusercontent.com/stanfordnmbl/mobile-gaitlab/master/README.md
- 원자료의 처리 궤적: CC BY-NC 2.0. https://creativecommons.org/licenses/by-nc/2.0/
- 참고 코드: Apache 2.0. 원자료의 조건과 구분한다.
- 본 프로젝트는 공개 처리 좌표를 변환하고 일부 기록을 선별해 좌우 GDI 예측을 학습했다. 원자료 그대로의 재배포가 아니며 원본 영상·개인별 좌표·정답표는 포함하지 않았다.

프로젝트 연구기록: https://github.com/youngni05/gait-scoring/tree/main/research

## 고정 MotionBERT 특징을 사용하는 조건의 출처

저자: Wentao Zhu, Xiaoxuan Ma, Zhaoyang Liu, Libin Liu, Wayne Wu, Yizhou Wang. MotionBERT: A Unified Perspective on Learning Human Motion Representations, ICCV 2023. https://arxiv.org/abs/2210.06551 ; 코드 https://github.com/Walter0807/MotionBERT ; 공식 모델 배포 https://huggingface.co/walterzhu/MotionBERT . 사용한 Lite 초기 가중치 SHA-256은 6a6ad0055c7ad50da083af0549a24c52ec1c21f89e440912645054d74be0a461이다. 원 코드의 Apache 2.0 고지를 보존한다. 이 고지는 고정 특징을 사용한 조건에 해당하며, 이번 압축에 사전학습 본체 가중치나 실행 라이브러리를 다시 넣었다는 뜻이 아니다.

## 이번 실험

구간 위치·정규화 Ridge 비교
원 실험 날짜: 2026-10-27
프로젝트의 학습·검산 코드와 결과는 공개 저장소의 해당 실험 폴더를 따른다. 이번 묶음에는 MotionBERT 사전학습 본체·원본 데이터·개인별 정답·사람 명단을 포함하지 않았다. 고정 특징을 사용하는 조건은 원 실험과 동일하게 준비한 특징이 필요하다.
