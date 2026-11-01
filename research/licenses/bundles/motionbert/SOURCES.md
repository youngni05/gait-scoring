# 출처와 적용 라이선스

## 학습 자료

- 저자: Łukasz Kidziński, Bryan Yang, Jennifer L. Hicks, Apoorva Rajagopal, Scott L. Delp, Michael H. Schwartz.
- 논문: Deep neural networks enable quantitative movement analysis using single-camera videos, Nature Communications 11, 4054 (2020). https://doi.org/10.1038/s41467-020-17807-z
- 처리된 관절 궤적·임상값 배포: https://doi.org/10.18735/j0rz-0k12 ; https://simtk.org/projects/video-gaitlab
- 공식 코드·이용 조건: https://github.com/stanfordnmbl/mobile-gaitlab ; https://raw.githubusercontent.com/stanfordnmbl/mobile-gaitlab/master/README.md
- 원자료의 처리 궤적: CC BY-NC 2.0. https://creativecommons.org/licenses/by-nc/2.0/
- 참고 코드: Apache 2.0. 원자료의 조건과 구분한다.
- 본 프로젝트는 공개 처리 좌표를 변환하고 일부 기록을 선별해 좌우 GDI 예측을 학습했다. 원자료 그대로의 재배포가 아니며 원본 영상·개인별 좌표·정답표는 포함하지 않았다.

## 사전학습 모델·코드

- 저자: Wentao Zhu, Xiaoxuan Ma, Zhaoyang Liu, Libin Liu, Wayne Wu, Yizhou Wang.
- 논문: MotionBERT: A Unified Perspective on Learning Human Motion Representations, ICCV 2023. https://arxiv.org/abs/2210.06551
- 코드: https://github.com/Walter0807/MotionBERT
- 공식 가중치 배포 저장소: https://huggingface.co/walterzhu/MotionBERT
- 사용한 MotionBERT-Lite 초기 가중치: https://huggingface.co/walterzhu/MotionBERT/resolve/main/checkpoint/pretrain/MB_lite/latest_epoch.bin
- 초기 가중치 SHA-256: 6a6ad0055c7ad50da083af0549a24c52ec1c21f89e440912645054d74be0a461
- 코드와 공식 가중치 배포 저장소에 제공된 Apache 2.0 LICENSE 사본을 각각 보존했다. 권리자 표기는 사본에 기재된 원문을 수정하지 않았다.
- 이번 모델은 원 사전학습 가중치 그대로가 아니라 마지막 Transformer 단계·출력부를 GDI 자료에 맞춰 부분학습한 결과다. 수정된 텐서가 포함됨을 명시한다.

프로젝트 연구기록: https://github.com/youngni05/gait-scoring/tree/main/research
