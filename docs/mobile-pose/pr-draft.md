# PR 초안

제목: 로컬 보행 영상 전체 프레임 처리와 coords-1 JSON 개발용 파이프라인 추가

MP4/H.264·HEVC 원본의 모든 프레임을 순차 디코딩하고 원본 PTS와 개수를 검증한다. MediaPipe full/CPU로33관절을 추출하며 미검출 프레임은null로 보존한다. 실험적 서 있기 검출, 오버레이 확인, 촬영일 수동 입력/메타데이터 fallback, coords-1 JSON과 mock 재시도를 개발 화면에서 제공한다.

검증: 자동115개, mobile 타입 검사/빌드, Android debug 빌드 통과. B-758 실제영상 전체PTS/직렬화 대조 및 취소 후758프레임 재시작 통과. 독립 PyAV 시각 대조. 실제 영상/좌표는 커밋하지 않는다. 8단계 별도 깨끗한 설치 검증 결과는 stage-8-verification.txt에 기록한다.

제한: 실제 서버·인증·GDI/분석 점수 없음. 서 있기 수동 정답과 Galaxy 실기기 검증은 보류. CSP는 SDK 외부 통신 시도를 차단하며 실제 서버 연결 때 명시적으로 검토해야 한다. 현재 UI/production 산출물도 개발 시험용이다.

리뷰 순서: handoff.md → coords-1 schema/validate → pose ports와 adapters → 취소/정리 → 날짜 우선순위 → mock/직렬화 → 검증 기록. HTTP/인증/멱등성/응답 계약은 서버 담당자와 추가 합의가 필요하다.

기본 브랜치 병합이나 팀원 메시지 전송은 수행하지 않는다.
