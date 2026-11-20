/**
 * 화면이 쓰는 데이터 기능의 목록(연결 지점).
 *
 * 화면은 이 인터페이스만 알고, 실제 값은 data/index.ts가 고른 구현체에서 받는다.
 * 지금은 가상 데이터 구현(mockSource.ts)만 있다. 서버 연결 담당자는 같은 모양으로
 * ApiClient(src/api/client.ts)를 감싼 구현을 만들어 data/index.ts의 한 줄만 바꾸면 된다.
 *
 * 서버 응답과의 대응(참고용, 확정 아님):
 *   Subject.sex '여'/'남'          ↔ Participant.gender 'F'/'M'
 *   Subject.birth 'YYYY-MM-DD'     ↔ Participant.birth_date
 *   Subject.heightCm               ↔ Participant.height_cm
 *   ResultRecord.ageAtCapture      ↔ Measurement.age
 *   ResultRecord.gdi / cadence     ↔ analysis_result.gdi_score / cadence_spm
 *   ResultRecord.symmetry·curves   → 현재 서버 응답에 없음(추후 확정)
 */
import type { ResultRecord, Subject } from './mock';

export type SubjectInput = Omit<Subject, 'id'>;

export interface SubjectSummary extends Subject {
  /** 가장 최근 측정 시각(ISO). 기록이 없으면 null */
  latestMeasuredAt: string | null;
  resultCount: number;
}

/** 로그인 실패(아이디·비밀번호 불일치). 화면은 이 오류일 때 01-2 안내를 띄운다. */
export class LoginFailedError extends Error {}

/** 영상 분석 실패. code는 화면 하단에 ‘오류 코드’로 보여준다. */
export class AnalysisFailedError extends Error {
  constructor(public code: string, public reason: string) {
    super(reason);
  }
}

export interface GaitDataSource {
  login(username: string, password: string): Promise<void>;
  logout(): void;

  listSubjects(signal?: AbortSignal): Promise<SubjectSummary[]>;
  getSubject(id: string, signal?: AbortSignal): Promise<Subject | null>;
  createSubject(input: SubjectInput): Promise<Subject>;
  updateSubject(id: string, input: SubjectInput): Promise<Subject>;

  /** 최신순 */
  listResults(subjectId: string, signal?: AbortSignal): Promise<ResultRecord[]>;
  getResult(subjectId: string, resultId: string, signal?: AbortSignal): Promise<ResultRecord | null>;

  /**
   * 영상을 분석해 결과를 만든다. 진행률(0~100)을 onProgress로 알린다.
   * video가 null이면 예시 영상(데모)으로 진행한다.
   */
  analyzeVideo(
    subjectId: string,
    video: File | null,
    onProgress: (percent: number) => void,
    signal?: AbortSignal,
  ): Promise<ResultRecord>;

  /** 결과지를 파일로 내려받는다. 파일 형식은 추후 확정. */
  downloadResult(resultId: string): Promise<void>;
}
