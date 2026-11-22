/**
 * 화면이 쓰는 데이터 기능의 목록(연결 지점).
 *
 * 화면은 이 인터페이스만 알고, 실제 값은 data/index.ts가 고른 구현체에서 받는다.
 * 기본 앱은 팀 API를 사용하며, UI 데모는 메모리의 예시 데이터를 사용한다.
 */
import type { ResultRecord, Subject } from './mock';

export type SubjectInput = Omit<Subject, 'id'>;
export type CaptureDaySource = 'metadata' | 'user_entered' | 'unknown';

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
  deleteResult(subjectId: string, resultId: string): Promise<void>;

  /**
   * 영상을 분석해 결과를 만든다. 진행률(0~100)을 onProgress로 알린다.
   * video가 null이면 예시 영상(데모)으로 진행한다.
   */
  analyzeVideo(
    subjectId: string,
    video: File | null,
    onProgress: (percent: number, detail?: string) => void,
    signal?: AbortSignal,
    capture?: {day:string; daySource?:CaptureDaySource; mode:"guided"|"fixed"; height:number|null; extractor?:"mediapipe"|"openpose"},
  ): Promise<ResultRecord>;

  /** 결과지를 파일로 내려받는다. 파일 형식은 추후 확정. */
  downloadResult(resultId: string): Promise<void>;
}
