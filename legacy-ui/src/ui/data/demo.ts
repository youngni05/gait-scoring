/**
 * 데모 상태 설정. 서버 없이 실패·빈 목록 같은 상태 화면을 보기 위한 검토용 값이다.
 * 가상 데이터 구현(mockSource)만 이 값을 읽는다. 실제 서버 구현에서는 쓰지 않는다.
 */
import { useSyncExternalStore } from 'react';

export interface DemoOptions {
  login: 'success' | 'fail';
  subjectList: 'normal' | 'loading' | 'empty';
  analysis: 'success' | 'partial' | 'fail';
  download: 'success' | 'fail';
}

let state: DemoOptions = { login: 'success', subjectList: 'normal', analysis: 'success', download: 'success' };
const listeners = new Set<() => void>();

export const getDemo = () => state;

export function setDemo(patch: Partial<DemoOptions>) {
  state = { ...state, ...patch };
  listeners.forEach((l) => l());
}

function subscribe(l: () => void) {
  listeners.add(l);
  return () => listeners.delete(l);
}

export const useDemo = () => useSyncExternalStore(subscribe, getDemo);
