/**
 * 화면이 사용할 데이터 구현을 고르는 곳.
 * 서버 연결 시 아래 한 줄을 ApiClient 기반 구현으로 바꾼다(source.ts의 GaitDataSource 모양을 따른다).
 */
import { mockSource } from './mockSource';
import type { GaitDataSource } from './source';

export const dataSource: GaitDataSource = mockSource;

export * from './source';
export * from './mock';
