/**
 * 화면이 사용할 데이터 구현을 고르는 곳.
 * 기본값은 팀 API이고 demo=ui에서는 예시 데이터를 사용한다.
 */
import { mockSource } from './mockSource';
import type { GaitDataSource } from './source';

import { liveSource } from './liveSource';
export const isDemo = new URLSearchParams(location.search).get('demo') === 'ui';
export const dataSource: GaitDataSource = isDemo ? mockSource : liveSource;

export * from './source';
export * from './mock';
