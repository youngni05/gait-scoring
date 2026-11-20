import { ApiDataSource } from './apiSource';
import type { GaitDataSource } from './source';

export const apiSource = new ApiDataSource(import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000');
export const dataSource: GaitDataSource = apiSource;

export * from './source';
export * from './mock';
