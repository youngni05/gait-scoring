import { resolve } from 'node:path';
import { defineConfig } from 'vite';

// 빌드에 두 화면을 함께 넣는다.
// - index.html: 로컬 보행 분석 화면
// - connected.html: 기존 서버 연동 화면
export default defineConfig({
  server: {
    proxy: {
      '/team-assets/': 'http://127.0.0.1:8881',
      '/v1/team/': 'http://127.0.0.1:8881',
      '/v1/history/': 'http://127.0.0.1:8881',
      '/health': 'http://127.0.0.1:8881',
    },
  },
  build: {
    rolldownOptions: {
      input: {
        index: resolve(import.meta.dirname, 'index.html'),
        connected: resolve(import.meta.dirname, 'connected.html'),
      },
    },
  },
});
