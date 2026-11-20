import { resolve } from 'node:path';
import { defineConfig } from 'vite';

// 빌드(안드로이드 앱 포함)에 두 화면을 함께 넣는다.
// - index.html: 새 디자인 화면(가상 데이터)
// - connected.html: 기존 서버 연동 화면
export default defineConfig({
  build: {
    rolldownOptions: {
      input: {
        index: resolve(import.meta.dirname, 'index.html'),
        connected: resolve(import.meta.dirname, 'connected.html'),
      },
    },
  },
});
