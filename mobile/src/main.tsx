// 기본 화면에는 로컬 분석을 연결한다. 기존 API 연동 화면은 connected.html에서 열 수 있다.
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './ui/App';
import './ui/ui.css';
createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>);
if (import.meta.env.PROD && 'serviceWorker' in navigator) window.addEventListener('load', () => { void navigator.serviceWorker.register('/sw.js').catch(console.error); });
