// 현재 앱의 시작 화면과 로컬 분석을 연결한다.
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './ui/App';
import { Splash } from './ui/components/Splash';
import './ui/ui.css';
// Splash: 앱을 처음 열 때만 잠깐 덮는 시작 화면. 뒤의 App은 그동안에도 평소처럼 준비된다.
createRoot(document.getElementById('root')!).render(<StrictMode><App /><Splash /></StrictMode>);
if (import.meta.env.PROD && 'serviceWorker' in navigator) window.addEventListener('load', () => { void navigator.serviceWorker.register('/sw.js').catch(console.error); });
