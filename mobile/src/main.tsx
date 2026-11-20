// 앱 첫 화면: UX Pilot 디자인 기반 화면(가상 데이터). 서버 연동 화면은 connected.html에 있다.
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './ui/App';
import './ui/ui.css';
createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>);
