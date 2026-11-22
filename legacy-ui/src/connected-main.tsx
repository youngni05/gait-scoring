// 구버전 FastAPI 연동 화면의 시작점.
import { createRoot } from 'react-dom/client';
import { ConnectedApp } from './ConnectedApp';
import './style.css';
createRoot(document.getElementById('root')!).render(<ConnectedApp />);
