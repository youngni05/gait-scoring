// 서버 연동 화면(기존 첫 화면). 새 디자인 화면이 index.html을 쓰게 되어 이 주소로 옮겼다.
import { createRoot } from 'react-dom/client';
import { ConnectedApp } from './ConnectedApp';
import './style.css';
createRoot(document.getElementById('root')!).render(<ConnectedApp />);
