// API-connected design UI. The legacy integration screen remains at connected.html.
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './ui/App';
import './ui/ui.css';
createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>);
