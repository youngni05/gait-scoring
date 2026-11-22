import { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Capacitor } from '@capacitor/core';
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import { checkProcessingInput } from '../../mobile/src/pose/processing-input';
import { resolveRecordedAt } from '../../mobile/src/pose/recorded-at';
import './style.css';
import { VideoTrial } from './VideoTrial';

function App() {
  const [subjectId, setSubject] = useState('DEV-SUBJECT-001');
  const [height, setHeight] = useState('170');
  const [day, setDate] = useState('');
  const [metadata, setMetadata] = useState<string | null>(null);
  const {recordedAt, recordedAtSource: source} = resolveRecordedAt(day, metadata);
  const [message, setMessage] = useState('입력값을 확인하면 이곳에 결과가 표시됩니다.');
  const [engine, setEngine] = useState('미실행');
  const [busy, setBusy] = useState(false);
  async function probe() {
    setBusy(true); setEngine('로컬 모델과 WASM을 불러오는 중…');
    let landmarker: PoseLandmarker | undefined;
    try {
      const files = await FilesetResolver.forVisionTasks('/vendor/mediapipe');
      landmarker = await PoseLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: '/models/pose_landmarker_full.task', delegate: 'CPU' },
        runningMode: 'IMAGE', numPoses: 1,
      });
      const canvas = document.createElement('canvas'); canvas.width = 256; canvas.height = 256;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Canvas 2D를 사용할 수 없습니다.');
      ctx.fillStyle = '#808080'; ctx.fillRect(0, 0, 256, 256);
      const result = landmarker.detect(canvas);
      setEngine(`초기화·합성 이미지 실행 성공 · 검출 ${result.landmarks.length}명. 실제 영상 추출은 미검증입니다.`);
    } catch (error) { setEngine(`실패: ${error instanceof Error ? error.message : String(error)}`); }
    finally { landmarker?.close(); setBusy(false); }
  }
  return <main>
    <header><span className="eyebrow">GAIT LAB / STAGE 06</span><span className="badge">개발 전용</span></header>
    <section className="intro"><p className="step">01 — 실행 환경 확인</p><h1>보행 영상<br />처리 시험실</h1><p>입력과 실행 환경을 먼저 확인합니다.<br />실제 대상자 등록이나 분석은 수행하지 않습니다.</p></section>
    <div className="environment"><span>현재 실행 환경</span><strong>{Capacitor.isNativePlatform() ? 'Android 앱' : 'PC · 웹 브라우저'}</strong><small>개발 빌드 0.6.0 · 서버 연결 없음</small></div>
    <VideoTrial input={{subjectId,heightCm:Number(height),recordedAt,recordedAtSource:source}} onMetadata={setMetadata} onSelection={() => { setDate(''); setMetadata(null); setMessage('영상 변경됨 · 촬영일 메타데이터를 확인합니다.'); }} />
    <section className="card"><h2>시험 입력</h2><p className="muted">실제 개인정보 대신 시험용 값을 사용하세요.</p>
      <form noValidate onSubmit={event => { event.preventDefault(); const errors = checkProcessingInput({ subjectId, heightCm: Number(height), recordedAt, recordedAtSource: source }); setMessage(errors.length ? errors.join(' ') : '입력 형식 확인 완료. 서버 발급 ID 확인·저장·전송은 수행하지 않았습니다.'); }}>
        <label>시험용 대상자 ID<input value={subjectId} onChange={e => { setSubject(e.target.value); setMessage('입력 변경됨 · 다시 확인하세요.'); }} /></label>
        <label>촬영 당시 키 <span className="unit">cm</span><input type="number" min="100" max="220" step="0.1" value={height} onChange={e => { setHeight(e.target.value); setMessage('입력 변경됨 · 다시 확인하세요.'); }} /></label>
        <label>촬영일 (년·월·일, 선택 입력)<input type="text" placeholder="YYYY-MM-DD (예: 2026-11-01)" aria-label="촬영일" value={day} onChange={e => { setDate(e.target.value); setMessage('입력 변경됨 · 다시 확인하세요.'); }} /></label>
        <small>직접 입력한 날짜를 우선 사용합니다. 날짜만 입력하므로 한국 시간 00:00으로 저장합니다. 비워 두면 영상 메타데이터를 사용합니다.</small>
        <p role="status">{recordedAt ? `적용 촬영일: ${recordedAt.slice(0,10).replaceAll('-', '.')} ${recordedAt.slice(11,19)} (한국 시간) · 출처: ${source === 'user_entered' ? '사용자 입력' : '영상 메타데이터'}` : '촬영일 메타데이터가 없거나 날짜가 유효하지 않습니다. 년·월·일을 입력하세요.'}</p>
        <small>영상 메타데이터는 편집·전달 과정에서 바뀔 수 있습니다. 실제 촬영일과 다르면 직접 입력하세요.</small>
        <button type="submit">입력 형식 확인 <span>→</span></button>
      </form><p className="status" role="status">{message}</p>
    </section>
    <section className="card"><h2>기기 기능 점검</h2><p className="muted">이전 단계의 합성 이미지 엔진 점검입니다.</p>
      <button className="secondary" disabled={busy} onClick={() => void probe()}>{busy ? '엔진 점검 중…' : 'MediaPipe 엔진 점검'}</button><p className="status" role="status">{engine}</p><small>CPU · full 모델 · tasks-vision 1.2.3<br />빈 합성 이미지로 초기화만 확인합니다. 최종 모델 선정 전입니다.</small>
    </section>
    <footer>실제 서버 연결 대기 · 서 있기 수동 확인과 갤럭시 검증 보류<br />입력값은 저장되지 않으며 앱을 다시 열면 초기화됩니다.</footer>
  </main>;
}
createRoot(document.getElementById('root')!).render(<App />);
