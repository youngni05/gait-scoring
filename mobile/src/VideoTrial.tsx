import { useEffect, useRef, useState } from 'react';
import { BrowserVideoSource, type BrowserVideoSession } from './pose/browser-video-source';
import type { FrameSummary } from './pose/frame-audit';
import type { VideoInfo } from './pose/ports';

export function VideoTrial({ onSelection }: { onSelection: () => void }) {
  const mount = useRef<HTMLDivElement>(null);
  const session = useRef<BrowserVideoSession | null>(null);
  const controller = useRef<AbortController | null>(null);
  const epoch = useRef(0);
  const [info, setInfo] = useState<VideoInfo | null>(null);
  const [name, setName] = useState('선택한 영상 없음');
  const [status, setStatus] = useState('영상을 선택하세요.');
  const [phase, setPhase] = useState<'idle' | 'loading' | 'reading'>('idle');
  const [summary, setSummary] = useState<FrameSummary | null>(null);
  const [quality, setQuality] = useState<{ dropped: number; total: number } | null>(null);
  useEffect(() => () => { epoch.current++; controller.current?.abort(); void session.current?.dispose(); }, []);
  async function select(file?: File) {
    const id = ++epoch.current;
    controller.current?.abort(); void session.current?.dispose(); session.current = null;
    setInfo(null); setSummary(null); setQuality(null); setPhase('idle'); onSelection();
    if (!file) { setName('선택한 영상 없음'); setStatus('영상 선택을 해제했습니다.'); return; }
    setName(`${file.name} · ${(file.size / 1048576).toFixed(1)} MB`);
    const abort = new AbortController(); controller.current = abort; setPhase('loading'); setStatus('영상 정보 읽는 중…');
    try {
      const opened = await new BrowserVideoSource(mount.current!).open(file, abort.signal);
      if (id !== epoch.current) { await opened.dispose(); return; }
      session.current = opened; setInfo(opened.info); setStatus('미리보기 준비 완료. 촬영일은 아래에서 직접 확인·입력하세요.');
    } catch (error) { if (id === epoch.current) setStatus(abort.signal.aborted ? '선택 취소됨' : `오류: ${error instanceof Error ? error.message : String(error)}`); }
    finally { if (id === epoch.current) setPhase('idle'); }
  }
  async function read() {
    const current = session.current; if (!current || phase !== 'idle') return;
    const id = ++epoch.current; const abort = new AbortController(); controller.current = abort;
    setPhase('reading'); setSummary(null); setQuality(null); setStatus('처음부터 재생하며 프레임을 확인합니다. 이 화면을 유지하세요.');
    try {
      for await (const frame of current.frames(abort.signal)) {
        try { if (id !== epoch.current) return; setSummary(current.audit.snapshot()); }
        finally { frame.release(); }
      }
      if (id === epoch.current) setStatus(current.audit.snapshot().count ? '재생 관측 완료. 전체 원본 프레임 추출을 보장하지 않습니다.' : '관측된 프레임이 없습니다. 브라우저/코덱을 확인하세요.');
    } catch (error) {
      if (id === epoch.current) setStatus(abort.signal.aborted ? '프레임 읽기 취소됨. 부분 기록입니다. 다시 실행할 수 있습니다.' : `오류: ${error instanceof Error ? error.message : String(error)}`);
    } finally { if (id === epoch.current) { setSummary(current.audit.snapshot()); setQuality(current.playbackQuality()); setPhase('idle'); } }
  }
  return <section className="card">
    <h2>영상 선택 · 프레임 점검</h2>
    <p className="muted">원본은 이 기기에만 둡니다. 이번 단계는 관절 추출 전 영상 점검입니다.</p>
    <label>로컬 영상 파일 선택<input type="file" accept="video/*" onChange={e => { const file = e.target.files?.[0]; if (file) void select(file); e.target.value = ''; }} /></label>
    <p className="file-name">{name}</p><div className="video-preview" ref={mount} />
    {info && <p className="status">표시·캔버스 기준 {info.width} × {info.height} · {info.orientation === 'landscape' ? '가로' : '세로'} · {info.durationSeconds.toFixed(3)}초<br />회전은 브라우저 해석을 사용합니다. 원본 회전 각도는 미확인입니다.</p>}
    <small>촬영일 자동 추출은 지원하지 않습니다. 파일 수정일은 촬영일로 사용하지 않습니다. 영상을 바꾸면 촬영일 입력도 초기화됩니다.</small>
    <div className="actions"><button disabled={!info || phase !== 'idle'} onClick={() => void read()}>프레임 읽기 시작</button><button className="secondary" disabled={phase === 'idle'} onClick={() => controller.current?.abort()}>취소</button><button className="secondary" onClick={() => void select()}>선택 해제</button></div>
    <p role="status" className="status">{status}</p>
    {summary && <div className="frame-report"><strong>관측 {summary.count}프레임</strong><p>첫 시각 {summary.first?.toFixed(6) ?? '—'}초 / 끝 시각 {summary.last?.toFixed(6) ?? '—'}초<br />최대 시각 간격 {summary.maxGap.toFixed(6)}초<br />콜백 사이 표시 번호 누락 {summary.missedCallbacks}개</p><small>표시 번호 누락은 관측 간 놓친 표시 프레임 수입니다. 디코더가 버린 원본 프레임 수나 전체 누락률이 아닙니다.</small><details><summary>최근 실제 프레임 시각 (초, 최대 10개)</summary><pre>{summary.recent.join('\n')}</pre></details></div>}
    <small>실제 시각 API(mediaTime)를 사용합니다. 고정 FPS로 시각을 만들지 않습니다. 느린 처리·화면 전환 시 누락될 수 있어 정확한 오프라인 디코딩은 별도 검토가 필요합니다.</small>
    {summary && phase === 'idle' && <p className="status">{quality ? `브라우저 재생 통계: 처리 대상 ${quality.total}개 / 드롭 ${quality.dropped}개` : '이 환경에서는 브라우저 드롭 통계를 확인하지 못했습니다.'}<br />이 통계도 파일 전체 원본 프레임 수를 보증하지 않습니다.</p>}
  </section>;
}
