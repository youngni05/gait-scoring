import { PoseTrial } from './PoseTrial';
import { useEffect, useRef, useState } from 'react';
import { BrowserVideoSource, type BrowserVideoSession } from './pose/browser-video-source';
import type { FrameSummary } from './pose/frame-audit';
import type { ProcessingInput, VideoInfo } from './pose/ports';

export function VideoTrial({ onSelection, input }: { onSelection: () => void; input: Omit<ProcessingInput,'recordingId'> }) {
  const [recordingId,setRecordingId]=useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const mount = useRef<HTMLDivElement>(null);
  const session = useRef<BrowserVideoSession | null>(null);
  const controller = useRef<AbortController | null>(null);
  const epoch = useRef(0);
  const [info, setInfo] = useState<VideoInfo | null>(null);
  const [name, setName] = useState('선택한 영상 없음');
  const [status, setStatus] = useState('영상을 선택하세요.');
  const [phase, setPhase] = useState<'idle' | 'loading' | 'reading'>('idle');
  const [summary, setSummary] = useState<FrameSummary | null>(null);
  const [expected, setExpected] = useState<number | null>(null);
  useEffect(() => () => { epoch.current++; controller.current?.abort(); void session.current?.dispose(); }, []);
  async function select(file?: File) {
    setSelectedFile(file ?? null); setRecordingId(file?crypto.randomUUID():'');
    const id = ++epoch.current;
    controller.current?.abort(); void session.current?.dispose(); session.current = null;
    setInfo(null); setSummary(null); setExpected(null); setPhase('idle'); onSelection();
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
    setPhase('reading'); setSummary(null); setExpected(null); setStatus('파일을 직접 디코딩하고 모든 프레임 시각을 대조합니다.');
    try {
      for await (const frame of current.frames(abort.signal)) {
        try { if (id !== epoch.current) return; setSummary(current.audit.snapshot()); }
        finally { frame.release(); }
      }
      if (id === epoch.current) setStatus(current.verification?.complete ? '전체 프레임 검증 완료: 개수와 모든 원본 시각이 일치합니다.' : '전체 프레임 검증 미완료');
    } catch (error) {
      if (id === epoch.current) setStatus(abort.signal.aborted ? '프레임 읽기 취소됨. 부분 기록입니다. 다시 실행할 수 있습니다.' : `오류: ${error instanceof Error ? error.message : String(error)}`);
    } finally { if (id === epoch.current) { setSummary(current.audit.snapshot()); setExpected(current.verification?.expected.length ?? null); setPhase('idle'); } }
  }
  return <><section className="card">
    <h2>영상 선택 · 프레임 점검</h2>
    <p className="muted">원본은 이 기기에만 둡니다. 아래에서 프레임 점검과 관절 추출을 실행할 수 있습니다.</p>
    <label>로컬 영상 파일 선택<input type="file" accept="video/*" onChange={e => { const file = e.target.files?.[0]; if (file) void select(file); e.target.value = ''; }} /></label>
    <p className="file-name">{name}</p><div className="video-preview" ref={mount} />
    {info && <p className="status">표시·캔버스 기준 {info.width} × {info.height} · {info.orientation === 'landscape' ? '가로' : '세로'} · {info.durationSeconds.toFixed(3)}초<br />파일 회전 메타데이터 {info.rotationAppliedDegrees}°를 캔버스에 적용합니다.</p>}
    <small>촬영일 자동 추출은 지원하지 않습니다. 파일 수정일은 촬영일로 사용하지 않습니다. 영상을 바꾸면 촬영일 입력도 초기화됩니다.</small>
    <div className="actions"><button disabled={!info || phase !== 'idle'} onClick={() => void read()}>프레임 읽기 시작</button><button className="secondary" disabled={phase === 'idle'} onClick={() => controller.current?.abort()}>취소</button><button className="secondary" onClick={() => void select()}>선택 해제</button></div>
    <p role="status" className="status">{status}</p>
    {summary && <div className="frame-report"><strong>추출 {summary.count} / 원본 {expected ?? '확인 중'}프레임</strong><p>첫 시각 {summary.first?.toFixed(6) ?? '—'}초 / 끝 시각 {summary.last?.toFixed(6) ?? '—'}초<br />최대 시각 간격 {summary.maxGap.toFixed(6)}초</p><details><summary>최근 원본 프레임 시각 (초, 최대 10개)</summary><pre>{summary.recent.join('\n')}</pre></details></div>}
    <small>MP4/H.264·HEVC(H.265) 파일을 재생 속도와 무관하게 순차 디코딩합니다. 원본 PTS를 보존하며 개수·시각 불일치는 오류입니다. 취소 결과는 부분 기록입니다.</small>
  </section><PoseTrial file={selectedFile} input={{...input,recordingId}}/></>;
}
