/**
 * 06 영상 선택 · 06-1 영상 미리보기 · 06-2 분석 진행 중 · 06-3 분석 실패.
 * 한 주소 안에서 단계(phase)만 바꾼다. 분석 중에 뒤로 가기로 빠져나가 결과를 잃는 일을 줄이기 위해서다.
 *
 * 실제 연결 상태: 휴대폰 영상 선택과 미리보기 재생은 브라우저 기능으로 실제 동작한다.
 * 분석은 dataSource.analyzeVideo가 맡는다. 지금은 가상 데이터 구현이 진행률과 결과를 흉내 낸다.
 * (실제 연결 시 기존 관절 추출·측정 저장 코드(pose/, MeasurementUpload)를 이 함수 뒤에서 호출하게 된다.)
 */
import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { PageError, PageLoading, Steps, TopBar, type StepState } from '../components/layout';
import {
  CheckCircleIcon,
  CheckIcon,
  CircleIcon,
  FileIcon,
  FilmIcon,
  FolderOpenIcon,
  InfoIcon,
  PauseIcon,
  PlayIcon,
  RefreshIcon,
  RotateIcon,
  SpinnerIcon,
  TriangleAlertIcon,
} from '../components/icons';
import { AnalysisFailedError, ageOn, isDemo, dataSource, fmtDate } from '../data';
import { openPoseEnabled } from '../data/analysis';
import { readCaptureDay } from '../../pose/browser-video-source';
import { useNavigate, useParams } from '../router';
import { useAsync } from '../session';
import { NotFound } from './NotFound';

type Phase = 'pick' | 'preview' | 'running' | 'failed' | 'done';

interface Picked {
  /** 실제로 고른 파일. 예시 영상이면 null */
  file: File | null;
  name: string;
  /** 실제 영상이면 재생용 주소, 예시 영상이면 null */
  url: string | null;
  durationSec: number | null;
  fileDate: Date;
  metadataDay: string | null | undefined; // undefined while reading
}

const STAGES = ['영상 확인', '보행 구간 찾기', '관절 움직임 분석', '결과지 만들기'];
/** 남은 시간 안내용 대략적인 전체 분석 시간(초). 실제 연결 시 측정값으로 바꾼다. */
const EXPECTED_SEC = 6;

const fmtClock = (sec: number) => `${Math.floor(sec / 60)}:${String(Math.floor(sec % 60)).padStart(2, '0')}`;
const fmtLength = (sec: number | null) => (sec === null ? '확인 중' : `${Math.floor(sec / 60)}분 ${Math.round(sec % 60)}초`);

export function AnalyzePage() {
  const { subjectId = '' } = useParams();
  const navigate = useNavigate();
  const loaded = useAsync((signal) => dataSource.getSubject(subjectId, signal), [subjectId]);
  const subject = loaded.data;

  const [editedDay, setEditedDay] = useState<string | null>(null);
  const [mode,setMode]=useState<'fixed'|'guided'>('guided');
  const [extractor,setExtractor]=useState<'mediapipe'|'openpose'>('mediapipe');
  const [editedHeight, setEditedHeight] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>('pick');
  const [picked, setPicked] = useState<Picked | null>(null);
  const [progressText,setProgressText]=useState('');
  const [pct, setPct] = useState(0);
  const [resultId, setResultId] = useState<string | null>(null);
  const [failure, setFailure] = useState<{ code: string; reason: string } | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const day = editedDay ?? picked?.metadataDay ?? '';
  const daySource = !day ? 'unknown' : editedDay === null ? 'metadata' : 'user_entered';
  const registeredHeight = subject?.heightCm;
  const usableHeight = registeredHeight != null && registeredHeight >= 100 && registeredHeight <= 220;
  const height = editedHeight ?? (usableHeight ? String(registeredHeight) : '');
  const invalidHeight = Boolean(height && (Number(height) < 100 || Number(height) > 220));
  const ageAtCapture = subject?.birth && day >= subject.birth ? ageOn(subject.birth, new Date(day + 'T12:00:00')) : null;

  // 선택한 영상의 임시 주소는 화면을 떠나거나 영상을 바꿀 때 해제한다.
  const pickedUrl = picked?.url ?? null;
  useEffect(() => () => { if (pickedUrl) URL.revokeObjectURL(pickedUrl); }, [pickedUrl]);

  // 분석 진행. 화면을 떠나면 진행 중인 분석을 취소한다.
  useEffect(() => {
    if (phase !== 'running') return;
    const abort = new AbortController();
    dataSource
      .analyzeVideo(subjectId, picked?.file ?? null, (p,detail) => {if(!abort.signal.aborted){setPct(p);setProgressText(detail??'');}}, abort.signal, {day,daySource,mode,extractor,height:height ? Number(height) : null})
      .then((rec) => {
        if (abort.signal.aborted) return;
        setResultId(rec.id);
        setPhase('done');
      })
      .catch((e) => {
        if (abort.signal.aborted) return;
        setFailure(
          e instanceof AnalysisFailedError
            ? { code: e.code, reason: e.reason }
            : { code: 'GA-000', reason: e instanceof Error ? e.message : '분석하지 못했습니다' },
        );
        setPhase('failed');
      });
    return () => abort.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  const backToDetail = () => navigate(`/subjects/${subjectId}`);
  if (loaded.loading) return <PageLoading title="새 보행 영상 분석" nav="back" onNav={backToDetail} />;
  if (loaded.error) return <PageError title="새 보행 영상 분석" onNav={backToDetail} onRetry={loaded.reload} />;
  if (!subject) return <NotFound />;

  const openPicker = () => fileInput.current?.click();

  const onFile = (file: File | undefined) => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setEditedDay(null);
    setEditedHeight(null);
    setPicked({ file, name: file.name, url, durationSec: null, fileDate: new Date(file.lastModified), metadataDay: isDemo ? null : undefined });
    setPhase('preview');
    if (!isDemo) {
      void readCaptureDay(file).catch(() => null).then(metadataDay => {
        setPicked(current => current?.url === url ? { ...current, metadataDay } : current);
      });
    }
  };

  const useSample = () => {
    setPicked({ file: null, name: 'walk_sample.mp4', url: null, durationSec: 24, fileDate: new Date(), metadataDay: null });
    setPhase('preview');
  };

  const start = () => {
    setPct(0);setProgressText('');
    setFailure(null);
    setPhase('running');
  };

  const stepStates: [StepState, StepState, StepState] =
    phase === 'pick' || phase === 'preview'
      ? ['now', 'todo', 'todo']
      : phase === 'running'
        ? ['done', 'now', 'todo']
        : phase === 'failed'
          ? ['done', 'fail', 'todo']
          : ['done', 'done', 'now'];

  const title = phase === 'pick' || phase === 'preview' ? '새 보행 영상 분석' : '보행 영상 분석';
  // 분석 중에는 뒤로 가기를 숨긴다(디자인 06-2).
  const nav = phase === 'running' ? undefined : 'back';
  const onBack = () => {
    if (phase === 'preview') return setPhase('pick');
    backToDetail();
  };

  const stageIndex = isDemo ? Math.min(3, Math.floor(pct / 25)) : pct === 0 ? 0 : pct < 90 ? 1 : pct < 100 ? 2 : 3;
  const remainSec = Math.max(0, Math.round(((100 - pct) / 100) * EXPECTED_SEC));

  return (
    <div className={`screen ${phase === 'failed' ? 'has-cta-tall' : 'has-cta'}`} style={phase === 'pick' ? { paddingBottom: 160 } : undefined}>
      <TopBar title={title} nav={nav} onNav={onBack} subject={subject} />
      <Steps states={stepStates} />

      <input
        ref={fileInput}
        type="file"
        accept="video/*"
        hidden
        onChange={(e) => {
          onFile(e.target.files?.[0]);
          e.target.value = '';
        }}
      />

      {/* 06 영상 선택 */}
      {phase === 'pick' && (
        <section style={{ margin: '24px 20px 0' }}>
          <h2 style={{ fontSize: 20, fontWeight: 800 }}>분석할 보행 영상을 선택해 주세요</h2>
          <p style={{ marginTop: 6, fontSize: 15 }} className="muted">휴대폰에 저장된 영상 중 1개를 선택합니다.</p>
          <button className="pick-area" onClick={openPicker}>
            <span style={{ width: 80, height: 80, borderRadius: 999, background: '#fff', color: 'var(--primary)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <FilmIcon size={34} />
            </span>
            <span style={{ marginTop: 20, fontSize: 19, fontWeight: 800 }}>휴대폰에서 영상 선택</span>
            <span style={{ marginTop: 6, fontSize: 15 }} className="muted">갤러리 또는 파일에서 불러옵니다</span>
            <span
              style={{ marginTop: 24, height: 48, padding: '0 24px', borderRadius: 'var(--r-lg)', background: 'var(--primary)', color: '#fff', fontSize: 16, fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: 8 }}
            >
              <FolderOpenIcon size={18} /> 영상 선택
            </span>
          </button>
          {isDemo && <button className="demo-link" onClick={useSample}>
            <span className="pill pill-draft">데모</span> 휴대폰 영상 없이 예시 영상으로 진행
          </button>}
        </section>
      )}

      {/* 06-1 영상 미리보기 */}
      {phase === 'preview' && picked && (
        <section style={{ margin: '20px 20px 0' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
            <h2 style={{ fontSize: 18, fontWeight: 800 }}>선택한 영상</h2>
            <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--ok)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              <CheckCircleIcon size={16} /> 선택 완료
            </span>
          </div>
          {!isDemo && <div className="card" style={{display:'grid',gap:8,padding:16,marginBottom:12}}>
            <p>촬영일 · {day ? day.replaceAll('-', '.') : picked.metadataDay === undefined ? '확인 중' : '미확인'}</p>
            {ageAtCapture != null && <p>촬영 당시 나이 · 만 {ageAtCapture}세</p>}
            <p>{editedHeight === null ? '등록된 키' : '촬영 당시 키'} · {height ? height + ' cm' : registeredHeight != null && editedHeight === null ? '분석 범위 밖' : '미입력'}</p>
            <details key={picked.url}>
              <summary>촬영 정보 수정</summary>
              <label>촬영일 <input type="date" value={day} onChange={e => setEditedDay(e.target.value)} /></label>
              <label>촬영 당시 키 (100~220cm) <input type="number" min={100} max={220} step="0.1" value={height} onChange={e => setEditedHeight(e.target.value)} /></label>
            </details>
            {openPoseEnabled ? <label>좌표 추출 모델 <select value={extractor} onChange={e=>setExtractor(e.target.value as typeof extractor)}><option value="mediapipe">MediaPipe Full · 기기 GPU 우선</option><option value="openpose">OpenPose BODY_25 · 학교 GPU</option></select></label> : <p>좌표 추출 · MediaPipe Full</p>}
            {extractor === 'openpose' && <p className="muted">MP4를 분석 PC로 보내고, 크기를 줄인 영상 프레임을 학교 GPU로 전송합니다. GPU 배정 대기가 있을 수 있습니다.</p>}
            <label>촬영 방식 <select value={mode} onChange={e=>setMode(e.target.value as typeof mode)}><option value="fixed">고정 촬영</option><option value="guided">따라가며 촬영</option></select></label>
          </div>}
          <VideoPreview picked={picked} onDuration={(d) => setPicked((p) => (p ? { ...p, durationSec: d } : p))} />
          <dl className="card" style={{ marginTop: 16, borderRadius: 'var(--r-md)', padding: '14px 16px', display: 'grid', gridTemplateColumns: 'auto 1fr', rowGap: 8, fontSize: 15 }}>
            <dt className="muted">파일명</dt>
            <dd style={{ fontWeight: 600, textAlign: 'right', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{picked.name}</dd>
            <dt className="muted">영상 길이</dt>
            <dd style={{ fontWeight: 600, textAlign: 'right' }}>{fmtLength(picked.durationSec)}</dd>
            <dt className="muted">파일 날짜</dt>
            <dd style={{ fontWeight: 600, textAlign: 'right' }}>{fmtDate(picked.fileDate.toISOString())}</dd>
          </dl>
          <button className="btn btn-outline btn-sm" style={{ marginTop: 12 }} onClick={openPicker}>
            <RefreshIcon size={18} /> 다른 영상 선택
          </button>
        </section>
      )}

      {/* 06-2 분석 진행 중 · 완료 */}
      {(phase === 'running' || phase === 'done') && picked && <FileCard picked={picked} />}
      {phase === 'running' && (
        <section style={{ margin: '24px 20px 0' }} role="status" aria-live="polite">
          <p style={{ fontSize: 16, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6 }} className="primary">
            <SpinnerIcon size={16} /> 분석하고 있습니다
          </p>
          {extractor === 'openpose' && !isDemo ? <p style={{marginTop:12}}>OpenPose BODY_25 · 학교 GPU 분석</p> : <><p style={{ marginTop: 4, fontSize: 48, fontWeight: 800, lineHeight: 1, letterSpacing: '-0.02em' }}>
            {pct}
            <span style={{ fontSize: 26 }}>%</span>
          </p>
          <div className="bar" style={{ marginTop: 16, height: 12 }}>
            <div style={{ width: `${pct}%` }} />
          </div>
          </>}
          <p style={{ marginTop: 8, fontSize: 14 }} className="muted">{isDemo ? '예상 남은 시간 약 '+remainSec+'초' : progressText || (pct >= 90 ? '좌표로 모델값 계산 중' : '영상 프레임에서 관절 위치 추출 중')}</p>
          <ol style={{ marginTop: 24, borderRadius: 'var(--r-lg)', border: '1px solid var(--border)' }}>
            {(isDemo ? STAGES : ["영상 확인","프레임별 관절 위치 추출","보행 구간과 모델값 계산","결과 저장"]).map((name, i) => {
              const state = i < stageIndex ? 'done' : i === stageIndex ? 'now' : 'wait';
              return (
                <li
                  key={name}
                  style={{
                    padding: '14px 16px', display: 'flex', alignItems: 'center', gap: 12,
                    borderTop: i === 0 ? undefined : '1px solid var(--border)',
                    background: state === 'now' ? 'rgb(227 236 246 / .6)' : undefined,
                    borderRadius: i === 0 ? 'var(--r-lg) var(--r-lg) 0 0' : i === 3 ? '0 0 var(--r-lg) var(--r-lg)' : undefined,
                  }}
                >
                  {state === 'done' && <CheckCircleIcon size={22} className="ok-icon" />}
                  {state === 'now' && <SpinnerIcon size={22} className="primary" />}
                  {state === 'wait' && <CircleIcon size={22} className="faint" />}
                  <span style={{ fontSize: 16, fontWeight: state === 'now' ? 700 : 400, color: state === 'done' ? 'var(--fg-70)' : state === 'wait' ? 'var(--fg-50)' : undefined }}>
                    {name}
                  </span>
                  {state === 'now' && <span style={{ marginLeft: 'auto', fontSize: 13, fontWeight: 700 }} className="primary">진행 중</span>}
                  {state === 'wait' && <span style={{ marginLeft: 'auto', fontSize: 13, fontWeight: 600, color: 'var(--fg-40)' }}>대기</span>}
                </li>
              );
            })}
          </ol>
        </section>
      )}
      {phase === 'done' && (
        <section style={{ margin: '40px 20px 0', display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }} role="status">
          <span style={{ width: 96, height: 96, borderRadius: 999, background: 'rgb(23 191 99 / .15)', color: 'var(--ok)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <CheckIcon size={40} />
          </span>
          <h2 style={{ marginTop: 24, fontSize: 24, fontWeight: 800 }}>분석이 완료되었습니다</h2>
          <p style={{ marginTop: 8, fontSize: 16, lineHeight: 1.6 }} className="muted">
            {subject.name} 님의 측정 결과지가
            <br />
            준비되었습니다.
          </p>
        </section>
      )}

      {/* 06-3 분석 실패 */}
      {phase === 'failed' && picked && (
        <>
          <section role="alert" style={{ padding: '32px 20px 0', display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
            <span style={{ width: 80, height: 80, borderRadius: 999, background: 'rgb(244 33 46 / .1)', color: 'var(--danger)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <TriangleAlertIcon size={34} />
            </span>
            <h2 style={{ marginTop: 20, fontSize: 23, fontWeight: 800 }}>영상을 분석하지 못했습니다</h2>
            <p style={{ marginTop: 8, fontSize: 16 }} className="muted">아래 원인을 확인한 뒤 다시 시도해 주세요.</p>
          </section>
          <section style={{ margin: '24px 20px 0', borderRadius: 'var(--r-lg)', border: '2px solid rgb(244 33 46 / .3)', background: 'rgb(244 33 46 / .05)', padding: 20 }}>
            <p style={{ fontSize: 14, fontWeight: 700, color: 'var(--danger)' }}>원인</p>
            <p style={{ marginTop: 4, fontSize: 18, fontWeight: 800, lineHeight: 1.4 }}>{failure?.reason ?? '영상을 분석하지 못했습니다'}</p>
            <ul style={{ marginTop: 16, display: 'grid', gap: 10, fontSize: 15, lineHeight: 1.6 }}>
              {['대상자의 머리부터 발끝까지 화면에 보이는지', '여러 걸음을 이어서 걷는 장면이 포함되어 있는지', '화면이 심하게 흔들리거나 어둡지 않은지'].map((t) => (
                <li key={t} style={{ display: 'flex', gap: 10 }}>
                  <CheckIcon size={16} className="primary check-mt" /> {t}
                </li>
              ))}
            </ul>
          </section>
          <FileCard picked={picked} errorCode={failure?.code ?? 'GA-000'} dim />
        </>
      )}

      {/* 하단 고정 영역 */}
      <div className="bottom-cta" style={phase === 'pick' ? { paddingTop: 12 } : undefined}>
        {phase === 'pick' && (
          <>
            <p className="hint">영상을 선택하면 분석을 시작할 수 있습니다</p>
            <button className="btn btn-primary" disabled>
              <PlayIcon size={16} /> 분석 시작
            </button>
          </>
        )}
        {phase === 'preview' && (
          <button className="btn btn-primary" disabled={invalidHeight || picked?.metadataDay === undefined} onClick={start}>
            <PlayIcon size={16} /> 분석 시작
          </button>
        )}
        {phase === 'running' && (
          <div className="stack"><p style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: 15, color: 'var(--fg-80)', padding: '4px 0' }}>
            <InfoIcon size={20} className="primary" /> 분석이 끝날 때까지 앱을 닫지 말아 주세요.
          </p>{!isDemo && <button className="btn btn-outline" onClick={()=>setPhase('preview')}>분석 취소</button>}</div>
        )}
        {phase === 'done' && (
          <button className="btn btn-primary" onClick={() => navigate(`/subjects/${subject.id}/results/${resultId}`, { replace: true })}>
            <FileIcon /> 결과지 보기
          </button>
        )}
        {phase === 'failed' && (
          <div className="stack">
            <button className="btn btn-primary" disabled={invalidHeight} onClick={start}>
              <RotateIcon /> 다시 시도
            </button>
            <button className="btn btn-outline" onClick={openPicker}>
              <FolderOpenIcon /> 다른 영상 선택
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function FileCard({ picked, errorCode, dim }: { picked: Picked; errorCode?: string | undefined; dim?: boolean | undefined }) {
  return (
    <section className="card" style={{ margin: errorCode ? '16px 20px 0' : '20px 20px 0', padding: 12, display: 'flex', alignItems: 'center', gap: 12 }}>
      <Thumb picked={picked} size={56} dim={dim} />
      <div style={{ minWidth: 0, flex: 1 }}>
        <p style={{ fontSize: 16, fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{picked.name}</p>
        <p style={{ fontSize: 14 }} className="muted">
          {fmtLength(picked.durationSec)} · {errorCode ? `오류 코드 ${errorCode}` : `파일 날짜 ${fmtDate(picked.fileDate.toISOString())}`}
        </p>
      </div>
    </section>
  );
}

/** 작은 미리보기. 실제 영상이면 첫 장면을, 예시 영상이면 그림을 보여준다. */
function Thumb({ picked, size, dim }: { picked: Picked; size: number; dim?: boolean | undefined }) {
  const style = { width: size, height: size, borderRadius: 'var(--r-md)', objectFit: 'cover' as const, flexShrink: 0, opacity: dim ? 0.7 : 1, background: 'var(--secondary)' };
  if (picked.url) return <video src={`${picked.url}#t=0.5`} muted playsInline preload="metadata" style={style} />;
  return <SampleArt style={style} />;
}

/** 예시 영상 자리에 쓰는 간단한 그림(복도를 걷는 사람). */
function SampleArt({ style }: { style: CSSProperties }) {
  return (
    <svg viewBox="0 0 300 400" preserveAspectRatio="xMidYMid slice" style={style} aria-hidden="true">
      <rect width="300" height="400" fill="#33424c" />
      <path d="M0 400 L110 170 L190 170 L300 400Z" fill="#4a5b66" />
      <path d="M110 170 L110 0 M190 170 L190 0" stroke="#5b6d78" strokeWidth="2" />
      <g fill="none" stroke="#e3ecf6" strokeWidth="9" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="150" cy="150" r="16" fill="#e3ecf6" stroke="none" />
        <path d="M150 170 L148 250 M148 190 L126 230 M148 190 L172 226 M148 250 L128 320 L118 340 M148 250 L170 312 L186 334" />
      </g>
    </svg>
  );
}

/** 06-1의 큰 미리보기. 재생 버튼과 진행 막대는 디자인에 맞춰 직접 그린다. */
function VideoPreview({ picked, onDuration }: { picked: Picked; onDuration: (sec: number) => void }) {
  const ref = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [cur, setCur] = useState(0);
  const total = picked.durationSec ?? 0;

  const toggle = () => {
    const v = ref.current;
    if (!v) return;
    if (v.paused) void v.play();
    else v.pause();
  };

  return (
    <div style={{ position: 'relative', width: '100%', aspectRatio: '3 / 4', borderRadius: 'var(--r-lg)', overflow: 'hidden', background: 'var(--secondary)' }}>
      {picked.url ? (
        <video
          ref={ref}
          src={picked.url}
          playsInline
          preload="metadata"
          style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'contain' }}
          onLoadedMetadata={(e) => onDuration(e.currentTarget.duration)}
          onTimeUpdate={(e) => setCur(e.currentTarget.currentTime)}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
        />
      ) : (
        <SampleArt style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }} />
      )}
      {picked.url && (
        <button
          aria-label={playing ? '일시 정지' : '재생'}
          onClick={toggle}
          style={{
            position: 'absolute', inset: 0, margin: 'auto', width: 80, height: 80, borderRadius: 999,
            background: 'rgb(15 20 25 / .6)', backdropFilter: 'blur(6px)', color: '#fff',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            opacity: playing ? 0 : 1, transition: 'opacity .2s',
          }}
        >
          {playing ? <PauseIcon size={30} /> : <PlayIcon size={30} />}
        </button>
      )}
      <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, padding: 16, background: 'linear-gradient(to top, rgb(15 20 25 / .8), transparent)' }}>
        <div style={{ height: 6, borderRadius: 999, background: 'rgb(255 255 255 / .3)' }}>
          <div style={{ height: 6, borderRadius: 999, background: 'var(--primary)', width: total ? `${(cur / total) * 100}%` : 0 }} />
        </div>
        <div style={{ marginTop: 8, display: 'flex', justifyContent: 'space-between', color: '#fff', fontSize: 13, fontWeight: 600 }}>
          <span>{fmtClock(cur)}</span>
          <span>{total ? fmtClock(total) : '--:--'}</span>
        </div>
      </div>
    </div>
  );
}
