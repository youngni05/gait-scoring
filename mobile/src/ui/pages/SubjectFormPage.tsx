/**
 * 04 대상자 등록 · 04-1 대상자 정보 수정.
 * 두 화면은 입력 항목이 같아서 하나의 화면으로 만들고, 주소에 대상자 번호가 있으면 수정으로 동작한다.
 */
import { useEffect, useRef, useState } from 'react';
import { BirthDatePicker } from '../components/BirthDatePicker';
import { PageError, PageLoading, TopBar } from '../components/layout';
import { AlertCircleIcon, CalculatorIcon, CalendarIcon, CheckCircleIcon, InfoIcon } from '../components/icons';
import { ageOn, dataSource, fmtBirth, maskBirth, parseBirth, type Sex, type Subject } from '../data';
import { useLeaveGuard, useNavigate, useParams } from '../router';
import { useAsync } from '../session';
import { NotFound } from './NotFound';
import { ApiError, errorMessage } from '../../api/client';
import { validateParticipant } from '../../api/participant-input';

type Errors = Partial<Record<'name' | 'sex' | 'birth' | 'height', string>>;

/** 수정이면 대상자 정보를 먼저 불러온 뒤 입력 화면을 연다. */
export function SubjectFormPage() {
  const { subjectId } = useParams();
  const navigate = useNavigate();
  const { data, loading, error, reload } = useAsync(
    (signal) => (subjectId ? dataSource.getSubject(subjectId, signal) : Promise.resolve(null)),
    [subjectId],
  );
  if (!subjectId) return <SubjectForm editing={undefined} />;
  const back = () => navigate(`/subjects/${subjectId}`);
  if (loading) return <PageLoading title="대상자 정보 수정" nav="back" onNav={back} />;
  if (error) return <PageError title="대상자 정보 수정" onNav={back} onRetry={reload} />;
  if (!data) return <NotFound />;
  return <SubjectForm editing={data} />;
}

function SubjectForm({ editing }: { editing: Subject | undefined }) {
  const navigate = useNavigate();
  const isEdit = Boolean(editing);

  const [name, setName] = useState(editing?.name ?? '');
  const [sex, setSex] = useState<Sex | null>(editing?.sex ?? null);
  const [birth, setBirth] = useState(editing ? fmtBirth(editing.birth) : '');
  const [height, setHeight] = useState(editing?.heightCm == null ? '' : String(editing.heightCm));
  const [errors, setErrors] = useState<Errors>({});
  const [tried, setTried] = useState(false);
  const [saving, setSaving] = useState(false);
  const [savedId, setSavedId] = useState<string | null>(null);
  useEffect(() => {
    if (savedId) navigate(`/subjects/${savedId}`, { replace: true });
  }, [savedId, navigate]);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [uncertain, setUncertain] = useState(false);
  const alive = useRef(true);
  const savingNow = useRef(false);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  useLeaveGuard(saving, '대상자 정보를 저장 중입니다. 화면을 떠나도 서버에 저장될 수 있습니다. 나갈까요?');
  const [calendarOpen, setCalendarOpen] = useState(false);
  const calendarButton = useRef<HTMLButtonElement>(null);
  const closeCalendar = () => { setCalendarOpen(false); calendarButton.current?.focus(); };

  const birthIso = parseBirth(birth);
  const age = birthIso ? ageOn(birthIso) : null;

  const validate = (): Errors => {
    const e: Errors = {};
    const { errors } = validateParticipant({ name, gender: sex === '여' ? 'F' : sex === '남' ? 'M' : '', birth_date: birthIso ?? '', height_cm: height });
    if (errors.name) e.name = errors.name;
    if (errors.gender) e.sex = errors.gender;
    if (errors.birth_date) e.birth = errors.birth_date;
    if (errors.height_cm) e.height = errors.height_cm;
    return e;
  };

  // 한 번 저장을 시도한 뒤에는 고칠 때마다 오류를 다시 계산해, 고친 항목의 안내가 바로 사라지게 한다.
  useEffect(() => {
    if (tried) setErrors(validate());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [name, sex, birth, height]);

  const save = () => {
    if (savingNow.current || uncertain) return;
    const e = validate();
    setErrors(e);
    setTried(true);
    if (Object.keys(e).length > 0) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      return;
    }
    const data = { name: name.trim(), sex: sex!, birth: birthIso!, heightCm: Number(height) };
    savingNow.current = true;
    setSaving(true);
    setSaveError(null);
    const request = editing ? dataSource.updateSubject(editing.id, data) : dataSource.createSubject(data);
    request
      .then((saved) => {
        if (!alive.current) return;
        savingNow.current = false;
        setSaving(false);
        // Run navigation after the render releases the save guard.
        setSavedId(saved.id);
      })
      .catch((error) => {
        if (!alive.current) return;
        // 저장이 이미 반영되었을 수도 있으므로 무작정 다시 누르게 하지 않고 목록 확인을 안내한다.
        const ambiguous = !(error instanceof ApiError) || error.status === 0 || error.status >= 500 || error.code === 'INVALID_RESPONSE';
        setUncertain(!editing && ambiguous);
        setSaveError(ambiguous ? '저장 결과를 확인하지 못했습니다. 이미 반영되었을 수 있으니 대상자 목록에서 확인해 주세요.' : errorMessage(error));
      })
      .finally(() => { if (alive.current) { savingNow.current = false; setSaving(false); } });
  };

  const errCount = Object.keys(errors).length;
  const back = () => navigate(editing ? `/subjects/${editing.id}` : '/subjects');

  return (
    <div className="screen has-cta">
      <TopBar
        title={isEdit ? '대상자 정보 수정' : '대상자 등록'}
        nav={isEdit ? 'back' : 'close'}
        onNav={back}
        subject={editing}
      />

      {errCount > 0 && (
        <div className="alert alert-danger" role="alert" style={{ margin: '20px 20px 0', alignItems: 'center', padding: '14px 16px' }}>
          <AlertCircleIcon size={20} />
          <p style={{ fontWeight: 600 }}>입력 항목 {errCount}개를 확인해 주세요.</p>
        </div>
      )}

      <form noValidate onSubmit={(e) => e.preventDefault()} style={{ padding: '24px 20px 0', display: 'grid', gap: 28 }}>
        {!isEdit && (
          <p style={{ fontSize: 14 }} className="muted">
            <span className="req" style={{ fontWeight: 700 }}>*</span> 표시는 필수 입력 항목입니다.
          </p>
        )}

        <div className="field">
          <label htmlFor="name">이름 <span className="req">*</span></label>
          <input
            id="name"
            className={`input ${errors.name ? 'invalid' : ''}`}
            placeholder="예) 홍길동"
            value={name}
            aria-invalid={Boolean(errors.name) || undefined}
            onChange={(e) => { setName(e.target.value) }}
          />
          {errors.name && <p className="field-error"><AlertCircleIcon size={16} />{errors.name}</p>}
        </div>

        <div className="field">
          <p className="label" id="sex-label">성별 <span className="req">*</span></p>
          <div className="choice-grid" role="radiogroup" aria-labelledby="sex-label">
            {(['여', '남'] as Sex[]).map((v) => (
              <button
                key={v}
                type="button"
                role="radio"
                aria-checked={sex === v}
                className={`choice ${errors.sex ? 'invalid' : ''}`}
                onClick={() => { setSex(v) }}
              >
                {sex === v && <CheckCircleIcon size={18} />}
                {v === '여' ? '여성' : '남성'}
              </button>
            ))}
          </div>
          {errors.sex && <p className="field-error"><AlertCircleIcon size={16} />{errors.sex}</p>}
        </div>

        <div className="field">
          <label htmlFor="birth">생년월일 <span className="req">*</span></label>
          <div className="input-wrap">
            <input
              id="birth"
              inputMode="numeric"
              className={`input pad-right ${errors.birth ? 'invalid' : ''}`}
              placeholder="YYYY.MM.DD"
              value={birth}
              aria-invalid={Boolean(errors.birth) || undefined}
              onChange={(e) => { setBirth(maskBirth(e.target.value)) }}
            />
            <button ref={calendarButton} type="button" className="icon-btn input-action"
              aria-label="생년월일 달력 열기" aria-expanded={calendarOpen} aria-controls="birth-calendar"
              onClick={() => setCalendarOpen(open => !open)}><CalendarIcon size={20} /></button>
          </div>
          {calendarOpen && <BirthDatePicker value={birthIso} onClose={closeCalendar}
            onSelect={date => { setBirth(fmtBirth(date)); closeCalendar(); }} />}
          {errors.birth && <p className="field-error"><AlertCircleIcon size={16} />{errors.birth}</p>}
          <div
            style={{ marginTop: 12, borderRadius: 'var(--r-md)', background: 'var(--accent)', padding: '12px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
            aria-live="polite"
          >
            <span style={{ fontSize: 15, color: 'var(--fg-70)', display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <CalculatorIcon size={16} className="primary" /> 나이 (자동 계산)
            </span>
            <span style={{ fontSize: 18, fontWeight: 800 }} className={age === null ? 'muted' : 'primary'}>
              {age === null ? '—' : `만 ${age}세`}
            </span>
          </div>
        </div>

        <div className="field">
          <label htmlFor="height">키 <span className="req">*</span></label>
          <div className="input-wrap">
            <input
              id="height"
              inputMode="decimal"
              className={`input ${errors.height ? 'invalid' : ''}`}
              style={{ paddingRight: 64 }}
              placeholder="예) 160"
              value={height}
              aria-invalid={Boolean(errors.height) || undefined}
              onChange={(e) => { setHeight(e.target.value.replace(/[^\d.]/g, '')) }}
            />
            <span className="input-suffix">cm</span>
          </div>
          {errors.height && <p className="field-error"><AlertCircleIcon size={16} />{errors.height}</p>}
        </div>

        {isEdit && (
          <div className="alert alert-info" style={{ borderRadius: 'var(--r-md)' }}>
            <InfoIcon size={20} />
            <p>
              저장된 분석 시작일 기준 나이는 유지됩니다. 결과지의 이름·성별·키는 현재 대상자 정보를 표시합니다.
            </p>
          </div>
        )}
      </form>

      <div className="bottom-cta">
        {saveError && <p className="hint" role="alert" style={{ color: 'var(--danger)' }}>{saveError}</p>}
        {uncertain && <button className="btn btn-outline" onClick={() => navigate('/subjects')}>대상자 목록에서 확인</button>}
        <button className="btn btn-primary" onClick={save} disabled={saving || uncertain}>
          {saving ? '저장 중…' : isEdit ? '수정 내용 저장' : '저장'}
        </button>
      </div>
    </div>
  );
}
