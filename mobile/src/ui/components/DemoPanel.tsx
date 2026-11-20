/**
 * 데모 상태 조절판. 실제 서버가 없어서 실패·빈 목록 같은 상태 화면을 자연스럽게 만들 수 없으므로,
 * 검토하는 사람이 원하는 상태를 골라 볼 수 있게 한다. 실제 앱에는 넣지 않는 초안 전용 도구다.
 */
import { useState } from 'react';
import { setDemo, useDemo, type DemoOptions } from '../data/demo';
import { useSession } from '../session';
import { CloseIcon, SlidersIcon } from './icons';

type Opt<K extends keyof DemoOptions> = { key: K; title: string; items: [DemoOptions[K], string][] };

const GROUPS: [Opt<'login'>, Opt<'subjectList'>, Opt<'analysis'>, Opt<'download'>] = [
  { key: 'login', title: '로그인 결과', items: [['success', '성공'], ['fail', '실패 (01-2)']] },
  { key: 'subjectList', title: '대상자 목록', items: [['normal', '정상'], ['loading', '계속 로딩 (02-2)'], ['empty', '빈 상태 (02-1)']] },
  { key: 'analysis', title: '영상 분석 결과', items: [['success', '성공'], ['partial', '일부 누락 (07-1)'], ['fail', '실패 (06-3)']] },
  { key: 'download', title: '결과 다운로드', items: [['success', '성공 (08-1)'], ['fail', '실패 (08-2)']] },
];

export function DemoPanel() {
  const demo = useDemo();
  const { loggedIn, logout } = useSession();
  const [open, setOpen] = useState(false);

  return (
    <>
      <button className="demo-fab" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        {open ? <CloseIcon size={16} /> : <SlidersIcon size={16} />} 데모
      </button>
      {open && (
        <div className="demo-panel" role="dialog" aria-label="데모 상태 설정">
          <h2>데모 상태 설정</h2>
          {GROUPS.map((g) => (
            <fieldset key={g.key}>
              <legend>{g.title}</legend>
              <div className="opts">
                {g.items.map(([val, label]) => (
                  <button
                    key={val}
                    aria-pressed={demo[g.key] === val}
                    onClick={() => setDemo({ [g.key]: val } as Partial<DemoOptions>)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </fieldset>
          ))}
          {loggedIn && (
            <fieldset>
              <legend>계정</legend>
              <div className="opts">
                <button onClick={logout}>로그아웃</button>
              </div>
            </fieldset>
          )}
          <p className="note">
            서버·분석 모델과 연결되지 않은 화면 초안입니다. 모든 대상자·결과는 예시용 가상 데이터이며 새로고침하면 처음 상태로 돌아갑니다. 서버 연동 화면은 connected.html에 있습니다.
          </p>
        </div>
      )}
    </>
  );
}
