/**
 * 새 디자인 화면의 시작점. 화면 번호는 UX Pilot 디자인의 화면 이름과 같다.
 *
 *   #/login                                   01 로그인
 *   #/subjects                                02 대상자 목록
 *   #/subjects/new                            04 대상자 등록
 *   #/subjects/:id                            03 대상자 상세
 *   #/subjects/:id/edit                       04-1 대상자 정보 수정
 *   #/subjects/:id/overview                   03-1 전체 기록 해석
 *   #/subjects/:id/results                    05 이전 결과 목록
 *   #/subjects/:id/analyze                    06 새 보행 영상 분석(선택·미리보기·진행·실패)
 *   #/subjects/:id/results/:rid               07 분석 결과지(+08 다운로드 시트)
 *   #/subjects/:id/results/:rid/chart/:joint  07-2 그래프 확대 보기
 */
import { useEffect, type ReactNode } from 'react';
import { isDemo } from './data';
import { DemoPanel } from './components/DemoPanel';
import { AnalyzePage } from './pages/AnalyzePage';
import { ChartZoomPage } from './pages/ChartZoomPage';
import { LoginPage } from './pages/LoginPage';
import { NotFound } from './pages/NotFound';
import { ReportPage } from './pages/ReportPage';
import { ResultListPage } from './pages/ResultListPage';
import { SubjectDetailPage } from './pages/SubjectDetailPage';
import { SubjectFormPage } from './pages/SubjectFormPage';
import { SubjectListPage } from './pages/SubjectListPage';
import { SubjectOverviewPage } from './pages/SubjectOverviewPage';
import { ParamsContext, RouterProvider, matchPath, useRouter } from './router';
import { SessionProvider, useSession } from './session';

const ROUTES: [string, () => ReactNode][] = [
  ['/subjects', () => <SubjectListPage />],
  ['/subjects/new', () => <SubjectFormPage />],
  ['/subjects/:subjectId', () => <SubjectDetailPage />],
  ['/subjects/:subjectId/edit', () => <SubjectFormPage />],
  ['/subjects/:subjectId/overview', () => <SubjectOverviewPage />],
  ['/subjects/:subjectId/results', () => <ResultListPage />],
  ['/subjects/:subjectId/analyze', () => <AnalyzePage />],
  ['/subjects/:subjectId/results/:recordId', () => <ReportPage />],
  ['/subjects/:subjectId/results/:recordId/chart/:joint', () => <ChartZoomPage />],
];

function Screens() {
  const { path, navigate } = useRouter();
  const { loggedIn } = useSession();

  // 로그인 전에는 로그인 화면만, 로그인 후에는 로그인 화면 대신 대상자 목록을 보여준다.
  const target = !loggedIn ? (path === '/login' ? null : '/login') : path === '/login' || path === '/' ? '/subjects' : null;
  useEffect(() => {
    if (target) navigate(target, { replace: true });
  }, [target, navigate]);
  if (target) return null;
  if (!loggedIn) return <LoginPage />;

  for (const [pattern, render] of ROUTES) {
    const params = matchPath(pattern, path);
    if (params) {
      // key: 다른 화면·대상자·결과로 가면 화면 상태를 새로 시작한다.
      // 그래프 확대 화면에서 관절 탭만 바꿀 때는 다시 불러오지 않도록 관절은 key에서 뺀다.
      const key = `${pattern}|${params.subjectId ?? ''}|${params.recordId ?? ''}`;
      return (
        <ParamsContext.Provider key={key} value={params}>
          {render()}
        </ParamsContext.Provider>
      );
    }
  }
  return <NotFound />;
}

export function App() {
  return (
    <RouterProvider>
      <SessionProvider>
        <Screens />
        {isDemo && <DemoPanel />}
      </SessionProvider>
    </RouterProvider>
  );
}
