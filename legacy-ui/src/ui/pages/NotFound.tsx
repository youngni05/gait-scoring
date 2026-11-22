import { TopBar } from '../components/layout';
import { useNavigate } from '../router';

/** 없는 대상자·결과 주소로 들어왔을 때(예: 새로고침 후 예시 데이터가 초기화된 경우). */
export function NotFound() {
  const navigate = useNavigate();
  return (
    <div className="screen">
      <TopBar title="찾을 수 없음" nav="back" onNav={() => navigate('/subjects')} />
      <section className="empty">
        <h2>요청한 화면을 찾을 수 없습니다</h2>
        <p>대상자 또는 결과가 삭제되었거나 주소가 올바르지 않습니다.</p>
        <button className="btn btn-primary" style={{ marginTop: 40 }} onClick={() => navigate('/subjects')}>
          대상자 목록으로
        </button>
      </section>
    </div>
  );
}
