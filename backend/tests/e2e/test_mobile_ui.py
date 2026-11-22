"""Current mobile UI -> real API/PostgreSQL. Pose extraction alone is synthetic.

The replacement extraction module is served only by Playwright routing; no test
controls or fake analysis code are added to the product bundle.
"""
import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from playwright.sync_api import expect
from sqlalchemy import text
import test_browser as support


class MobileUiTests(unittest.TestCase):
    ui_root = 'mobile'
    setUpClass = classmethod(support.BrowserIntegrationTests.setUpClass.__func__)
    launch = classmethod(support.BrowserIntegrationTests.launch.__func__)
    drop_schema = classmethod(support.BrowserIntegrationTests.drop_schema.__func__)
    report_failure = support.BrowserIntegrationTests.report_failure

    def setUp(self):
        self.context = self.browser.new_context(viewport={'width': 390, 'height': 844})
        self.addCleanup(self.context.close)
        self.page = self.context.new_page()
        self.page.set_default_timeout(15000)
        self.api_events = []
        self.addCleanup(self.report_failure)
        self.page.goto(self.ui)
        self.page.get_by_label('아이디', exact=True).fill(self.username)
        self.page.get_by_label('비밀번호', exact=True).fill(self.password)
        self.page.get_by_role('button', name='로그인', exact=True).click()
        expect(self.page.get_by_role('button', name='대상자 등록', exact=True)).to_be_visible()
        self.name = '모바일통합-' + uuid4().hex[:8]

    def fill_person(self):
        self.page.get_by_role('button', name='대상자 등록', exact=True).click()
        self.page.get_by_label('이름', exact=False).fill(self.name)
        self.page.get_by_role('radio', name='여성').click()
        self.page.get_by_label('생년월일', exact=False).fill('19500101')
        self.page.get_by_label('키', exact=False).fill('170.5')

    def create_person(self):
        self.fill_person()
        self.page.get_by_role('button', name='저장', exact=True).click()
        expect(self.page.get_by_role('button', name='대상자 정보 수정')).to_be_visible()
        self.person_id = self.page.url.split('#/subjects/')[1]

    def install_extractor(self):
        from app.ai_module import POSE_MODEL_SHA256
        coords = json.loads((support.ROOT / 'contracts/measurement-1/synthetic.request.json').read_text(encoding='utf-8'))['body']['keypoints_json']
        coords.update(schema_version='team-local-coords/1',
                      pose=dict(model='pose_landmarker_full', engine_version='1.2.3', delegate='CPU'),
                      marks=dict(stand_start=None, stand_end=None), frames=[dict(t=0, lm=None)])
        coords['capture']['recorded_at'] = None
        packet = dict(coords=coords, metadata=dict(recordedAtSource='unknown', rotationAppliedDegrees=0),
                      bridge_context=dict(profile='team-browser-full-cpu-video/1', source_sha256='unknown',
                                          model_sha256=POSE_MODEL_SHA256, input_object='ImageData',
                                          image_geometry='upright_unmirrored', clock_basis='video_pts',
                                          frame_audit_complete=True, fresh_tracker=True))
        script = '''export async function extractCapture(file, id, day, mode, signal) {
          if (window.holdExtraction) await new Promise((resolve, reject) => {
            signal.addEventListener('abort', () => reject(new DOMException('Aborted','AbortError')), {once:true});
          });
          signal.throwIfAborted();
          const packet = PACKET;
          packet.requestId = crypto.randomUUID();
          packet.coords.recording_id = packet.requestId;
          packet.coords.subject.id = id;
          if (window.oversizedTestPacket) packet.testPadding = 'x'.repeat(8000000);
          return packet;
        }
        export async function postJson() { throw Error('Not used by this test'); }
        '''.replace('PACKET', json.dumps(packet))
        self.page.route('**/src/ui/data/analysis.ts*', lambda route: route.fulfill(content_type='application/javascript', body=script))
        self.page.reload()
        expect(self.page.get_by_role('button', name='대상자 정보 수정')).to_be_visible()

    def open_capture(self):
        self.page.get_by_role('button', name='새 보행 영상 분석', exact=False).click()
        self.page.locator('input[type=file]').set_input_files(dict(name='synthetic.mp4', mimeType='video/mp4', buffer=b'synthetic'))
        expect(self.page.get_by_role('button', name='분석 시작', exact=True)).to_be_enabled()

    def count(self):
        with self.engine.connect() as conn:
            return conn.execute(text('SELECT count(*) FROM gait_measurements WHERE participant_id=:pid'), {'pid': self.person_id}).scalar_one()

    def test_current_registration_and_edit(self):
        self.create_person()
        self.page.get_by_role('button', name='대상자 정보 수정').click()
        self.page.get_by_label('키', exact=False).fill('172.5')
        self.page.get_by_role('button', name='수정 내용 저장', exact=True).click()
        expect(self.page.get_by_role('button', name='대상자 정보 수정')).to_be_visible()
        with self.engine.connect() as conn:
            self.assertEqual(float(conn.execute(text('SELECT height_cm FROM participants WHERE id=:id'), {'id': self.person_id}).scalar_one()), 172.5)

    def late_save(self, editing):
        if editing:
            self.create_person()
            self.page.get_by_role('button', name='대상자 정보 수정').click()
            self.page.get_by_label('키', exact=False).fill('172.5')
            endpoint = self.api + '/participants/' + self.person_id
        else:
            self.fill_person()
            endpoint = self.api + '/participants'
        held = []
        self.page.route(endpoint, lambda route: held.append(route) if route.request.method in ('POST', 'PATCH') else route.continue_())
        with self.page.expect_request(lambda request: request.url == endpoint and request.method in ('POST', 'PATCH')):
            self.page.get_by_role('button', name='수정 내용 저장' if editing else '저장', exact=True).click()
        self.page.once('dialog', lambda dialog: dialog.dismiss())
        self.page.get_by_role('button', name='뒤로' if editing else '닫기', exact=True).click()
        expect(self.page.get_by_role('button', name='저장 중…', exact=True)).to_be_visible()
        self.page.once('dialog', lambda dialog: dialog.accept())
        self.page.get_by_role('button', name='뒤로' if editing else '닫기', exact=True).click()
        expect(self.page.get_by_role('button', name='대상자 정보 수정' if editing else '대상자 등록', exact=True)).to_be_visible()
        destination = self.page.url
        response = held[0].fetch()
        self.assertEqual(response.status, 200 if editing else 201)
        held[0].fulfill(response=response)
        self.page.evaluate('() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))')
        self.assertEqual(self.page.url, destination)

    def test_late_create_does_not_navigate(self):
        self.late_save(False)

    def test_late_edit_does_not_navigate(self):
        self.late_save(True)

    def test_analysis_guards_history_and_document_unload(self):
        self.create_person()
        self.install_extractor()
        self.open_capture()
        self.page.evaluate('window.holdExtraction = true')
        self.page.get_by_role('button', name='분석 시작', exact=True).click()
        expect(self.page.get_by_role('button', name='분석 취소', exact=True)).to_be_visible()
        path = self.page.url
        self.page.once('dialog', lambda dialog: dialog.dismiss())
        self.page.go_back()
        expect(self.page).to_have_url(path)
        self.assertTrue(self.page.evaluate("() => { const e = new Event('beforeunload',{cancelable:true}); window.dispatchEvent(e); return e.defaultPrevented; }"))
        self.page.once('dialog', lambda dialog: dialog.accept())
        self.page.go_back()
        expect(self.page.get_by_role('button', name='대상자 정보 수정')).to_be_visible()
        self.assertEqual(self.count(), 0)

    def recover_after_reload(self, accepted):
        self.create_person()
        self.install_extractor()
        self.open_capture()
        endpoint = self.api + '/team/participants/' + self.person_id + '/recordings'
        attempts = []
        def lose_once(route):
            if route.request.method != 'POST':
                return route.continue_()
            attempts.append(route.request.post_data)
            if len(attempts) == 1:
                if accepted:
                    self.assertEqual(route.fetch().status, 202)
                route.abort('failed')
            else:
                route.continue_()
        self.page.route(endpoint, lose_once)
        self.page.get_by_role('button', name='분석 시작', exact=True).click()
        expect(self.page.get_by_role('button', name='다시 시도', exact=True)).to_be_visible()
        self.assertEqual(self.count(), int(accepted))
        self.page.once('dialog', lambda dialog: dialog.accept())
        self.page.reload()
        recover = self.page.get_by_role('button', name='이전 저장 요청 이어가기', exact=True)
        expect(recover).to_be_visible()
        recover.click()
        expect(self.page.get_by_role('heading', name='좌표가 서버에 저장되었습니다')).to_be_visible()
        self.assertEqual(self.count(), 1)
        self.assertEqual(len(attempts), 1 if accepted else 2)
        if not accepted:
            self.assertEqual(attempts[0], attempts[1])
        self.assertEqual(self.page.evaluate('''async () => {
          const db = await new Promise((resolve,reject) => { const r=indexedDB.open('gait-pending-captures',1); r.onsuccess=()=>resolve(r.result); r.onerror=()=>reject(r.error); });
          const count=await new Promise(resolve=>{const r=db.transaction('pending').objectStore('pending').count();r.onsuccess=()=>resolve(r.result);});db.close();return count;
        }'''), 0)

    def test_lost_receipt_reload_queries_existing_record(self):
        self.recover_after_reload(True)

    def test_unsent_request_reload_replays_identical_bytes(self):
        self.recover_after_reload(False)

    def test_outbox_is_partitioned_and_first_writer_wins(self):
        # Real IndexedDB transactions, including racing tabs/repeated attempts.
        result = self.page.evaluate('''async () => {
          const {keepPending,readPending,clearPending} = await import('/src/ui/data/pending-capture.ts');
          const a={key:'account-a/subject',id:'first',subjectId:'subject',fileName:'a.mp4',body:'first body'};
          const b={...a,id:'second',body:'second body'};
          const rows=await Promise.all([keepPending(a),keepPending(b)]);
          await clearPending(a.key,'wrong-id');
          const kept=await readPending(a.key), other=await readPending('account-b/subject');
          await clearPending(a.key,kept.id);
          return {same:rows[0].id===rows[1].id,kept:kept.id,other};
        }''')
        self.assertEqual(result, dict(same=True, kept='first', other=None))

    def seed_records(self, count):
        ids = []
        with self.engine.begin() as conn:
            for i in range(count):
                mid = uuid4(); ids.append(str(mid))
                conn.execute(text('''INSERT INTO gait_measurements
                    (id,participant_id,age,keypoints_json,extractor_name,extractor_version,measured_at,analysis_started_at,created_by)
                    VALUES (:id,:pid,76,CAST(:pose AS jsonb),'synthetic','1',NULL,:at,:uid)'''),
                    dict(id=mid,pid=self.person_id,uid=self.user_id,pose=json.dumps({"private_pose": True}),
                         at=datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(days=i)))
        return ids

    def test_pagination_and_subject_summary_avoid_detail_requests(self):
        self.create_person()
        self.seed_records(25)
        requests = []
        self.page.on('request', lambda request: requests.append(request.url) if request.url.startswith(self.api) else None)
        self.page.reload()
        expect(self.page.get_by_role('button', name='이전 결과 조회', exact=False)).to_be_visible()
        self.assertFalse(any('/recordings' in url or 'recording-summaries' in url for url in requests))
        self.page.get_by_role('button', name='이전 결과 조회', exact=False).click()
        expect(self.page.locator('.record-btn')).to_have_count(20)
        self.assertFalse(any('offset=20' in url for url in requests))
        self.page.get_by_role('button', name='다음 페이지', exact=True).click()
        expect(self.page.locator('.record-btn')).to_have_count(5)
        self.page.get_by_role('button', name='이전 페이지', exact=True).click()
        expect(self.page.locator('.record-btn')).to_have_count(20)
        self.assertFalse(any('/recordings' in url for url in requests))

    def test_legacy_date_and_zero_graph_in_current_report_and_zoom(self):
        self.create_person()
        mid = self.seed_records(1)[0]
        graph = json.dumps(dict(schema_version='gait-cycle-degrees-1',left=[0]*51,right=[0]*51))
        with self.engine.begin() as conn:
            conn.execute(text('''INSERT INTO gait_analysis_results
                (id,measurement_id,gdi_score,hip_graph_json,knee_graph_json,ankle_graph_json,model_version,analyzed_at)
                VALUES (:id,:mid,0,CAST(:g AS jsonb),CAST(:g AS jsonb),CAST(:g AS jsonb),'synthetic',now())'''),dict(id=uuid4(),mid=mid,g=graph))
        self.page.goto(self.ui+'/#/subjects/'+self.person_id+'/results/'+mid)
        expect(self.page.get_by_text('분석 시작일',exact=True)).to_be_visible()
        expect(self.page.get_by_text('2026.01.01 (목)',exact=True)).to_be_visible()
        graphs=self.page.locator('svg[aria-label*="관절 각도 그래프"]')
        expect(graphs).to_have_count(3)
        for zoom in (False,True):
            if zoom:
                self.page.get_by_role('button',name='확대',exact=True).first.click()
                expect(graphs).to_have_count(1)
            values=graphs.locator('path').evaluate_all('(els)=>els.map(e=>e.getAttribute("d"))')
            self.assertTrue(values)
            self.assertTrue(all(v and 'NaN' not in v and 'Infinity' not in v for v in values))

    def wait_for_held_route(self, held):
        deadline = support.time.monotonic() + 10
        while not held and support.time.monotonic() < deadline:
            self.page.wait_for_timeout(20)
        self.assertEqual(len(held), 1)

    def open_report_stub(self, state='pending'):
        self.create_person()
        rid = str(uuid4())
        record = dict(id=rid, subjectId=self.person_id, measuredAt='2026-01-01T00:00:00Z',
                      dateBasis='capture', ageAtCapture=76, heightAtCapture=170.5,
                      fileName='synthetic.mp4', serverState=state, live=None,
                      gdi=None, cadence=None, symmetry=None, curves={}, missing={})
        endpoint = self.api + '/team/recordings/' + rid
        self.page.route(endpoint, lambda route: route.fulfill(json=record))
        self.page.goto(self.ui+'/#/subjects/'+self.person_id+'/results/'+rid)
        expect(self.page.get_by_role('heading', name='좌표가 서버에 저장되었습니다' if state == 'pending' else '서버 분석에 실패했습니다')).to_be_visible()
        return endpoint, record

    def test_pending_poll_preserves_screen_on_slow_and_failed_refresh(self):
        self.page.clock.install()
        endpoint, record = self.open_report_stub()
        held = []
        self.page.route(endpoint, lambda route: held.append(route))
        self.page.locator('main.screen').evaluate("el => el.dataset.retained = 'yes'")
        self.page.clock.fast_forward(3000)
        expect(self.page.get_by_role('button', name='상태 확인 중…')).to_be_disabled()
        expect(self.page.locator('main[data-retained="yes"]')).to_be_visible()
        self.wait_for_held_route(held)
        held.pop().abort('failed')
        expect(self.page.get_by_role('alert')).to_contain_text('최신 분석 상태를 확인하지 못했습니다')
        expect(self.page.locator('main[data-retained="yes"]')).to_be_visible()
        self.page.clock.fast_forward(10000)
        expect(self.page.get_by_role('button', name='상태 확인 중…')).to_be_disabled()
        self.wait_for_held_route(held)
        held.pop().fulfill(json={**record, 'serverState': 'legacy'})
        expect(self.page.get_by_text('촬영일', exact=True)).to_be_visible()

    def test_pending_long_wait_keeps_navigation_available(self):
        self.page.clock.install()
        endpoint, _ = self.open_report_stub()
        self.page.route(endpoint, lambda route: None)  # A slow GET must not stop the notice timer.
        self.page.clock.fast_forward(60000)
        expect(self.page.get_by_text('이 화면에서 1분 이상 기다리고 있습니다.', exact=False)).to_be_visible()
        self.page.get_by_role('button', name='이전 결과 목록으로', exact=True).click()
        expect(self.page).to_have_url(self.ui+'/#/subjects/'+self.person_id+'/results')

    def test_retry_has_single_inflight_request_and_recovers_after_get_failure(self):
        endpoint, record = self.open_report_stub('failed')
        held = []
        self.page.route(endpoint+'/retry', lambda route: held.append(route))
        button = self.page.get_by_role('button', name='같은 촬영 다시 분석', exact=True)
        button.evaluate('(el) => { el.click(); el.click(); }')
        expect(self.page.get_by_role('button', name='재분석 요청 중…')).to_be_disabled()
        self.wait_for_held_route(held)
        self.page.route(endpoint, lambda route: route.abort('failed'))
        held[0].fulfill(json={'id':record['id']}, status=202)
        expect(self.page.get_by_role('heading', name='좌표가 서버에 저장되었습니다')).to_be_visible()
        expect(self.page.get_by_role('alert')).to_contain_text('최신 분석 상태를 확인하지 못했습니다')
        self.page.route(endpoint, lambda route: route.fulfill(json={**record, 'serverState':'legacy'}))
        self.page.get_by_role('button', name='지금 상태 확인', exact=True).click()
        expect(self.page.get_by_text('촬영일', exact=True)).to_be_visible()

    def rejected_capture(self):
        self.create_person()
        self.install_extractor()
        self.open_capture()
        endpoint = self.api + '/team/participants/' + self.person_id + '/recordings'
        self.page.route(endpoint, lambda route: route.fulfill(status=422, json={'detail':'Invalid coordinate packet'}))
        self.page.get_by_role('button', name='분석 시작', exact=True).click()
        expect(self.page.get_by_role('button', name='다시 분석하기', exact=True)).to_be_visible()

    def test_rejected_request_cleanup_requires_confirmation_and_survives_reload(self):
        self.rejected_capture()
        self.page.once('dialog', lambda d: d.accept())
        self.page.reload()
        clean = self.page.get_by_role('button', name='다시 분석하기', exact=True)
        expect(clean).to_be_visible()
        self.page.once('dialog', lambda d: d.dismiss())
        clean.click()
        expect(clean).to_be_visible()
        self.page.once('dialog', lambda d: d.accept())
        clean.click()
        expect(self.page.get_by_role('button', name='휴대폰에서 영상 선택', exact=False)).to_be_visible()
        self.assertEqual(self.count(), 0)

    def test_rejected_request_cleanup_keeps_body_when_server_check_fails(self):
        self.rejected_capture()
        self.page.route(self.api+'/team/recordings/*', lambda route: route.fulfill(status=500, json={'detail':'unavailable'}))
        self.page.once('dialog', lambda d: d.accept())
        self.page.get_by_role('button', name='다시 분석하기', exact=True).click()
        expect(self.page.get_by_text('저장 여부를 확인하지 못해 요청을 보관했습니다.', exact=False)).to_be_visible()
        self.page.once('dialog', lambda d: d.accept())
        self.page.reload()
        expect(self.page.get_by_role('button', name='다시 분석하기', exact=True)).to_be_visible()

    def test_uncertain_request_cannot_be_discarded_after_later_rejection(self):
        self.create_person()
        self.install_extractor()
        self.open_capture()
        endpoint = self.api + '/team/participants/' + self.person_id + '/recordings'
        self.page.route(endpoint, lambda route: route.abort('failed'))
        self.page.get_by_role('button', name='분석 시작', exact=True).click()
        expect(self.page.get_by_role('button', name='다시 시도', exact=True)).to_be_visible()
        self.page.route(endpoint, lambda route: route.fulfill(status=422, json={'detail':'Invalid coordinate packet'}))
        self.page.get_by_role('button', name='다시 시도', exact=True).click()
        expect(self.page.get_by_text('이전 저장 여부가 불확실해 요청을 보관합니다.', exact=False)).to_be_visible()
        expect(self.page.get_by_role('button', name='다시 분석하기', exact=True)).to_have_count(0)

    def test_cleanup_rejects_invalid_success_and_proxy_404(self):
        self.rejected_capture()
        endpoint = self.api+'/team/recordings/*'
        for status, body, content_type in [(200, 'null', 'application/json'), (404, '<h1>Missing proxy route</h1>', 'text/html')]:
            with self.subTest(status=status):
                def respond(route):
                    route.fulfill(status=status, body=body, content_type=content_type)
                self.page.route(endpoint, respond)
                self.page.once('dialog', lambda d: d.accept())
                self.page.get_by_role('button', name='다시 분석하기', exact=True).click()
                expect(self.page.get_by_text('저장 여부를 확인하지 못해 요청을 보관했습니다.', exact=False)).to_be_visible()
                expect(self.page.get_by_role('button', name='다시 분석하기', exact=True)).to_be_enabled()
                self.page.unroute(endpoint, respond)
        self.page.once('dialog', lambda d: d.accept())
        self.page.get_by_role('button', name='다시 분석하기', exact=True).click()
        expect(self.page.get_by_role('button', name='휴대폰에서 영상 선택', exact=False)).to_be_visible()

    def test_indexeddb_fallback_blocks_concurrent_submissions(self):
        self.create_person()
        self.install_extractor()
        statuses = self.page.evaluate('''async (id) => {
          Object.defineProperty(navigator, 'locks', {value:undefined,configurable:true});
          const {liveSource} = await import('/src/ui/data/liveSource.ts');
          const file = new File(['synthetic'], 'synthetic.mp4', {type:'video/mp4'});
          const run = () => liveSource.analyzeVideo(id, file, () => {}, new AbortController().signal);
          return (await Promise.allSettled([run(),run()])).map(r=>r.status).sort();
        }''', self.person_id)
        self.assertEqual(statuses, ['fulfilled', 'rejected'])
        self.assertEqual(self.count(), 1)

    def test_oversized_packet_never_creates_pending_or_posts(self):
        self.create_person()
        self.install_extractor()
        self.open_capture()
        self.page.evaluate('window.oversizedTestPacket = true')
        posts = []
        self.page.on('request', lambda r: posts.append(r.url) if r.method == 'POST' and r.url.endswith('/recordings') else None)
        self.page.get_by_role('button', name='분석 시작', exact=True).click()
        expect(self.page.get_by_text('분석 데이터가 너무 큽니다.', exact=False)).to_be_visible()
        self.assertEqual(posts, [])
        self.assertEqual(self.count(), 0)
        expect(self.page.get_by_role('button', name='다른 영상 선택', exact=False)).to_be_enabled()

    def test_expired_owner_cannot_mark_request_rejected(self):
        result = self.page.evaluate('''async () => {
          const m = await import('/src/ui/data/pending-capture.ts');
          const key='expired-test';
          await m.keepPending({key,id:'r',subjectId:'p',fileName:'x',body:'{}',delivery:'rejected'});
          const db=await new Promise(resolve=>{const q=indexedDB.open('gait-pending-captures');q.onsuccess=()=>resolve(q.result)});
          await new Promise((resolve,reject)=>{const t=db.transaction('pending','readwrite');t.objectStore('pending').put({key:'!capture-lock:'+key,owner:'old',expires:0});t.oncomplete=resolve;t.onabort=reject});db.close();
          return m.withPendingLock(key,async lease=>{
            let fenced=false;
            try{await m.markDelivery(key,'r','rejected',422,{key:lease.key,owner:'old'})}catch{fenced=true}
            return {delivery:(await m.readPending(key)).delivery,fenced};
          });
        }''')
        self.assertEqual(result, dict(delivery='unconfirmed', fenced=True))
