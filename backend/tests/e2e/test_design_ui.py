"""New design screens against real API/PostgreSQL, isolated by the existing harness."""
import json
import unittest
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4
from datetime import datetime, timedelta, timezone

import jwt
from playwright.sync_api import expect
from sqlalchemy import text
import test_browser as support


class DesignUiTests(unittest.TestCase):
    setUpClass = classmethod(support.BrowserIntegrationTests.setUpClass.__func__)
    launch = classmethod(support.BrowserIntegrationTests.launch.__func__)
    drop_schema = classmethod(support.BrowserIntegrationTests.drop_schema.__func__)

    def setUp(self):
        self.context = self.browser.new_context(viewport={'width': 390, 'height': 844})
        self.addCleanup(self.context.close)
        self.page = self.context.new_page()
        self.addCleanup(self.report_failure)
        self.page.set_default_timeout(15000)
        self.page.goto(self.ui + "/design.html")
        self.page.get_by_label('서버 주소', exact=True).fill(self.api)
        self.login()
        self.name = 'UI합성-' + uuid4().hex[:8]

    report_failure = support.BrowserIntegrationTests.report_failure

    def login(self):
        self.page.get_by_label('아이디', exact=True).fill(self.username)
        self.page.get_by_label('비밀번호', exact=True).fill(self.password)
        with self.page.expect_response(lambda r: r.url == self.api + '/auth/login') as response:
            self.page.get_by_role('button', name='로그인', exact=True).click()
        self.assertEqual(response.value.status, 200)

    def fill_person(self):
        self.page.get_by_role('button', name='대상자 등록', exact=True).click()
        self.page.get_by_role('textbox', name='이름 *', exact=True).fill(self.name)
        self.page.get_by_role('radio', name='여성').click()
        self.page.get_by_role('textbox', name='생년월일', exact=False).fill('19500101')
        self.page.get_by_label('키', exact=False).fill('170.5')

    def create_person(self):
        self.fill_person()
        self.page.get_by_role('button', name='저장', exact=True).click()
        expect(self.page.get_by_role('button', name='대상자 정보 수정')).to_be_visible()
        self.person_id = self.page.url.split('#/subjects/')[1]

    def open_save(self):
        self.page.get_by_role('button', name='새 보행 측정', exact=False).click()
        self.page.get_by_role('button', name='합성 좌표 준비').click()
        return self.page.get_by_role('button', name='서버 저장 / 같은 요청 재시도')

    def measurement_rows(self):
        with self.engine.connect() as conn:
            return conn.execute(text('SELECT id,age FROM gait_measurements WHERE participant_id=:id'), {'id': self.person_id}).all()

    def test_register_edit_save_result_and_json_download(self):
        self.create_person()
        self.page.get_by_role('button', name='대상자 정보 수정').click()
        self.page.get_by_label('키', exact=False).fill('171.5')
        self.page.get_by_role('button', name='수정 내용 저장').click()
        expect(self.page.get_by_role('button', name='대상자 정보 수정')).to_be_visible()
        self.open_save().click()
        expect(self.page.get_by_role('button', name='저장 기록 보기', exact=True)).to_be_visible()
        rows = self.measurement_rows()
        self.assertEqual(len(rows), 1)
        mid = str(rows[0].id)
        self.page.get_by_role('button', name='측정 기록 확인').click()
        expect(self.page.get_by_text('분석 결과 없음', exact=True)).to_be_visible()
        self.page.locator('.record-btn').click()
        expect(self.page.get_by_role('heading', name='측정 저장 완료')).to_be_visible()
        expect(self.page.get_by_role('button', name='결과 다운로드')).to_have_count(0)
        # Seed an explicitly synthetic stored analysis; this does not run an AI model.
        with self.engine.begin() as conn:
            conn.execute(text('INSERT INTO gait_analysis_results (id,measurement_id,gdi_score,stride_length_cm,cadence_spm,model_version,analyzed_at) VALUES (:id,:mid,0,100,NULL,:version,now())'),
                         {'id': uuid4(), 'mid': mid, 'version': 'synthetic-ui-test'})
        self.page.get_by_role('button', name='결과 다시 확인').click()
        expect(self.page.get_by_text('현재 등록 키', exact=True)).to_be_visible()
        expect(self.page.get_by_text('일부 결과가 누락되었습니다')).to_be_visible()
        with self.page.expect_download() as downloaded:
            self.page.get_by_role('button', name='결과 다운로드').click()
        payload = json.loads(downloaded.value.path().read_text(encoding='utf-8'))
        self.assertEqual(payload['analysis_result']['gdi_score'], 0)
        self.assertIsNone(payload['analysis_result']['cadence_spm'])
        self.assertEqual(payload['measurement']['id'], mid)
        self.assertEqual(downloaded.value.suggested_filename, f'gait-result-{mid}.json')
        # Capture the real connected result at mobile viewport for visual QA.
        self.page.get_by_role('button', name='확인', exact=True).click()
        from pathlib import Path
        output = support.ROOT / 'mobile/local-output'
        output.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(output / 'api-report.png'), full_page=True)

    def test_expiry_renewal_keeps_pending_save(self):
        self.create_person()
        save = self.open_save()
        expired = jwt.encode({'sub': str(self.user_id), 'exp': datetime.now(timezone.utc) - timedelta(minutes=1)}, self.secret, algorithm='HS256')
        calls = []
        def intercept(route):
            request = route.request
            calls.append((request.post_data, request.headers.get('idempotency-key')))
            if len(calls) == 1:
                route.continue_(headers={**request.headers, 'authorization': 'Bearer ' + expired})
            else:
                route.continue_()
        self.page.route('**/participants/*/measurements', intercept)
        save.click()
        expect(self.page.get_by_role('dialog', name='인증 갱신')).to_be_visible()
        self.assertEqual(self.measurement_rows(), [])
        self.login()
        expect(self.page.get_by_role('dialog', name='인증 갱신')).to_have_count(0)
        save.click()
        expect(self.page.get_by_role('button', name='저장 기록 보기', exact=True)).to_be_visible()
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(len(self.measurement_rows()), 1)

    def test_analysis_start_korea_birthday_boundary_and_replay(self):
        self.create_person()
        token = self.context.request.post(self.api + '/auth/login', form={'username': self.username, 'password': self.password}).json()['access_token']
        headers = {'Authorization': 'Bearer ' + token}
        self.assertEqual(self.context.request.patch(self.api + '/participants/' + self.person_id, headers=headers, data={'birth_date': '1950-01-02'}).status, 200)
        body = json.loads((support.ROOT / 'contracts/measurement-1/synthetic.request.json').read_text(encoding='utf-8'))['body']
        body.pop('measured_at')
        body['keypoints_json']['schema_version'] = 'coords-2'
        body['keypoints_json']['subject']['id'] = self.person_id
        body['keypoints_json']['capture'].pop('recorded_at')
        body['processing_metadata']['schema_version'] = 'processing-2'
        body['processing_metadata'].pop('recordedAtSource')
        endpoint = self.api + '/participants/' + self.person_id + '/measurements'
        for started, expected_age in [('2026-01-01T14:59:59Z', 75), ('2026-01-01T15:00:00Z', 76)]:
            body['analysis_started_at'] = started
            request_headers = {**headers, 'Idempotency-Key': str(uuid4())}
            response = self.context.request.post(endpoint, headers=request_headers, data=body)
            self.assertEqual(response.status, 201, response.text())
            data = response.json()
            self.assertEqual(data['age'], expected_age)
            self.assertIsNone(data['measured_at'])
            self.assertIsNotNone(data['analysis_started_at'])
            replay = self.context.request.post(endpoint, headers=request_headers, data=body)
            self.assertEqual(replay.status, 200)
            self.assertEqual(replay.json(), data)
            with self.engine.connect() as conn:
                row = conn.execute(text('SELECT measured_at,analysis_started_at,keypoints_json,processing_metadata FROM gait_measurements WHERE id=:id'), {'id': data['id']}).one()
                self.assertIsNone(row.measured_at)
                self.assertNotIn('recorded_at', row.keypoints_json['capture'])
                self.assertNotIn('recordedAtSource', row.processing_metadata)

    def test_save_response_loss_retries_exact_request_once(self):
        self.create_person()
        save = self.open_save()
        calls = []
        def lose_first(route):
            request = route.request
            calls.append((request.post_data, request.headers.get('idempotency-key')))
            if len(calls) == 1:
                response = route.fetch()
                self.assertEqual(response.status, 201)
                route.abort('failed')
            else:
                route.continue_()
        self.page.route('**/participants/*/measurements', lose_first)
        save.click()
        expect(self.page.get_by_text('서버 응답을 확인하지 못했습니다.', exact=False)).to_be_visible()
        save.click()
        expect(self.page.get_by_role('button', name='저장 기록 보기', exact=True)).to_be_visible()
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(len(self.measurement_rows()), 1)

    def test_registration_response_loss_blocks_duplicate_and_logout(self):
        self.fill_person()
        def lose_response(route):
            response = route.fetch()
            self.assertEqual(response.status, 201)
            route.abort('failed')
        self.page.route(self.api + '/participants', lose_response)
        self.page.get_by_role('button', name='저장', exact=True).click()
        expect(self.page.get_by_role('button', name='대상자 목록에서 확인')).to_be_visible()
        expect(self.page.get_by_role('button', name='저장', exact=True)).to_be_disabled()
        self.page.unroute(self.api + '/participants', lose_response)
        self.page.get_by_role('button', name='대상자 목록에서 확인').click()
        expect(self.page.get_by_text(self.name, exact=True)).to_be_visible()
        self.page.get_by_role('button', name='계정 메뉴', exact=True).click()
        expect(self.page.get_by_role('region', name='로그인 계정')).to_contain_text(self.username)
        self.page.get_by_role('button', name='로그아웃').click()
        expect(self.page.get_by_role('button', name='로그인', exact=True)).to_be_visible()
        self.page.get_by_label('아이디', exact=True).fill(self.username)
        self.page.get_by_label('비밀번호', exact=True).fill('incorrect-password')
        self.page.get_by_role('button', name='로그인', exact=True).click()
        expect(self.page.get_by_text('로그인에 실패했습니다', exact=True)).to_be_visible()


    def test_pending_measurement_guards_buttons_and_browser_history(self):
        self.create_person()
        save = self.open_save()
        measurement_url = self.page.url
        self.page.once("dialog", lambda dialog: dialog.dismiss())
        self.page.get_by_role("button", name="측정 기록 확인", exact=True).click()
        self.assertEqual(self.page.url, measurement_url)
        expect(save).to_be_enabled()
        self.page.once("dialog", lambda dialog: dialog.dismiss())
        self.page.go_back()
        expect(self.page).to_have_url(measurement_url)
        expect(save).to_be_enabled()
        # A rejected traversal must preserve history as well as the pending body.
        self.page.once("dialog", lambda dialog: dialog.accept())
        self.page.go_back()
        expect(self.page.get_by_role("button", name="대상자 정보 수정")).to_be_visible()
        save = self.open_save()
        save.click()
        expect(self.page.get_by_role("button", name="저장 기록 보기", exact=True)).to_be_visible()
        self.page.get_by_role("button", name="측정 기록 확인", exact=True).click()
        expect(self.page.locator(".record-btn")).to_have_count(1)

    def assert_late_participant_save_stays_on_new_screen(self, editing):
        if editing:
            self.create_person()
            self.page.get_by_role("button", name="대상자 정보 수정").click()
            self.page.get_by_label("키", exact=False).fill("172.5")
            endpoint = self.api + "/participants/" + self.person_id
            action = "수정 내용 저장"
        else:
            self.fill_person()
            endpoint = self.api + "/participants"
            action = "저장"
        held = []
        self.page.route(endpoint, lambda route: held.append(route) if route.request.method in ("POST", "PATCH") else route.continue_())
        with self.page.expect_request(lambda request: request.url == endpoint and request.method in ("POST", "PATCH")):
            self.page.get_by_role("button", name=action, exact=True).click()
        expect(self.page.get_by_role("button", name="저장 중…", exact=True)).to_be_visible()
        self.page.once("dialog", lambda dialog: dialog.accept())
        self.page.get_by_role("button", name="뒤로" if editing else "닫기", exact=True).click()
        if editing:
            self.page.get_by_role("button", name="이전 결과 조회", exact=False).click()
            expect(self.page.get_by_role("heading", name="이전 결과", exact=True)).to_be_visible()
        else:
            expect(self.page.get_by_role("button", name="대상자 등록", exact=True)).to_be_visible()
        destination = self.page.url
        self.assertEqual(len(held), 1)
        response = held[0].fetch()
        self.assertEqual(response.status, 200 if editing else 201)
        with self.page.expect_response(lambda response: response.url == endpoint and response.request.method in ("POST", "PATCH")):
            held[0].fulfill(response=response)
        self.page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        self.assertEqual(self.page.url, destination)

    def test_late_registration_response_does_not_navigate(self):
        self.assert_late_participant_save_stays_on_new_screen(False)

    def test_late_edit_response_does_not_navigate(self):
        self.assert_late_participant_save_stays_on_new_screen(True)

    def test_result_pagination_never_downloads_graph_details(self):
        self.create_person()
        with self.engine.begin() as conn:
            for i in range(25):
                conn.execute(text("""INSERT INTO gait_measurements
                    (id,participant_id,age,keypoints_json,extractor_name,extractor_version,measured_at,created_by)
                    VALUES (:id,:pid,76,'{}'::jsonb,'synthetic','1',:at,:uid)"""),
                    {"id": uuid4(), "pid": self.person_id, "uid": self.user_id,
                     "at": datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)})
        requests = []
        completed = []
        self.page.on("request", lambda request: requests.append(request.url))
        self.page.on("requestfinished", lambda request: completed.append(request.url))
        self.page.get_by_role("button", name="이전 결과 조회", exact=False).click()
        expect(self.page.locator(".record-btn")).to_have_count(20)
        self.page.get_by_role("button", name="다음 페이지", exact=True).click()
        expect(self.page.locator(".record-btn")).to_have_count(5)
        expect(self.page.get_by_role("button", name="다음 페이지", exact=True)).to_be_disabled()
        self.page.get_by_role("button", name="이전 페이지", exact=True).click()
        expect(self.page.locator(".record-btn")).to_have_count(20)
        # Ignore transfers aborted by the development StrictMode mount check.
        # Keep monitoring every attempt below for unwanted detail requests.
        queries = [parse_qs(urlsplit(url).query) for url in completed if "/measurement-summaries?" in url]
        self.assertEqual(queries, [
            {"limit": ["20"], "offset": ["0"]},
            {"limit": ["20"], "offset": ["20"]},
            {"limit": ["20"], "offset": ["0"]},
        ])
        self.assertFalse(any("/result" in url or "/measurements?" in url for url in requests))

    def test_zero_graph_has_finite_coordinates_in_report_and_zoom(self):
        self.create_person()
        self.open_save().click()
        expect(self.page.get_by_role("button", name="저장 기록 보기", exact=True)).to_be_visible()
        mid = str(self.measurement_rows()[0].id)
        graph = json.dumps({"schema_version": "gait-cycle-degrees-1", "left": [0] * 51, "right": [0] * 51})
        with self.engine.begin() as conn:
            conn.execute(text("""INSERT INTO gait_analysis_results
                (id,measurement_id,gdi_score,hip_graph_json,knee_graph_json,ankle_graph_json,model_version,analyzed_at)
                VALUES (:id,:mid,0,CAST(:graph AS jsonb),CAST(:graph AS jsonb),CAST(:graph AS jsonb),'synthetic-zero',now())"""),
                {"id": uuid4(), "mid": mid, "graph": graph})
        self.page.get_by_role("button", name="저장 기록 보기", exact=True).click()
        graphs = self.page.locator('svg[aria-label*="관절 각도 그래프"]')
        expect(graphs).to_have_count(3)
        def assert_finite():
            paths = self.page.locator('svg[aria-label*="관절 각도 그래프"] path')
            values = paths.evaluate_all("(elements) => elements.map(el => el.getAttribute('d'))")
            self.assertTrue(values)
            self.assertTrue(all(value and "NaN" not in value and "Infinity" not in value for value in values))
        assert_finite()
        self.page.get_by_role("button", name="확대", exact=True).first.click()
        expect(graphs).to_have_count(1)
        assert_finite()
