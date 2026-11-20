"""New design screens against real API/PostgreSQL, isolated by the existing harness."""
import json
import unittest
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
        self.page.goto(self.ui)
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

    def test_video_picker_preview_invalid_media_and_back(self):
        self.create_person()
        self.page.get_by_role('button', name='새 보행 측정', exact=False).click()
        expect(self.page.get_by_role('heading', name='분석할 보행 영상을 선택해 주세요')).to_be_visible()
        expect(self.page.get_by_role('button', name='분석 시작', exact=True)).to_be_disabled()
        with self.page.expect_file_chooser() as chooser:
            self.page.get_by_role('button', name='기기에서 영상 선택', exact=False).click()
        chooser.value.set_files({'name': 'invalid-video.mp4', 'mimeType': 'video/mp4', 'buffer': b'not a video'})
        expect(self.page.get_by_role('heading', name='선택한 영상')).to_be_visible()
        expect(self.page.get_by_text('invalid-video.mp4', exact=True)).to_be_visible()
        expect(self.page.get_by_role('alert')).to_contain_text('재생할 수 없습니다')
        expect(self.page.get_by_role('button', name='분석 시작', exact=True)).to_be_disabled()
        with self.page.expect_file_chooser() as chooser:
            self.page.get_by_role('button', name='다른 영상 선택').click()
        chooser.value.set_files([])
        expect(self.page.get_by_text('invalid-video.mp4', exact=True)).to_be_visible()
        self.assertEqual(self.measurement_rows(), [])
        self.page.get_by_role('button', name='뒤로', exact=True).click()
        expect(self.page.get_by_role('button', name='대상자 정보 수정')).to_be_visible()

    def test_analysis_start_prepares_then_saves_without_second_click(self):
        # Stub only local pose preparation; exercise the real screen, request contract,
        # API and DB. This is not a test of MediaPipe accuracy or AI inference.
        fixture = json.loads((support.ROOT / 'contracts/measurement-1/synthetic.request.json').read_text(encoding='utf-8'))['body']
        module = 'const fixture = ' + json.dumps(fixture) + ''';
export async function prepareRecording(file, mount, input, signal, progress) {
  progress(30, 30);
  await new Promise(resolve => setTimeout(resolve, 300));
  if (signal.aborted) throw new DOMException('Cancelled', 'AbortError');
  const coords = structuredClone(fixture.keypoints_json);
  coords.subject.id = input.subjectId;
  coords.recording_id = input.recordingId;
  coords.capture.recorded_at = input.analysisStartedAt;
  const {schema_version, ...metadata} = fixture.processing_metadata;
  return {coords, metadata, requestId: input.requestId};
}
'''
        self.page.route('**/src/pose/prepare-recording.ts', lambda route: route.fulfill(status=200, content_type='application/javascript', body=module))
        self.page.reload()
        self.login()
        self.create_person()
        self.page.get_by_role('button', name='새 보행 측정', exact=False).click()
        output = support.ROOT / 'mobile/local-output'
        output.mkdir(parents=True, exist_ok=True)
        expect(self.page.get_by_role('heading', name='분석할 보행 영상을 선택해 주세요')).to_be_visible()
        self.page.screenshot(path=str(output / 'measurement-picker.png'), full_page=True)
        # Browser-generated rectangle video: no personal files or images.
        clip = self.page.evaluate('''async () => {
          const canvas = document.createElement('canvas'); canvas.width=160; canvas.height=120;
          canvas.getContext('2d').fillRect(0,0,160,120);
          const stream=canvas.captureStream(10); const chunks=[];
          const recorder=new MediaRecorder(stream, {mimeType:'video/webm'});
          const done=new Promise(resolve => {recorder.onstop=resolve;});
          recorder.ondataavailable=e=>chunks.push(e.data); recorder.start();
          await new Promise(resolve=>setTimeout(resolve,300)); recorder.stop(); await done;
          stream.getTracks().forEach(track=>track.stop());
          return Array.from(new Uint8Array(await new Blob(chunks).arrayBuffer()));
        }''')
        self.page.get_by_label('보행 영상 선택', exact=True).set_input_files({'name':'synthetic.webm','mimeType':'video/webm','buffer':bytes(clip)})
        expect(self.page.get_by_role('heading', name='선택한 영상')).to_be_visible()
        self.page.screenshot(path=str(output / 'measurement-preview.png'), full_page=True)
        self.page.get_by_role('button', name='분석 시작', exact=True).click()
        expect(self.page.get_by_role('heading', name='측정 기록이 저장되었습니다')).to_be_visible(timeout=15000)
        self.assertEqual(len(self.measurement_rows()), 1)
        self.page.screenshot(path=str(output / 'measurement-saved.png'), full_page=True)

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
