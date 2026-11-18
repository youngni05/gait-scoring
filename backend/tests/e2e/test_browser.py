"""Chromium UI -> actual API -> isolated PostgreSQL schema. Synthetic data only.

Run from backend: python -m unittest discover -s tests/e2e -v
Requires GAIT_E2E_DB_WRITES=1 and a loopback PostgreSQL *_e2e_ci database.
"""
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.error import URLError
from urllib.request import urlopen
from urllib.parse import urlparse
from uuid import uuid4

import jwt
from alembic import command
from alembic.config import Config
from playwright.sync_api import sync_playwright, expect
from pwdlib import PasswordHash
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

BACKEND = Path(__file__).resolve().parents[2]
ROOT = BACKEND.parent


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


class BrowserIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        url = make_url(os.environ['DATABASE_URL'])
        if (os.getenv('GAIT_E2E_DB_WRITES') != '1' or not url.drivername.startswith('postgresql')
                or not (url.database or '').endswith('_e2e_ci')
                or url.host not in ('localhost', '127.0.0.1', '::1')):
            raise RuntimeError('Requires explicit consent and a loopback disposable *_e2e_ci database')
        cls.admin = create_engine(url)
        cls.addClassCleanup(cls.admin.dispose)
        cls.schema = 'browser_test_' + uuid4().hex
        with cls.admin.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA "{cls.schema}"'))
        cls.addClassCleanup(cls.drop_schema)
        scoped = url.update_query_dict({'options': f'-csearch_path={cls.schema}'}).render_as_string(hide_password=False)
        cls.engine = create_engine(scoped)
        cls.addClassCleanup(cls.engine.dispose)
        previous = os.environ['DATABASE_URL']
        try:
            os.environ['DATABASE_URL'] = scoped
            command.upgrade(Config(str(BACKEND / 'alembic.ini')), 'head')
            command.check(Config(str(BACKEND / 'alembic.ini')))
        finally:
            os.environ['DATABASE_URL'] = previous
        cls.user_id = uuid4()
        cls.username = 'browser-' + uuid4().hex
        cls.password = secrets.token_urlsafe(24)
        cls.secret = secrets.token_urlsafe(40)
        with cls.engine.begin() as conn:
            conn.execute(text('INSERT INTO users (id,login_id,password_hash) VALUES (:id,:login,:hash)'),
                         {'id': cls.user_id, 'login': cls.username, 'hash': PasswordHash.recommended().hash(cls.password)})
        api_port, ui_port = free_port(), free_port()
        while ui_port == api_port:
            ui_port = free_port()
        cls.api = f'http://127.0.0.1:{api_port}'
        cls.ui = f'http://127.0.0.1:{ui_port}'
        cls.launch([sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', str(api_port)],
                   BACKEND, {**os.environ, 'DATABASE_URL': scoped, 'JWT_SECRET_KEY': cls.secret, 'CORS_ALLOW_ORIGINS': cls.ui}, cls.api + '/db-health')
        cls.launch(['node', str(ROOT / 'node_modules/vite/bin/vite.js'), 'mobile', '--host', '127.0.0.1', '--port', str(ui_port), '--strictPort'],
                   ROOT, {**os.environ, 'VITE_E2E_MODE': 'true'}, cls.ui)
        cls.playwright = sync_playwright().start()
        cls.addClassCleanup(cls.playwright.stop)
        cls.browser = cls.playwright.chromium.launch()
        cls.addClassCleanup(cls.browser.close)

    @classmethod
    def launch(cls, args, cwd, env, url):
        log = tempfile.TemporaryFile()
        cls.addClassCleanup(log.close)
        process = subprocess.Popen(args, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        cls.addClassCleanup(stop, process)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if process.poll() is not None:
                log.seek(0)
                raise RuntimeError(log.read().decode(errors='replace'))
            try:
                with urlopen(url, timeout=1) as response:
                    if response.status == 200:
                        return
            except (URLError, TimeoutError):
                time.sleep(0.1)
        raise RuntimeError('Test server readiness timed out')

    @classmethod
    def drop_schema(cls):
        with cls.admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{cls.schema}" CASCADE'))

    def setUp(self):
        self.context = self.browser.new_context(viewport={'width': 1280, 'height': 900})
        self.addCleanup(self.context.close)
        self.page = self.context.new_page()
        self.page.set_default_timeout(15000)
        # Record only API paths/statuses, never headers, tokens or form bodies.
        self.api_events = []
        self.page.on('response', lambda response: self.api_events.append(
            (response.request.method, urlparse(response.url).path, response.status)
        ) if response.url.startswith(self.api + '/') else None)
        self.page.on('requestfailed', lambda request: self.api_events.append(
            (request.method, urlparse(request.url).path, 'network-failed')
        ) if request.url.startswith(self.api + '/') else None)
        self.page.goto(self.ui)
        self.page.get_by_label('서버 주소', exact=True).fill(self.api)
        self.page.get_by_label('아이디', exact=True).fill(self.username)
        self.page.get_by_label('비밀번호', exact=True).fill(self.password)
        try:
            with self.page.expect_response(lambda response: response.url == self.api + '/auth/login'
                                           and response.request.method == 'POST', timeout=15000) as login:
                self.page.get_by_role('button', name='로그인', exact=True).click()
            self.assertEqual(login.value.status, 200, 'Real API login must succeed before testing the workflow')
            expect(self.page.get_by_role('button', name='새 대상자 등록', exact=True)).to_be_visible()
        except Exception:
            print('Login diagnostics:', self.api_events, self.page.get_by_role('status').all_text_contents(), flush=True)
            raise
        self.name = '합성-' + uuid4().hex[:10]

    def registration_form(self):
        self.page.get_by_role('button', name='새 대상자 등록', exact=True).click()
        form = self.page.get_by_role('region', name='대상자 등록', exact=True)
        form.get_by_label('이름', exact=True).fill(self.name)
        form.get_by_label('성별', exact=True).select_option('F')
        form.get_by_label('생년월일', exact=True).fill('1950-01-01')
        form.get_by_label('키 (cm)', exact=True).fill('170.50')
        return form

    def create_person(self):
        form = self.registration_form()
        form.get_by_role('button', name='등록 후 선택', exact=True).click()
        expect(form).not_to_be_visible()
        self.person_id = self.page.get_by_label('대상자', exact=True).input_value()
        self.assertTrue(self.person_id)
        self.page.get_by_role('button', name='합성 좌표 준비', exact=True).click()
        self.save_button = self.page.get_by_role('button', name='서버 저장 / 같은 요청 재시도', exact=True)
        expect(self.save_button).to_be_enabled()

    def measurements(self):
        with self.engine.connect() as conn:
            return conn.execute(text('SELECT id,age,keypoints_json,processing_metadata FROM gait_measurements WHERE participant_id=:pid'), {'pid': self.person_id}).mappings().all()

    def assert_saved(self):
        expect(self.page.get_by_role('button', name='저장 완료', exact=True)).to_be_disabled()
        expect(self.page.get_by_role('button', name='저장 기록 보기', exact=False)).to_have_count(1)
        rows = self.measurements()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['processing_metadata']['schema_version'], 'processing-1')
        self.assertEqual(rows[0]['keypoints_json']['subject']['id'], self.person_id)
        return rows[0]

    def test_register_save_edit_and_view_actual_record(self):
        self.create_person()
        self.save_button.click()
        before = self.assert_saved()
        self.page.get_by_role('button', name='선택한 대상자 수정', exact=True).click()
        form = self.page.get_by_role('region', name='대상자 정보 수정', exact=True)
        form.get_by_label('생년월일', exact=True).fill('1960-02-29')
        form.get_by_label('키 (cm)', exact=True).fill('180.25')
        form.get_by_role('button', name='수정 저장', exact=True).click()
        expect(form).not_to_be_visible()
        expect(self.page.get_by_label('촬영 당시 키 (cm)', exact=True)).to_have_value('170.50')
        after = self.assert_saved()
        self.assertEqual(dict(before), dict(after))
        with self.engine.connect() as conn:
            row = conn.execute(text('SELECT birth_date,height_cm FROM participants WHERE id=:pid'), {'pid': self.person_id}).one()
            self.assertEqual(str(row.birth_date), '1960-02-29')
            self.assertEqual(str(row.height_cm), '180.25')
        self.page.get_by_role('button', name='저장 기록 보기', exact=False).click()
        expect(self.page.get_by_text('분석 결과는 아직 없습니다.', exact=False)).to_be_visible()
        expect(self.page.get_by_text(f'촬영 당시 만 {before["age"]}세', exact=True)).to_be_visible()

    def test_committed_response_loss_retries_same_bytes_without_duplicate(self):
        self.create_person()
        sent = []
        def lose_once(route):
            if route.request.method != 'POST':
                return route.continue_()
            sent.append((route.request.post_data, route.request.headers['idempotency-key']))
            if len(sent) == 1:
                response = route.fetch()
                self.assertEqual(response.status, 201)
                route.abort('failed')
            else:
                route.continue_()
        self.page.route(self.api + '/participants/*/measurements', lose_once)
        self.save_button.click()
        expect(self.page.get_by_text('서버 응답을 확인하지 못했습니다.', exact=False)).to_be_visible()
        self.assertEqual(len(self.measurements()), 1)
        self.save_button.click()
        self.assert_saved()
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0], sent[1])

    def test_expired_auth_renewal_preserves_pending_request(self):
        self.create_person()
        expired = jwt.encode({'sub': str(self.user_id), 'exp': datetime.now(timezone.utc) - timedelta(minutes=1)}, self.secret, algorithm='HS256')
        sent = []
        def expire_once(route):
            if route.request.method != 'POST':
                return route.continue_()
            sent.append((route.request.post_data, route.request.headers['idempotency-key']))
            headers = dict(route.request.headers)
            if len(sent) == 1:
                headers['authorization'] = 'Bearer ' + expired
            route.continue_(headers=headers)
        self.page.route(self.api + '/participants/*/measurements', expire_once)
        self.save_button.click()
        expect(self.page.get_by_text('인증이 만료되었거나 로그인 정보가 잘못되었습니다.', exact=False)).to_be_visible()
        self.assertEqual(len(self.measurements()), 0)
        self.page.get_by_label('비밀번호', exact=True).fill(self.password)
        self.page.get_by_role('button', name='같은 계정으로 인증 갱신', exact=True).click()
        expect(self.page.get_by_text('인증 갱신 완료.', exact=False)).to_be_visible()
        self.save_button.click()
        self.assert_saved()
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0], sent[1])

    def test_registration_validation_and_ambiguous_creation_block_retry(self):
        form = self.registration_form()
        form.get_by_label('이름', exact=True).fill(' ')
        form.get_by_role('button', name='등록 후 선택', exact=True).click()
        expect(form.get_by_text('이름을 1~100자로 입력하세요.', exact=True)).to_be_visible()
        form.get_by_label('이름', exact=True).fill(self.name)
        def lose_creation(route):
            if route.request.method != 'POST':
                return route.continue_()
            self.assertEqual(route.fetch().status, 201)
            route.abort('failed')
        self.page.route(self.api + '/participants', lose_creation)
        form.get_by_role('button', name='등록 후 선택', exact=True).click()
        expect(form.get_by_text('저장 결과를 확인하지 못했습니다.', exact=False)).to_be_visible()
        expect(form.get_by_role('button', name='등록 후 선택', exact=True)).to_be_disabled()
        form.get_by_role('button', name='닫고 목록 확인', exact=True).click()
        expect(self.page.get_by_label('대상자', exact=True)).to_contain_text(self.name)
        with self.engine.connect() as conn:
            count = conn.scalar(text('SELECT count(*) FROM participants WHERE name=:name'), {'name': self.name})
            self.assertEqual(count, 1)


if __name__ == '__main__':
    unittest.main()
