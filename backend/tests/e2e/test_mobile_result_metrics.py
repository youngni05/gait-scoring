"""Render the actual mobile list and report with deterministic recording API responses."""
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import unittest
from urllib.error import URLError
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[3]


def record(name, intervals):
    steps = []
    clock = 0
    for side, duration in intervals:
        steps.append(dict(landing_side=side, start_ms=clock, stop_ms=clock+duration))
        clock += duration
    segment = dict(segment_id='walk', direction='image_right', start_ms=0, stop_ms=clock,
                   gait=dict(quality_reasons=[], steps=steps))
    tracking = dict(status='ok', transitions=[], possible_transitions=0, checked_transitions=0,
                    contact_radius_ms=75)
    report = dict(segment_id='walk', metrics=[dict(id='rhythm', evidence=dict(tracking=tracking))])
    visit = dict(id=name, subject_id='p1', captured_at='2026-01-01T00:00:00Z',
                 bridge=dict(sampling=dict(frame_count=60, missing_pose_frames=0)),
                 analysis=dict(segments=[segment], current_report=dict(segments=[report])))
    # An old non-null symmetry field must never be mistaken for a time in seconds.
    row = dict(id=name, subjectId='p1', fileName=name, measuredAt=visit['captured_at'],
               serverState='ready', symmetry=8.1, gdi=None, cadence=None, curves={}, missing={},
               ageAtCapture=None, heightAtCapture=None)
    return dict(**row, preview=visit), dict(**row, live=visit)


class MobileResultMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        cls.ui = f'http://127.0.0.1:{port}'
        log = tempfile.TemporaryFile()
        cls.addClassCleanup(log.close)
        server = subprocess.Popen(
            ['node', 'node_modules/vite/bin/vite.js', 'mobile', '--host', '127.0.0.1',
             '--port', str(port), '--strictPort'], cwd=ROOT,
            env={**os.environ, 'VITE_TEAM_API_BASE': cls.ui+'/api'},
            stdout=log, stderr=subprocess.STDOUT)
        cls.addClassCleanup(server.wait, timeout=10)
        cls.addClassCleanup(server.terminate)
        deadline = time.monotonic()+30
        while True:
            try:
                with urlopen(cls.ui, timeout=1) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError):
                pass
            if server.poll() is not None or time.monotonic() >= deadline:
                log.seek(0)
                raise RuntimeError('Vite could not serve the mobile entry point: '+log.read().decode(errors='replace'))
            time.sleep(.1)
        pw = sync_playwright().start()
        cls.addClassCleanup(pw.stop)
        cls.browser = pw.chromium.launch(channel=os.environ.get('PLAYWRIGHT_CHROMIUM_CHANNEL'))
        cls.addClassCleanup(cls.browser.close)

    def setUp(self):
        self.page = self.browser.new_page(viewport=dict(width=390, height=844))
        self.addCleanup(self.page.close)

    def test_recording_list_and_detail_agree_on_unsigned_seconds_zero_and_missing(self):
        samples = [
            ('negative', [('left', 350), ('right', 500)]*2, '0.15'),
            ('zero', [('left', 500), ('right', 500)]*2, '0.00'),
            ('one-sided', [('left', 500)]*4, None),
        ]
        summaries = []
        details = {}
        for name, intervals, _ in samples:
            summary, details[name] = record(name, intervals)
            summaries.append(summary)
        self.page.route('**/api/team/recordings/*', lambda route: route.fulfill(
            json=details[route.request.url.rsplit('/', 1)[-1]]))
        self.page.route('**/api/team/participants/p1/recording-summaries?*', lambda route: route.fulfill(
            json=dict(items=summaries, limit=20, offset=0, total=len(summaries))))
        self.page.route('**/api/participants/p1', lambda route: route.fulfill(
            json=dict(id='p1', name='합성 대상자', gender='F', birth_date=None, height_cm=None)))
        self.page.add_init_script("sessionStorage.setItem('team-access-token','synthetic-test')")
        self.page.goto(self.ui+'/#/subjects/p1/results')
        cards = self.page.locator('.record-btn')
        expect(cards).to_have_count(len(samples))
        for index, (name, _, expected) in enumerate(samples):
            with self.subTest(name=name):
                item = cards.nth(index)
                metric = item.locator('.metric-grid3 > div').filter(
                    has=self.page.get_by_text('좌우 걸음 시간 차', exact=True))
                expect(metric.locator('.v')).to_have_text(expected+'초' if expected else '—')
                item.click()
                self.page.wait_for_url('**#/subjects/p1/results/'+name)
                timing = self.page.locator('.ov-rows dt').filter(has_text='좌우 걸음 시간')
                if expected is None:
                    expect(timing.locator('small')).to_have_text('한 걸음 평균')
                    expect(self.page.locator('.ov-rows .lr .right').first).to_contain_text('—')
                else:
                    expect(timing.locator('small')).to_have_text('차이 '+expected+'초')
                self.page.go_back()

    def test_demo_keeps_its_original_symmetry_metric(self):
        self.page.goto(self.ui+'/?demo=ui#/login')
        self.page.get_by_label('아이디', exact=True).fill('demo')
        self.page.get_by_label('비밀번호', exact=True).fill('demo')
        self.page.get_by_role('button', name='로그인', exact=True).click()
        self.page.wait_for_url('**#/subjects')
        self.page.evaluate("location.hash='/subjects/s1/results'")
        item = self.page.locator('.record-btn').first
        metric = item.locator('.metric-grid3 > div').filter(has=self.page.get_by_text('좌우 대칭', exact=True))
        expect(metric.locator('.v')).to_have_text('8.1')
        item.click()
        symmetry = self.page.locator('.card').filter(has=self.page.get_by_role('button', name='계산 방식 보기'))
        expect(symmetry.get_by_text('8.1', exact=True)).to_be_visible()
        expect(symmetry.get_by_text('단위 확정 예정', exact=True)).to_be_visible()
