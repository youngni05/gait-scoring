"""Loopback-only demo. Uploaded videos are temporary; no external service is used."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json
import mimetypes
import secrets
import threading
import time
import webbrowser
from urllib.parse import urlparse
import engine

HERE = Path(__file__).resolve().parent
UPLOADS = Path('D:/gait-research-data/D_P/meeting-demo-20261106/uploads')
MAX_BYTES = 250 * 1024 * 1024
SAMPLES = {'normal': engine.ROOT/'영상자료/검증용/정상 보행.mp4',
           'simulation': engine.ROOT/'영상자료/검증용/비정상 보행.mp4'}
TOKEN = secrets.token_urlsafe(24)
LOCK = threading.Lock()
JOB = None

def update(**values):
    with LOCK:
        JOB.update(values)

def work(path, temporary):
    try:
        x, review = engine.extract(path, update)
        update(stage='review', message='관절점 확인 후 점수를 계산한다.', done=1, total=1, x=x, review=review)
    except Exception as e:
        update(stage='error', message=str(e))
    finally:
        if temporary:
            path.unlink(missing_ok=True)

def score():
    try:
        result = engine.predict(JOB['x'], update)
        update(stage='complete', result=result, message='연구용 예측값 계산 완료')
    except Exception as e:
        update(stage='error', message=str(e))

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, status, value, content='application/json; charset=utf-8'):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode() if not isinstance(value, bytes) else value
        self.send_response(status)
        self.send_header('Content-Type', content)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self' blob:; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def same_host(self):
        return self.headers.get('Host') in (f'PRIVATE_HOST:{self.server.server_port}', f'localhost:{self.server.server_port}')

    def do_GET(self):
        if not self.same_host():
            self.reply(403, {'error': '로컬 주소만 지원한다.'}); return
        route = urlparse(self.path).path
        if route == '/api/config':
            self.reply(200, {'token': TOKEN, 'modelCount': 109, 'comparisonCount': 6}); return
        if route == '/api/job':
            if self.headers.get('X-Demo-Token') != TOKEN:
                self.reply(403, {'error': '화면을 새로고침한다.'}); return
            with LOCK:
                value = {k:v for k,v in (JOB or {'stage':'idle'}).items() if k != 'x'}
            self.reply(200, value); return
        if route.startswith('/sample/'):
            path = SAMPLES.get(route.rsplit('/', 1)[-1])
            if path and path.is_file():
                # Range requests keep video seeking responsive.
                size = path.stat().st_size
                start, end = 0, size-1
                request_range = self.headers.get('Range', '')
                if request_range.startswith('bytes='):
                    try:
                        lo, hi = request_range[6:].split('-', 1)
                        start, end = int(lo or 0), min(int(hi) if hi else size-1, size-1)
                        if not 0 <= start <= end < size: raise ValueError()
                    except ValueError:
                        self.reply(416, {'error':'범위를 읽지 못했다.'}); return
                self.send_response(206 if request_range else 200)
                self.send_header('Content-Type','video/mp4')
                self.send_header('Accept-Ranges','bytes')
                self.send_header('Content-Length',str(end-start+1))
                if request_range: self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                self.end_headers()
                try:
                    with path.open('rb') as f:
                        f.seek(start)
                        remaining=end-start+1
                        while remaining:
                            chunk=f.read(min(1024*1024, remaining))
                            if not chunk: break
                            self.wfile.write(chunk); remaining-=len(chunk)
                except (BrokenPipeError, ConnectionResetError): pass
                return
        allowed = {'/':'index.html', '/app.js':'app.js', '/style.css':'style.css', '/meeting.html':'meeting.html'}
        name = allowed.get(route)
        if name and (HERE/'web'/name).is_file():
            self.reply(200, (HERE/'web'/name).read_bytes(), (mimetypes.guess_type(name)[0] or 'text/plain')+'; charset=utf-8')
        else: self.reply(404, {'error':'없는 페이지다.'})

    def do_POST(self):
        global JOB
        self.connection.settimeout(60)
        origin = self.headers.get('Origin')
        if not self.same_host() or self.headers.get('X-Demo-Token') != TOKEN or (origin and origin != 'http://'+self.headers.get('Host','')):
            self.reply(403, {'error':'이 PC의 시연 화면에서 요청한다.'}); return
        route = urlparse(self.path).path
        if route == '/api/score':
            with LOCK:
                if not JOB or JOB['stage'] != 'review':
                    self.reply(409, {'error':'먼저 관절점 추출을 완료한다.'}); return
                JOB.update(stage='score', done=0, total=109)
            threading.Thread(target=score, daemon=True).start()
            self.reply(202, {'ok': True}); return
        if route not in ('/api/upload', '/api/sample/normal', '/api/sample/simulation'):
            self.reply(404, {'error':'없는 요청이다.'}); return
        with LOCK:
            if JOB and JOB['stage'] in ('upload','pose','score'):
                self.reply(409, {'error':'현재 분석이 끝난 뒤 다시 선택한다.'}); return
            JOB = dict(stage='upload', done=0, total=1, message='파일 준비 중', created=time.time())
        path = None
        temporary = route == '/api/upload'
        try:
            if temporary:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_BYTES:
                    raise ValueError('0보다 크고 250MB 이하인 영상을 선택한다.')
                UPLOADS.mkdir(parents=True, exist_ok=True)
                path = UPLOADS/(secrets.token_hex(16)+'.mp4')
                with path.open('xb') as f:
                    remaining=length
                    while remaining:
                        chunk=self.rfile.read(min(1024*1024, remaining))
                        if not chunk: raise ValueError('영상 전송이 중단되었다.')
                        f.write(chunk); remaining-=len(chunk)
            else:
                path = SAMPLES[route.rsplit('/',1)[-1]]
            update(stage='pose', message='MediaPipe 준비 중')
            threading.Thread(target=work, args=(path,temporary), daemon=True).start()
            self.reply(202, {'ok':True})
        except Exception as e:
            if temporary and path: path.unlink(missing_ok=True)
            update(stage='error', message=str(e))
            self.reply(400, {'error':str(e)})

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--open', action='store_true')
    args=parser.parse_args()
    engine.check_assets()
    server=ThreadingHTTPServer(('PRIVATE_HOST',args.port), Handler)
    url=f'http://PRIVATE_HOST:{server.server_port}/'
    print('GDI local demo: '+url, flush=True)
    print('Close this window or press Ctrl+C to stop.', flush=True)
    if args.open: webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
