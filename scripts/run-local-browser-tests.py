"""Run browser integration tests using a disposable DB in the local Compose DB service.

From repository root: .venv/Scripts/python.exe scripts/run-local-browser-tests.py
Requires npm ci, npm run prepare:assets, backend requirements and Playwright Chromium.
Existing application data and .env are never modified. Credentials are never printed.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


def main():
    root = Path(__file__).resolve().parents[1]
    container = subprocess.check_output(['docker', 'compose', 'ps', '-q', 'db'], cwd=root, text=True).strip()
    if not container:
        raise RuntimeError('Start the local DB first: docker compose up -d db')
    config = json.loads(subprocess.check_output(['docker', 'inspect', container], text=True))[0]
    values = dict(item.split('=', 1) for item in config['Config']['Env'] if '=' in item)
    bindings = config['NetworkSettings']['Ports']['5432/tcp']
    binding = next(b for b in bindings if b['HostIp'] == '127.0.0.1')
    url = URL.create('postgresql+psycopg', username=values['POSTGRES_USER'], password=values['POSTGRES_PASSWORD'],
                     host='127.0.0.1', port=int(binding['HostPort']), database='postgres')
    engine = create_engine(url, isolation_level='AUTOCOMMIT')
    name = 'gait_local_' + uuid4().hex + '_e2e_ci'
    created = False
    try:
        with engine.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
        created = True
        env = dict(os.environ, DATABASE_URL=url.set(database=name).render_as_string(hide_password=False),
                   GAIT_E2E_DB_WRITES='1', PYTHONIOENCODING='utf-8')
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests/e2e', '-v'],
                                cwd=root / 'backend', env=env)
        return result.returncode
    finally:
        if created:
            with engine.connect() as conn:
                conn.execute(text(f'DROP DATABASE "{name}"'))
            print('Isolated test database removed; application database unchanged.')
        engine.dispose()


if __name__ == '__main__':
    sys.exit(main())
