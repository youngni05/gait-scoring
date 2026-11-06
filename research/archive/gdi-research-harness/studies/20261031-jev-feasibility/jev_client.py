"""Bounded TypeSafe requests. Credentials stay in memory; errors never log bodies."""
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ENDPOINT = 'https://api.typesafe.ai/v1/systemone'


def api_key():
    key = os.environ.get('TYPESAFE_API_KEY', '')
    if not key:
        path = ROOT / 'ai-measurement-review/app/packages/web/.env.local'
        if path.exists():
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                name, sep, value = line.partition('=')
                if sep and name.strip() == 'TYPESAFE_API_KEY':
                    key = value.strip().strip('\"\'')
    if not key or key.startswith(('본인_', 'your_')):
        raise RuntimeError('missing_api_key')
    return key


def call(payload):
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf8')
    if len(data) > 80000:
        raise ValueError('request_too_large')
    req = urllib.request.Request(ENDPOINT, data=data, method='POST', headers={
        'Authorization': 'Bearer ' + api_key(), 'Content-Type': 'application/json'})
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=40) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'http_{error.code}') from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError('network_or_timeout') from None
    if body.get('model') != payload['model']:
        raise RuntimeError('unexpected_model')
    answers = body.get('answers', {})
    if set(answers) != set(payload['questions']):
        raise RuntimeError('unexpected_questions')
    for name, question in payload['questions'].items():
        answer = answers[name]
        if answer.get('type') != 'choice' or answer.get('choice') not in question['criteria']:
            raise RuntimeError('invalid_choice')
        p = answer.get('probabilities', {})
        if set(p) != set(question['criteria']) or any(not isinstance(v, (float, int)) or not 0 <= v <= 1 for v in p.values()):
            raise RuntimeError('invalid_probabilities')
        if abs(sum(p.values()) - 1) > .01:
            raise RuntimeError('invalid_probability_sum')
        if not isinstance(answer.get('confidence'), (float, int)) or not 0 <= answer['confidence'] <= 1:
            raise RuntimeError('invalid_confidence')
    return {'model': body['model'], 'answers': answers, 'usage': body.get('usage', {}),
            'elapsed_seconds': time.monotonic() - start, 'request_bytes': len(data)}


def probe():
    # Synthetic input only: connection/schema check, not gait accuracy evidence.
    payload = {'model': 'jev-1.14.0', 'state': {'observation': 'No motion was observed because all frames are missing.'},
               'questions': {'quality': {'type': 'choice', 'instructions': 'Are there usable motion observations?',
                             'criteria': {'usable': 'Motion observations are present.', 'missing': 'No motion observations are present.'}}}}
    result = {'synthetic_only': True, 'endpoint': ENDPOINT}
    try:
        result.update(call(payload))
        result['status'] = 'connected'
    except RuntimeError as error:
        result.update(status='blocked', reason=str(error))
    (HERE / 'connection.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    probe()
