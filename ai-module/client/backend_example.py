"""Send an existing contract request from a backend process to the local analysis API."""
import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _post(request, endpoint, base_url, timeout):
    body = json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8")
    operation = Request(base_url.rstrip("/") + endpoint, data=body,
                        headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
    with urlopen(operation, timeout=timeout) as response:
        return json.load(response)


def analyze(request, base_url="http://127.0.0.1:8877", timeout=30):
    return _post(request, "/v1/analyze", base_url, timeout)


def compare_history(request, base_url="http://127.0.0.1:8877", timeout=30):
    return _post(request, "/v1/compare-history", base_url, timeout)


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8877")
    parser.add_argument("--operation", choices=("analyze", "compare-history"), default="analyze")
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8-sig"))
    try:
        operation = compare_history if args.operation == "compare-history" else analyze
        result = operation(request, args.base_url)
    except HTTPError as error:
        print(json.dumps({"http_status": error.code, "response": json.load(error)}, ensure_ascii=False))
        raise SystemExit(1)
    except URLError as error:
        print("Local API connection failed: " + str(error.reason))
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
