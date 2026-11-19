"""Read a request file or stdin and write strict JSON to a file or stdout."""
import argparse
import json
import sys
from pathlib import Path

from core import analyze
from schema import LIMITS, RequestError, parse_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="Request JSON path, or - for stdin")
    parser.add_argument("output", nargs="?", help="Optional output JSON path")
    args = parser.parse_args()
    try:
        with (Path(args.request).open("rb") if args.request != "-" else sys.stdin.buffer) as stream:
            data = stream.read(LIMITS["max_body_bytes"] + 1)
        if len(data) > LIMITS["max_body_bytes"]:
            raise RequestError("Request body exceeds service limit", code="request_too_large", status=413)
        result, status = analyze(parse_json(data)), 0
    except RequestError as error:
        result, status = error.response(), 2
    text = json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return status


if __name__ == "__main__":
    sys.exit(main())
