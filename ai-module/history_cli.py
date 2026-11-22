"""Compare supplied visits without fetching or persisting private records."""
import json
from pathlib import Path
import sys
from history import compare_history, LIMITS
from schema import RequestError, parse_json


def main():
    if not 2 <= len(sys.argv) <= 3:
        print("Usage: python history_cli.py request.json [response.json]", file=sys.stderr)
        return 2
    try:
        with Path(sys.argv[1]).open("rb") as stream:
            body = stream.read(LIMITS["max_body_bytes"]+1)
        if len(body) > LIMITS["max_body_bytes"]:
            raise RequestError("Request body exceeds service limit", code="request_too_large", status=413)
        result = compare_history(parse_json(body))
        code = 0
    except RequestError as error:
        result, code = error.response(), 2
    output = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)+"\n"
    if len(sys.argv) == 3:
        Path(sys.argv[2]).write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
