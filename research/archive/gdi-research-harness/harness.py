"""One bounded, read-only GDI baseline audit. No AI/API/scheduler calls."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    """Replace a complete file atomically; leave the previous file on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


@contextmanager
def exclusive(root):
    """OS releases this lock if the process crashes; the file itself may remain."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "worker.lock").open("a+b") as f:
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise RuntimeError("다른 작업이 실행 중이므로 중복 실행하지 않는다.") from e
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-9)


def inspect_sources(config):
    blobs = {}
    for key, item in config["sources"].items():
        data = Path(item["path"]).read_bytes()
        require(digest(data) == item["sha256"], f"원본 지문 불일치: {key}. 원본 변경 여부를 검토해야 한다.")
        blobs[key] = data
    js = lambda key: json.loads(blobs[key])
    rows = lambda key: list(csv.DictReader(io.StringIO(blobs[key].decode("utf-8-sig"))))
    original = js("prepared")["records"]
    predictions, audited = rows("predictions"), rows("audit_csv")
    pilot, summary = js("pilot_result"), js("audit_summary")
    require(js("pilot_verification")["status"] == "passed", "기존 실험 검증 기록 실패")
    require(js("audit_verification")["status"] == "passed", "기존 제외 분석 검증 기록 실패")
    expected = config["expected"]
    require(len(original) == len(audited) == expected["candidate_records"], "후보 기록 수 불일치")
    originals = {x["video_id"]: x for x in original}
    audits = {x["video_id"]: x for x in audited}
    require(len(originals) == len(original) and len(audits) == len(audited), "영상 ID 중복")
    require(originals.keys() == audits.keys(), "원래 후보와 제외 목록의 영상 ID 불일치")
    for vid, row in audits.items():
        src = originals[vid]
        require(row["patient_id"] == src["Patient_ID"] and row["status"] == src["status"], "사람 또는 제외 상태 불일치")
        require(close(row["gdi_left"], src["gdi"]["L"]) and close(row["gdi_right"], src["gdi"]["R"]), "원래 GDI 정답 불일치")
        require(int(row["best_run"]) == src.get("best_run", 0), "연속 구간 길이 불일치")
    counts = dict(Counter(r["status"] for r in audited))
    require(counts == summary["status_counts"], "제외 요약의 상태별 수 불일치")
    included = {v for v, r in audits.items() if r["status"] == "included"}
    require(len(predictions) == len(included) == expected["included_records"], "예측/채택 기록 수 불일치")
    require(len({r["video_id"] for r in predictions}) == len(predictions), "예측 ID 중복")
    require({r["video_id"] for r in predictions} == included, "채택 목록과 예측 목록 불일치")
    splits, people_splits = defaultdict(list), defaultdict(set)
    for row in predictions:
        require(row["split"] in expected["splits"], "알 수 없는 자료 분할")
        src = originals[row["video_id"]]
        require(src["Patient_ID"] == row["patient_id"], "예측의 사람 ID 불일치")
        for side in ("L", "R"):
            require(close(row["actual_" + side], src["gdi"][side]), "예측 파일 정답 불일치")
        splits[row["split"]].append(row)
        people_splits[row["patient_id"]].add(row["split"])
    require(all(len(s) == 1 for s in people_splits.values()), "같은 사람이 여러 자료 분할에 존재")
    train_mean = {side: statistics.mean(float(r["actual_" + side]) for r in splits["train"]) for side in ("L", "R")}
    for row in predictions:
        require(all(close(row["baseline_" + s], train_mean[s]) for s in train_mean), "평균 기준값이 학습 자료의 평균과 다름")
    metrics = {}
    for split, shape in expected["splits"].items():
        rr = splits[split]
        require(len(rr) == shape["records"] and len({r["patient_id"] for r in rr}) == shape["people"], "분할별 기록/사람 수 불일치")
        metrics[split] = dict(shape)
        for method, recorded in (("ridge", "ridge"), ("baseline", "training_mean_baseline")):
            errors = defaultdict(list)
            for row in rr:
                e = [abs(float(row[method + "_" + s]) - float(row["actual_" + s])) for s in ("L", "R")]
                require(all(math.isfinite(x) for x in e), "비정상 예측값")
                errors[row["patient_id"]].append(statistics.mean(e))
            mae = statistics.mean(statistics.mean(x) for x in errors.values())
            require(close(mae, pilot["metrics"][split][recorded]["person_macro_MAE"]), "저장된 오차와 재계산 오차 불일치")
            metrics[split][method + "_person_mae"] = mae
    all_people = {r["patient_id"] for r in audited}
    retained_people = {audits[v]["patient_id"] for v in included}
    require(len(all_people) == expected["candidate_people"] == summary["people"], "후보 사람 수 불일치")
    require(len(retained_people) == expected["included_people"] == pilot["prepared_people"], "채택 사람 수 불일치")
    require(len(all_people - retained_people) == summary["people_excluded_entirely"], "완전히 제외된 사람 수 불일치")
    lengths = {}
    for length, saved in summary["length_screen_only"].items():
        candidates = [r for r in audited if int(r["best_run"]) >= int(length)]
        lengths[length] = {"records": len(candidates), "people": len({r["patient_id"] for r in candidates})}
        require(all(lengths[length][k] == saved[k] for k in ("records", "people")), "짧은 구간의 길이 선별 수 불일치")
    av = js("audit_verification")
    require(digest(blobs["audit_csv"]) == av["csv_sha256"], "제외 CSV의 기존 검증 지문 불일치")
    require(digest(blobs["audit_summary"]) == av["summary_sha256"], "제외 요약의 기존 검증 지문 불일치")
    require(digest(blobs["prepared"]) == summary["source_prepared_sha256"], "원래 전처리 지문 불일치")
    # Re-read every referenced file after processing to detect concurrent edits.
    for key, item in config["sources"].items():
        require(digest(Path(item["path"]).read_bytes()) == digest(blobs[key]), "검사 도중 원본 변경: " + key)
    return {"status": "passed", "candidate_records": len(audited), "candidate_people": len(all_people),
            "included_records": len(included), "included_people": len(retained_people),
            "excluded_records": len(audited) - len(included), "never_included_people": len(all_people - retained_people),
            "status_counts": counts, "metrics": metrics, "length_screen_only": lengths,
            "source_files_unchanged": True, "fresh_training": False, "raw_archive_rechecked": False}


def report(result, run_id):
    test = result["metrics"]["test"]
    return f"""# 보행 점수 연구 — 현재 보고

첫 작업을 완료했다. 기존 실험의 예측값과 제외 자료 기록을 다시 읽고, 비교의 출발점을 고정했다.
새로운 모델을 학습하거나 보행 점수 기준을 바꾼 작업은 아니다.

## 왜 이 작업부터 했는가

앞으로 오차가 줄었는지 판단하려면 기존 숫자를 같은 방식으로 다시 계산할 수 있어야 한다.
또한 많은 자료가 제외된 상태이므로, 성능과 함께 실제로 사용할 수 있는 자료 수를 살펴야 한다.
그래서 새 학습보다 기존 기록 검사를 먼저 실행했다. 새로운 실험은 이 기록과 나란히 비교한다.

## 무엇을 어떻게 확인했는가

기존 원본 기록 → 사람·영상·정답 대조 → 학습/검증/평가 사람 중복 검사 → 오차 재계산 → 원본 지문 재확인 순서로 처리했다.
오차는 좌우 각각의 예측값과 정답의 차이를 절댓값으로 구해 평균하고, 사람마다 여러 기록을 평균한 뒤, 사람 간 평균을 구했다.
평균값만 답하는 비교 모델은 학습 자료의 평균만 사용했는지 확인했다. 별도 프로그램 설치나 유료 AI 호출은 없었다.

| 항목 | 이번에 다시 확인한 값 |
|---|---:|
| 전체 후보 | {result['candidate_records']:,}건 / {result['candidate_people']:,}명 |
| 기존 실험에서 사용 | {result['included_records']:,}건 / {result['included_people']:,}명 |
| 제외 기록 | {result['excluded_records']:,}건 |
| 모든 기록이 제외된 사람 | {result['never_included_people']:,}명 |
| 평가 자료 | {test['records']}건 / {test['people']}명 |
| 기존 모델의 사람별 평균 절대 오차 | {test['ridge_person_mae']:.6f} |
| 학습 평균만 답한 모델의 같은 오차 | {test['baseline_person_mae']:.6f} |

오차는 작을수록 좋다. 평균값 모델 대비 차이는 {test['baseline_person_mae'] - test['ridge_person_mae']:.6f}이다.
이 차이는 기존 예비 실험에서 확인한 값이며, 논문의 다른 모델보다 우수하다는 뜻은 아니다.
참조한 7개 원본 파일의 내용 지문이 실행 전후 같았다.

## 다음 실험으로 이어지는 관찰

"연속 243프레임" 조건에서는 대부분의 기록이 제외됐다.
길이 조건만 124프레임으로 바꾸어 세면 {result['length_screen_only']['124']['records']:,}건 / {result['length_screen_only']['124']['people']:,}명이 후보가 된다.
이는 사용 가능성이 있는 후보 수다. 전처리 적합성·모델 오차·사람 추적 정확성을 확인한 수가 아니다.
조건을 짧게 바꾸는 것만으로 연구의 새로움이나 성능 개선이 입증되지 않는다.

## 어디까지 확인했는가

이번에는 저장된 표와 예측값을 재계산했다. 원본 영상·전체 압축파일을 다시 읽거나 모델을 다시 학습하지 않았다.
실제 가정 촬영 영상, 노인 보행, 임상적 유용성은 미검증이다.
이미 결과를 본 평가 자료 38명은 이후 후보를 고르는 기준으로 반복 사용하지 않아야 한다.
새 실험에서는 후보 선택에 쓸 자료와 마지막 평가에 쓸 자료를 먼저 구분해야 한다.

## 자동 실행과 다음 순서

현재 보고는 첫 기록 검사에 대한 결과다. 후속 진행은 [두 번째 작업 보고](jobs/development-baseline/REPORT.md)를 확인한다.
Pro 전용 로그인·확인 도구는 [로그인 안내](ACCOUNT_SETUP.md)에 정리했다. 최신 연결 상태는 `account-status.json`을 확인한다. 앱 계정 교대 후 유지 검증은 남아 있다.
예약 실행, 자동 AI 판단, 새 개선 후보 학습은 아직 연결하지 않았다.
첫 개선 실험의 구체적인 비교 조건을 고정하고 AI 실행 도구를 연결하는 일이 남아 있다.
검토할 실험 후보와 미확정 사항은 [작업 목록](backlog.json)에 보관했다.

실행 번호: `{run_id}`. [결과 수치](runs/{run_id}/result.json), [시작 전 의도](runs/{run_id}/plan.json), [완료 기록](runs/{run_id}/receipt.json).
실행 전체 이력은 `runs/`에 남는다. 실패하면 이전 성공 보고를 대신 보여주지 않고 현재 오류를 표시한다.
"""


def run(root=ROOT, retry_failed=False):
    root = Path(root)
    with exclusive(root):
        config = load(root / "config.json")
        handler, renderer = inspect_sources, report
        code = Path(__file__).read_bytes()
        if config.get("job_type", "baseline-audit") == "development-baseline":
            import development_job
            handler = lambda value: development_job.inspect(value, root, folder)
            renderer = development_job.report
            code += Path(development_job.__file__).read_bytes()
        elif config.get("job_type") == "weighted-ridge":
            import weighted_ridge_job
            handler = lambda value: weighted_ridge_job.inspect(value, root, folder)
            renderer = weighted_ridge_job.report
            code += Path(weighted_ridge_job.__file__).read_bytes()
        elif config.get("job_type") == "huber":
            import huber_job
            handler = lambda value: huber_job.inspect(value, root, folder)
            renderer = huber_job.report
            code += Path(huber_job.__file__).read_bytes()
        elif config.get("job_type", "baseline-audit") != "baseline-audit":
            raise ValueError("등록되지 않은 작업 종류")
        code_hash = digest(code)
        fingerprint = digest((json.dumps(config, sort_keys=True) + code_hash).encode())
        runs = root / "runs"
        runs.mkdir(exist_ok=True)
        previous = []
        for path in sorted(runs.iterdir()):
            if not path.is_dir() or not (path / "plan.json").exists():
                continue
            if not (path / "receipt.json").exists():
                save(path / "receipt.json", {"status": "interrupted", "ended": now(), "reason": "완료 기록 없이 중단됨. 읽기 전용 작업을 처음부터 재검사한다."})
            if load(path / "plan.json")["fingerprint"] == fingerprint:
                previous.append(path)
        if previous:
            last = previous[-1]
            receipt = load(last / "receipt.json")
            if receipt["status"] == "completed":
                # A completed audit is reused only while both evidence and output are intact.
                for key, src in config["sources"].items():
                    require(digest(Path(src["path"]).read_bytes()) == src["sha256"], "원본 변경으로 기존 결과를 재사용할 수 없음: " + key)
                require(digest((last / "result.json").read_bytes()) == receipt["result_sha256"], "완료 결과 파일이 변경됨. 검토 필요")
                result = load(last / "result.json")
                for artifact in result.get("artifacts", []):
                    require(digest((root / artifact["path"]).read_bytes()) == artifact["sha256"], "작업 산출물 변경. 검토 필요")
                save(root / "REPORT.md", renderer(result, last.name))
                save(root / "state.json", {"status": "completed", "run_id": last.name, "fingerprint": fingerprint, "reused": True})
                return {"status": "reused", "run_id": last.name}
            if receipt["status"] == "failed" and not retry_failed:
                raise RuntimeError("같은 작업이 실패했다. 원인을 검토하고 --retry-failed로 명시적으로 재시도한다.")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:6]
        folder = runs / run_id
        plan = {"fingerprint": fingerprint, "started": now(), "code_sha256": code_hash,
                "question": config["question"], "why_first": config["why_first"], "config": config,
                "method": config.get("method", "기존 표 대조 및 저장 예측의 사람별 오차 재계산. 학습 없음."),
                "success_scope": config.get("success_scope", "기존 기록과 계산의 일치. 모델 정확도 개선 또는 임상 검증이 아님.")}
        save(folder / "plan.json", plan)
        save(root / "state.json", {"status": "running", "run_id": run_id, "fingerprint": fingerprint})
        try:
            result = handler(config)
            save(folder / "result.json", result)
            save(folder / "receipt.json", {"status": "completed", "ended": now(), "result_sha256": digest((folder / "result.json").read_bytes())})
            save(root / "REPORT.md", renderer(result, run_id))
            save(root / "state.json", {"status": "completed", "run_id": run_id, "fingerprint": fingerprint, "reused": False})
        except BaseException as e:
            save(folder / "receipt.json", {"status": "failed", "ended": now(), "error": type(e).__name__ + ": " + str(e)})
            save(root / "state.json", {"status": "failed", "run_id": run_id, "fingerprint": fingerprint})
            raise
        return {"status": "completed", "run_id": run_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--job", choices=["baseline-audit", "development-baseline", "weighted-ridge", "huber"], default="baseline-audit")
    args = parser.parse_args()
    job_root = ROOT if args.job == "baseline-audit" else ROOT / "jobs" / args.job
    try:
        if job_root == ROOT:
            outcome = run(ROOT, retry_failed=args.retry_failed)
        else:
            with exclusive(ROOT):
                outcome = run(job_root, retry_failed=args.retry_failed)
        print(json.dumps(outcome, ensure_ascii=False))
    except Exception as e:
        # Include preflight/reuse failures in the current status/report as well.
        # Do not replace the other worker's report on a lock contention.
        try:
            with exclusive(ROOT):
                save(job_root / "events" / (uuid.uuid4().hex + ".json"), {"status": "needs_review", "at": now(), "error": str(e)})
                save(job_root / "state.json", {"status": "needs_review", "at": now(), "error": str(e)})
                save(job_root / "REPORT.md", "# 현재 작업에 확인이 필요하다\n\n" + str(e) + "\n\n이전 실행 증거는 runs/에 보존돼 있다.\n")
        except RuntimeError:
            pass
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
