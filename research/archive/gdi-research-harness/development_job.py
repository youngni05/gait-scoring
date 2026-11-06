"""Reproduce the existing fixed Ridge on development data only; no new selection."""
import csv
import io
import json
import os
from pathlib import Path
import sys

from harness import digest, require


def inspect(config, root, run_folder):
    blobs = {}
    for key, item in config["sources"].items():
        blobs[key] = Path(item["path"]).read_bytes()
        require(digest(blobs[key]) == item["sha256"], "원본 지문 불일치: " + key)
    # Reuse installed experiment dependencies without changing those directories.
    sys.path.insert(0, config["dependency_path"])
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    protocol = json.loads(blobs["protocol"])
    prepared = json.loads(blobs["prepared"])
    predictions = list(csv.DictReader(io.StringIO(blobs["predictions"].decode("utf-8-sig"))))
    with np.load(io.BytesIO(blobs["features"]), allow_pickle=False) as table:
        x, y = table["X"].copy(), table["Y"].copy()
        people, videos = table["patient_ids"].copy(), table["video_ids"].copy()
    require(x.shape == (285, 512) and y.shape == (285, 2), "기존 특징/정답 크기 불일치")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "비정상 특징/정답")
    require(len(set(videos)) == len(videos), "영상 ID 중복")
    records = {r["video_id"]: r for r in prepared["records"] if r["status"] == "included"}
    require(set(videos) == set(records), "특징과 원래 채택 기록 불일치")
    for i, vid in enumerate(videos):
        row = records[vid]
        require(people[i] == row["Patient_ID"], "특징의 사람 ID 불일치")
        require(np.allclose(y[i], [row["gdi"]["L"], row["gdi"]["R"]], rtol=0, atol=1e-9), "특징 파일의 정답 불일치")
    groups = {k: set(v["patient_ids"]) for k, v in protocol["split"].items()}
    require(all(not groups[a] & groups[b] for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]), "사람 분할 중복")
    require(set(people) == set.union(*groups.values()), "분할에 없는 사람")
    masks = {k: np.isin(people, list(ids)) for k, ids in groups.items()}
    for key, shape in config["expected_splits"].items():
        require(int(masks[key].sum()) == shape["records"] and len(set(people[masks[key]])) == shape["people"], "분할 수 불일치")
    # No test rows are passed to scaler/model/metric calculation or the exported packet.
    dev = masks["train"] | masks["validation"]
    xd, yd, pd, vd = x[dev], y[dev], people[dev], videos[dev]
    split = np.where(np.isin(pd, list(groups["train"])), "train", "validation")
    require(not set(pd) & groups["test"], "개발 묶음에 기존 평가 사람 포함")
    del x, y, people, videos
    train, val = split == "train", split == "validation"
    with threadpool_limits(limits=4):
        scaler = StandardScaler().fit(xd[train])
        model = Ridge(alpha=1.0, solver="svd").fit(scaler.transform(xd[train]), yd[train])
        pv = model.predict(scaler.transform(xd[val]))
    previous = {r["video_id"]: r for r in predictions if r["split"] == "validation"}
    saved = np.array([[float(previous[v]["ridge_L"]), float(previous[v]["ridge_R"])] for v in vd[val]])
    require(np.allclose(pv, saved, rtol=0, atol=1e-6), "고정 Ridge의 기존 검증 예측을 재현하지 못함")
    def person_mae(pred):
        err = np.abs(pred - yd[val]).mean(axis=1)
        return float(np.mean([err[pd[val] == p].mean() for p in sorted(set(pd[val]))]))
    ridge_mae = person_mae(pv)
    baseline_mae = person_mae(np.broadcast_to(yd[train].mean(axis=0), yd[val].shape))
    require(abs(ridge_mae - config["expected_validation_mae"]) < 1e-9, "기존 검증 오차 재현 실패")
    require(np.allclose(scaler.mean_, xd[train].mean(axis=0), rtol=0, atol=1e-12), "학습 자료만으로 평균을 계산하지 않음")
    for key, item in config["sources"].items():
        require(digest(Path(item["path"]).read_bytes()) == digest(blobs[key]), "검사 중 원본 변경: " + key)
    packet = run_folder / "development_packet.npz"
    temp = packet.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, X=xd, Y=yd, patient_ids=pd, video_ids=vd, split=split)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, packet)
    with np.load(packet, allow_pickle=False) as checked:
        require(set(checked.files) == {"X", "Y", "patient_ids", "video_ids", "split"}, "개발 묶음 구조 불일치")
        for key, value in {"X": xd, "Y": yd, "patient_ids": pd, "video_ids": vd, "split": split}.items():
            require(np.array_equal(checked[key], value), "개발 묶음 저장 후 값 불일치")
    return {"status": "passed", "train_people": 175, "train_records": 203, "validation_people": 37,
            "validation_records": 40, "packet_records": len(xd), "packet_people": len(set(pd)),
            "excluded_previous_test_people": len(groups["test"]), "new_test_prediction_computed": False,
            "new_hyperparameter_search": False, "alpha": 1.0, "validation_person_mae": ridge_mae,
            "validation_mean_baseline_mae": baseline_mae,
            "validation_prediction_max_absolute_difference": float(np.max(np.abs(pv - saved))),
            "source_files_unchanged": True, "versions": {"numpy": np.__version__, "sklearn": __import__("sklearn").__version__},
            "artifacts": [{"path": packet.relative_to(root).as_posix(), "sha256": digest(packet.read_bytes())}]}


def report(result, run_id):
    return f"""# 첫 개선 실험을 위한 준비 결과

기존 모델의 점수 계산 부분을 다시 학습해 **검증 자료에서 같은 예측이 나오는지** 확인했다.
개선 모델을 선택하거나 새로운 최종 평가를 한 것은 아니다.

## 왜 이 순서인가

저장된 오차를 읽는 검사 다음에는 실제 계산 환경에서 같은 모델을 다시 만들 수 있어야 한다.
입력 특징은 기존 값을 재사용하고 점수 계산 부분만 실행하여, 영상 처리와 모델 변경의 영향을 섞지 않았다.
새 실험 도구에는 이미 결과를 본 평가 자료 38명이 들어가지 않도록 별도 개발 묶음을 만들었다.

| 용도 | 사람 수 | 기록 수 |
|---|---:|---:|
| 기존 조건 그대로 학습 | {result['train_people']} | {result['train_records']} |
| 기존 조건 재현 확인 | {result['validation_people']} | {result['validation_records']} |
| 개발 묶음 합계 | {result['packet_people']} | {result['packet_records']} |

## 무엇을 실행했는가

기존 사람 분할·정답 대조 → 기존 평가 사람 제외 → 학습 자료만으로 특징의 크기 조정 → Ridge(alpha=1) 재학습 → 검증 예측 대조 → 개발 묶음 저장 순서다.
512개 특징에서 좌우 GDI 두 값을 예측했다. 기존에 설치된 NumPy와 scikit-learn을 사용했고 새로운 설치는 없었다.
변경할 설정을 찾는 탐색은 하지 않았다. 평가 자료의 새로운 예측도 계산하지 않았다.

검증 자료의 사람별 평균 절대 오차는 **{result['validation_person_mae']:.6f}**였다.
학습 평균만 답하는 비교 모델은 **{result['validation_mean_baseline_mae']:.6f}**였다.
기존 검증 예측과 이번 예측의 최대 차이는 **{result['validation_prediction_max_absolute_difference']:.3g}**였다.
이 숫자는 앞서 보고한 평가 자료 오차 8.8663과 자료가 다르므로 직접 성능 향상으로 비교하지 않는다.

## 만들어진 것과 남은 것

[개발 묶음](runs/{run_id}/development_packet.npz)에는 학습·검증 자료만 들어 있다. 원본은 경로로 읽고 그대로 유지했다.
기존 전체 특징 파일에는 평가 자료도 있으므로, 향후 후보 비교 코드는 이 개발 묶음만 입력으로 사용해야 한다.
현재 검증 자료 37명도 이미 여러 번 본 자료다. 여기서 좋아져도 독립적인 최종 성능이 입증되는 것은 아니다.
새 최종 평가 자료 확보·분리, 첫 개선 후보의 구체적인 비교 조건은 아직 확정하지 않았다.

첫 후보는 기존 특징을 그대로 쓰고 점수 계산 부분을 바꾸는 방향으로 준비할 수 있다.
이유는 새 영상 처리 없이 계산 방법의 효과부터 비교할 수 있기 때문이다. 이 후보의 우수성은 아직 미검증이다.
Pro 로그인은 별도 도구로 연결했으며, 최신 상태는 상위 폴더의 `account-status.json`을 확인한다.
AI 작업 실행·예약 실행은 아직 연결하지 않았다.

실행 번호: `{run_id}`. [시작 전 의도](runs/{run_id}/plan.json), [결과](runs/{run_id}/result.json), [완료 기록](runs/{run_id}/receipt.json).
"""
