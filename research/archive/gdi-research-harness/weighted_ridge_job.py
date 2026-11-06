"""Fixed person-weighted Ridge versus fixed original Ridge, development data only."""
from collections import Counter
import csv
import io
import json
from pathlib import Path
import statistics
import sys
import time

from harness import digest, require, save


def equal_person_weights(people):
    import numpy as np
    counts = Counter(people)
    raw = np.array([1.0 / counts[p] for p in people], dtype=np.float64)
    weights = raw / raw.mean()
    totals = [float(weights[np.array(people) == p].sum()) for p in counts]
    require(np.allclose(totals, totals[0], rtol=0, atol=1e-12), "사람별 총 학습 비중 불일치")
    require(abs(float(weights.mean()) - 1.0) < 1e-12, "평균 학습 가중치가 1이 아님")
    return weights


def metrics(y, prediction, people):
    import numpy as np
    errors = np.abs(y - prediction)
    require(np.isfinite(errors).all(), "비정상 예측/오차")
    per_person = np.stack([errors[people == p].mean(axis=0) for p in sorted(set(people))])
    return {"person_mae": float(per_person.mean()), "left_person_mae": float(per_person[:, 0].mean()),
            "right_person_mae": float(per_person[:, 1].mean()), "record_mae": float(errors.mean()),
            "people": len(per_person), "records": len(people)}, per_person.mean(axis=1)


def verify_saved_predictions(path):
    """Recompute errors from exported CSV with stdlib, independently of array metrics."""
    from collections import defaultdict
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    results = {}
    for method in ("baseline", "weighted"):
        people = defaultdict(list)
        for row in rows:
            error = statistics.mean(abs(float(row[method + "_" + side]) - float(row["actual_" + side])) for side in ("L", "R"))
            people[row["patient_id"]].append(error)
        results[method] = statistics.mean(statistics.mean(values) for values in people.values())
    return results


def inspect(config, root, run_folder):
    tick = time.perf_counter()
    blobs = {}
    for key, item in config["sources"].items():
        blobs[key] = Path(item["path"]).read_bytes()
        require(digest(blobs[key]) == item["sha256"], "원본 지문 불일치: " + key)
    previous = json.loads(blobs["development_result"])
    receipt = json.loads(blobs["development_receipt"])
    require(receipt["status"] == "completed" and digest(blobs["development_result"]) == receipt["result_sha256"], "개발 자료의 기존 완료 증거 불일치")
    require(digest(blobs["packet"]) == previous["artifacts"][0]["sha256"], "기존 개발 묶음과 다른 입력")
    require(config["alpha"] == 1.0, "합의된 고정 alpha=1과 다름")
    sys.path.insert(0, config["dependency_path"])
    import numpy as np
    import sklearn
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    with np.load(io.BytesIO(blobs["packet"]), allow_pickle=False) as packet:
        x, y = packet["X"].copy(), packet["Y"].copy()
        people, videos, split = packet["patient_ids"].copy(), packet["video_ids"].copy(), packet["split"].copy()
    require(x.shape == (243, 512) and y.shape == (243, 2), "개발 입력 크기 불일치")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "비정상 입력")
    require(set(split) == {"train", "validation"}, "허용하지 않은 자료 분할 포함")
    train, val = split == "train", split == "validation"
    require(len(set(videos)) == len(videos), "영상 ID 중복")
    require(not set(people[train]) & set(people[val]), "학습·검증 사람 중복")
    require((int(train.sum()), len(set(people[train])), int(val.sum()), len(set(people[val]))) == (203, 175, 40, 37), "고정된 사람·기록 수 불일치")
    weights = equal_person_weights(people[train])
    with threadpool_limits(limits=4):
        scaler = StandardScaler().fit(x[train])
        xt, xv = scaler.transform(x[train]), scaler.transform(x[val])
        baseline = Ridge(alpha=1.0, solver="svd").fit(xt, y[train])
        weighted = Ridge(alpha=1.0, solver="svd").fit(xt, y[train], sample_weight=weights)
        bp, wp = baseline.predict(xv), weighted.predict(xv)
        # Independent weighted ridge equation, using a dual system and weighted centering.
        xm, ym = np.average(xt, axis=0, weights=weights), np.average(y[train], axis=0, weights=weights)
        a = (xt - xm) * np.sqrt(weights)[:, None]
        b = (y[train] - ym) * np.sqrt(weights)[:, None]
        coefficients = a.T @ np.linalg.solve(a @ a.T + np.eye(len(a)), b)
        independent = (xv - xm) @ coefficients + ym
        require(np.allclose(wp, independent, rtol=0, atol=1e-7), "가중 Ridge의 독립 수식과 예측 불일치")
    bm, be = metrics(y[val], bp, people[val])
    wm, we = metrics(y[val], wp, people[val])
    require(abs(bm["person_mae"] - previous["validation_person_mae"]) < 1e-9, "비교 기준 모델이 기존 검증 오차를 재현하지 못함")
    parameters = run_folder / "model_parameters.npz"
    np.savez_compressed(parameters, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                        baseline_coef=baseline.coef_, baseline_intercept=baseline.intercept_,
                        weighted_coef=weighted.coef_, weighted_intercept=weighted.intercept_)
    with np.load(parameters, allow_pickle=False) as restored:
        restored_x = (x[val] - restored["scaler_mean"]) / restored["scaler_scale"]
        for name, pred in (("baseline", bp), ("weighted", wp)):
            restored_pred = restored_x @ restored[name + "_coef"].T + restored[name + "_intercept"]
            require(np.allclose(restored_pred, pred, rtol=0, atol=1e-10), "저장 모델의 예측 재현 실패")
    prediction_file = run_folder / "validation_predictions.csv"
    with prediction_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_id", "video_id", "actual_L", "actual_R", "baseline_L", "baseline_R", "weighted_L", "weighted_R"])
        for p, v, yy, bb, ww in zip(people[val], videos[val], y[val], bp, wp):
            writer.writerow([p, v, *yy, *bb, *ww])
    weight_file = run_folder / "training_weights.csv"
    with weight_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_id", "video_id", "weight"])
        writer.writerows(zip(people[train], videos[train], weights))
    recomputed = verify_saved_predictions(prediction_file)
    require(abs(recomputed["baseline"] - bm["person_mae"]) < 1e-10 and abs(recomputed["weighted"] - wm["person_mae"]) < 1e-10, "CSV 기준 독립 오차 재계산 실패")
    for key, item in config["sources"].items():
        require(digest(Path(item["path"]).read_bytes()) == digest(blobs[key]), "실행 도중 원본 변경: " + key)
    verification = {"baseline_reproduced": True, "equal_total_weight_per_training_person": True,
                    "training_weight_mean": float(weights.mean()), "independent_equation_max_prediction_difference": float(np.max(np.abs(independent - wp))),
                    "saved_models_reproduce_predictions": True, "csv_recomputed_mae": recomputed,
                    "train_validation_people_disjoint": True, "test_rows_available_to_training": False,
                    "scaler_fit": "unweighted train only, unchanged between methods", "source_files_unchanged": True}
    save(run_folder / "verification.json", verification)
    difference = bm["person_mae"] - wm["person_mae"]
    counts = Counter(people[train])
    artifacts = [parameters, prediction_file, weight_file, run_folder / "verification.json"]
    return {"status": "passed", "baseline": bm, "weighted": wm, "mae_reduction": difference,
            "relative_mae_reduction_percent": 100 * difference / bm["person_mae"],
            "validation_people_better": int((we < be).sum()), "validation_people_worse": int((we > be).sum()),
            "validation_people_equal": int((we == be).sum()), "train_people": len(counts), "train_records": int(train.sum()),
            "training_people_with_multiple_records": sum(n > 1 for n in counts.values()),
            "training_records_per_person_counts": dict(sorted(Counter(counts.values()).items())),
            "min_training_record_weight": float(weights.min()), "max_training_record_weight": float(weights.max()),
            "equal_total_weight_per_training_person": float(weights.sum() / len(counts)), "alpha": 1.0,
            "new_test_prediction_computed": False, "hyperparameter_search": False,
            "independent_final_validation": False, "clinical_validation": False,
            "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__},
            "elapsed_seconds": time.perf_counter() - tick,
            "artifacts": [{"path": path.relative_to(root).as_posix(), "sha256": digest(path.read_bytes())} for path in artifacts]}


def report(result, run_id):
    reduction = result["mae_reduction"]
    outcome = "오차가 줄었다" if reduction > 0 else "오차가 늘었다" if reduction < 0 else "오차가 같았다"
    return f"""# 첫 실제 비교 — 사람별 학습 비중을 같게 한 Ridge

**이번 검증 자료에서는 {outcome}.** 기존 모델과 같은 입력·자료·설정으로 학습하고, 각 사람에게 주는 학습 비중만 바꿨다.

| 조건 | 검증 사람별 평균 절대 오차 |
|---|---:|
| 기존 Ridge | {result['baseline']['person_mae']:.6f} |
| 사람별 비중을 같게 한 Ridge | {result['weighted']['person_mae']:.6f} |

기존 오차에서 새 오차를 뺀 값은 **{reduction:.6f}**다. 양수면 감소, 음수면 증가다.
기존 대비 오차 감소율은 {result['relative_mae_reduction_percent']:.3f}%다.
검증 37명 중 오차가 줄어든 사람은 {result['validation_people_better']}명, 늘어난 사람은 {result['validation_people_worse']}명이다.
전체 평균과 개인별 결과는 다를 수 있다. 이 변화의 통계적 유의성은 검사하지 않았다.

## 왜 이 비교를 했는가

학습 자료는 175명의 203건이며, {result['training_people_with_multiple_records']}명은 기록이 여러 개다.
기존 방식은 기록마다 같은 비중을 주므로 기록이 많은 사람이 학습에 더 큰 영향을 준다.
평가는 사람마다 같은 비중으로 하므로 학습 비중도 사람별로 맞추면 도움이 될 가능성을 먼저 확인했다.
이는 학습 조건의 효과를 보는 비교이며 새로운 알고리즘의 독창성을 입증한 실험은 아니다.

## 어떻게 바꿨는가

각 학습 기록에 `1 ÷ 그 사람의 학습 기록 수`만큼 가중치를 주고, 전체 기록의 평균 가중치가 1이 되도록 조정했다.
각 사람의 총 가중치는 {result['equal_total_weight_per_training_person']:.6f}로 같다.
반복 기록이 많은 사람의 기록 하나당 비중은 작아진다. 전체 가중치 합은 기존과 맞춰 Ridge의 alpha=1 설정과 함께 비교했다.

저장된 512개 특징 → 학습 자료만으로 같은 표준화 → 기존/가중 Ridge 학습 → 같은 검증 37명·40건 예측 → 사람별 오차 비교 순서다.
표준화는 두 모델 모두 기존의 기록별 동일 비중 방식으로 유지했다. 모델을 학습하는 단계의 가중치만 달라졌다.
특징 추출 모델을 새로 조정하는 파인튜닝, 새 영상 처리, 설정값 탐색은 하지 않았다.

## 확인한 것과 남은 것

기존 모델의 검증 오차 재현, 사람별 가중치 합, 별도 수식으로 계산한 가중 Ridge 예측, 저장 후 예측 재현, CSV에서의 독립 오차 재계산을 확인했다.
개발 자료의 내용 지문도 실행 전후 같았다. 기존에 설치된 NumPy·scikit-learn을 사용했고 추가 AI 요청은 없었다.
계산 시간은 {result['elapsed_seconds']:.3f}초였다.

기존 평가 38명은 입력 묶음에 없으며 새 예측도 계산하지 않았다.
이번 검증 37명은 이미 본 자료다. 이 결과를 독립적인 최종 성능이나 노인·가정 영상·임상적 효과로 해석하지 않는다.
이미 알려진 가중 학습 방법을 적용한 비교이므로 기존 연구보다 우수하거나 새롭다는 결론도 낼 수 없다.

실행 번호: `{run_id}`. [시작 전 계획](runs/{run_id}/plan.json), [수치](runs/{run_id}/result.json), [독립 검사](runs/{run_id}/verification.json), [예측값](runs/{run_id}/validation_predictions.csv).
다음 후보인 Huber 회귀는 별도 비교가 필요하다. 예약 실행은 아직 켜지 않았다.
"""
