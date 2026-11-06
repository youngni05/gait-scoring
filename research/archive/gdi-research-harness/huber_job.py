"""Second predeclared candidate. Convergence failures remain failures."""
import csv
import io
import json
from pathlib import Path
import statistics
import sys
import time
import warnings

from harness import digest, require, save


def fit_candidate(x, y):
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import HuberRegressor
    models, details = [], []
    for index, side in enumerate(("L", "R")):
        model = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=1000)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model.fit(x, y[:, index])
        problems = [str(w.message) for w in captured if issubclass(w.category, ConvergenceWarning)]
        details.append({"side": side, "iterations": int(model.n_iter_), "convergence_warnings": problems,
                        "other_warnings": [str(w.message) for w in captured if not issubclass(w.category, ConvergenceWarning)]})
        if problems:
            return None, details
        require(np.isfinite(model.coef_).all() and np.isfinite(model.intercept_), "Huber 모델 값 비정상")
        models.append(model)
    return models, details


def inspect(config, root, run_folder):
    tick = time.perf_counter()
    blobs = {}
    for key, item in config["sources"].items():
        blobs[key] = Path(item["path"]).read_bytes()
        require(digest(blobs[key]) == item["sha256"], "원본 지문 불일치: " + key)
    previous, receipt = json.loads(blobs["development_result"]), json.loads(blobs["development_receipt"])
    require(receipt["status"] == "completed" and receipt["result_sha256"] == digest(blobs["development_result"]), "개발 자료 완료 증거 불일치")
    require(previous["artifacts"][0]["sha256"] == digest(blobs["packet"]), "기존 개발 묶음과 다른 입력")
    sys.path.insert(0, config["dependency_path"])
    import numpy as np
    import sklearn
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from weighted_ridge_job import metrics
    with np.load(io.BytesIO(blobs["packet"]), allow_pickle=False) as data:
        x, y, p, v, s = [data[key].copy() for key in ("X", "Y", "patient_ids", "video_ids", "split")]
    require(x.shape == (243, 512) and y.shape == (243, 2) and np.isfinite(x).all() and np.isfinite(y).all(), "개발 자료 크기/값 불일치")
    require(set(s) == {"train", "validation"} and len(set(v)) == len(v), "자료 분할 또는 영상 중복 오류")
    train, val = s == "train", s == "validation"
    require(not set(p[train]) & set(p[val]), "학습·검증 사람 중복")
    require((int(train.sum()), len(set(p[train])), int(val.sum()), len(set(p[val]))) == (203, 175, 40, 37), "고정 사람/기록 수 불일치")
    with threadpool_limits(limits=4):
        scaler = StandardScaler().fit(x[train])
        xt, xv = scaler.transform(x[train]), scaler.transform(x[val])
        baseline = Ridge(alpha=1.0, solver="svd").fit(xt, y[train])
        bp = baseline.predict(xv)
        models, details = fit_candidate(xt, y[train])
        save(run_folder / "fit_details.json", {"fixed_settings": {"epsilon": 1.35, "alpha": 0.0001, "max_iter": 1000},
                                               "sides": details, "status": "converged" if models else "failed_to_converge"})
        require(models is not None, "Huber가 고정된 1000회 반복 한도 안에 수렴하지 않았다. fit_details.json에 남겼으며 조건을 바꾸거나 자동 재시도하지 않는다.")
        hp = np.column_stack([model.predict(xv) for model in models])
    bm, be = metrics(y[val], bp, p[val])
    hm, he = metrics(y[val], hp, p[val])
    require(abs(bm["person_mae"] - previous["validation_person_mae"]) < 1e-9, "기존 모델 재현 실패")
    parameter_file = run_folder / "model_parameters.npz"
    np.savez_compressed(parameter_file, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                        coef=np.stack([model.coef_ for model in models]), intercept=np.array([model.intercept_ for model in models]))
    with np.load(parameter_file, allow_pickle=False) as saved:
        restored = (x[val] - saved["scaler_mean"]) / saved["scaler_scale"] @ saved["coef"].T + saved["intercept"]
        require(np.allclose(restored, hp, rtol=0, atol=1e-10), "저장한 Huber 모델의 예측 재현 실패")
    pred_file = run_folder / "validation_predictions.csv"
    with pred_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_id", "video_id", "actual_L", "actual_R", "baseline_L", "baseline_R", "huber_L", "huber_R"])
        for pid, vid, actual, base, candidate in zip(p[val], v[val], y[val], bp, hp):
            writer.writerow([pid, vid, *actual, *base, *candidate])
    with pred_file.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    csv_metrics = {}
    for name in ("baseline", "huber"):
        by_person = {}
        for row in rows:
            by_person.setdefault(row["patient_id"], []).append(statistics.mean(abs(float(row[name + "_" + side]) - float(row["actual_" + side])) for side in ("L", "R")))
        csv_metrics[name] = statistics.mean(statistics.mean(values) for values in by_person.values())
    require(abs(csv_metrics["baseline"] - bm["person_mae"]) < 1e-10 and abs(csv_metrics["huber"] - hm["person_mae"]) < 1e-10, "CSV에서 독립 오차 재계산 실패")
    for key, item in config["sources"].items():
        require(digest(Path(item["path"]).read_bytes()) == digest(blobs[key]), "실행 중 원본 변경: " + key)
    save(run_folder / "verification.json", {"baseline_reproduced": True, "csv_recomputed_mae": csv_metrics,
        "saved_model_predictions_reproduced": True, "source_files_unchanged": True, "all_sides_converged": True,
        "test_rows_available_to_training": False, "scaler_fit": "unweighted train only"})
    artifacts = [parameter_file, pred_file, run_folder / "fit_details.json", run_folder / "verification.json"]
    return {"status": "passed", "baseline": bm, "huber": hm, "mae_reduction": bm["person_mae"] - hm["person_mae"],
            "validation_people_better": int((he < be).sum()), "validation_people_worse": int((he > be).sum()),
            "new_test_prediction_computed": False, "hyperparameter_search": False,
            "independent_final_validation": False, "clinical_validation": False,
            "fit_details": details, "elapsed_seconds": time.perf_counter() - tick,
            "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__},
            "artifacts": [{"path": path.relative_to(root).as_posix(), "sha256": digest(path.read_bytes())} for path in artifacts]}


def report(result, run_id):
    d = result["mae_reduction"]
    outcome = "오차가 줄었다" if d > 0 else "오차가 늘었다" if d < 0 else "오차가 같았다"
    return f"""# 두 번째 실제 비교 — Huber 회귀

**이번 검증 자료에서는 {outcome}.** 학습 175명·203건과 검증 37명·40건, 같은 512개 특징으로 비교했다.

| 조건 | 검증 사람별 평균 절대 오차 |
|---|---:|
| 기존 Ridge | {result['baseline']['person_mae']:.6f} |
| Huber 회귀 | {result['huber']['person_mae']:.6f} |

기존 오차에서 새 오차를 뺀 값은 {d:.6f}다. 양수면 감소, 음수면 증가다.
검증 37명 중 {result['validation_people_better']}명은 나아졌고 {result['validation_people_worse']}명은 나빠졌다.
통계적 유의성이나 독립적인 최종 성능을 확인한 결과는 아니다.

## 무엇을 왜 바꿨는가

Huber는 큰 예측 오차의 영향을 줄여 보는 방법이다. 현재 특징에서 이 계산 방식이 도움이 되는지 비교했다.
기존과 같은 학습 자료만으로 같은 표준화를 적용하고, 좌우 점수를 각각 별도 모델로 학습했다.
기록마다 같은 학습 비중을 사용했다. 사람별 비중을 바꾼 첫 후보와 섞지 않았다.
설정은 미리 정한 epsilon=1.35, alpha=0.0001, max_iter=1000으로 고정했고 값을 탐색하지 않았다.
회귀 방식과 정규화 설정이 함께 달라지므로 차이 전체를 큰 오차 처리만의 효과로 해석할 수 없다.

## 확인한 범위

좌우 모델이 수렴했는지, 기존 Ridge의 오차가 재현되는지, 저장한 모델이 같은 예측을 내는지, CSV로 오차를 다시 계산했을 때 같은지 확인했다.
원본 입력의 내용 지문은 실행 전후 같았다. 기존 평가 38명의 입력이나 정답은 이 작업에 없었다.
계산 시간은 {result['elapsed_seconds']:.3f}초였다. 추가 AI 요청은 없었다.

기존 검증 자료는 이미 본 자료다. 노인·가정 영상·임상적 효과, 다른 연구보다의 우수성은 확인하지 않았다.
Huber는 기존 방법을 적용한 비교이며, 이 결과 자체로 새로운 알고리즘의 독창성을 입증하지 않는다.

실행 번호: `{run_id}`. [계획](runs/{run_id}/plan.json), [수치](runs/{run_id}/result.json), [수렴 기록](runs/{run_id}/fit_details.json), [독립 검사](runs/{run_id}/verification.json).
"""
