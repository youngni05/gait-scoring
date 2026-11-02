"""Independently check labels, splits, cached inputs and reported errors."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/gdi-ridge-experiment-20261016"
SAMPLE = ROOT / "outputs/stanford-sample-check-20261016"


def numeric_id(value):
    return str(int(float(value)))


def main():
    prepared = json.loads((OUT / "prepared.json").read_text(encoding="utf-8"))
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    predictions = list(csv.DictReader((OUT / "predictions.csv").open(encoding="utf-8-sig")))
    records = {r["video_id"]:r for r in prepared["records"] if r["status"] == "included"}
    assert len(predictions) == len(records) == 285
    original_labels = defaultdict(list)
    for row in csv.DictReader((SAMPLE / "gdi.csv").open(encoding="utf-8-sig")):
        key = (numeric_id(row["Patient_ID"]),numeric_id(row["examid"]),row["side"])
        original_labels[key].append(float(row["GDI"]))
    patient_splits = defaultdict(set)
    input_hash_splits = defaultdict(set)
    errors = defaultdict(lambda:defaultdict(list))
    train_labels = []
    for row in predictions:
        record = records[row["video_id"]]
        person, split = row["patient_id"], row["split"]
        assert person == str(record["Patient_ID"])
        assert person in protocol["split"][split]["patient_ids"]
        patient_splits[person].add(split)
        data = np.load(OUT / record["input_file"])
        assert np.all(np.diff(data["frame_ids"]) == 1)
        input_hash_splits[hashlib.sha256(data["input"].tobytes()).hexdigest()].add(split)
        actual = []
        for side in ["L","R"]:
            labels = original_labels[(person,str(record["examid"]),side)]
            assert len(labels) == 1
            value = float(row[f"actual_{side}"])
            assert value == labels[0] == record["gdi"][side]
            actual.append(value)
        actual = np.array(actual)
        if split == "train":
            train_labels.append(actual)
        for model in ["ridge","baseline"]:
            error = np.abs(actual-np.array([float(row[f"{model}_{s}"]) for s in ["L","R"]]))
            errors[(split,model)][person].append(error)
    assert all(len(v) == 1 for v in patient_splits.values())
    assert all(len(v) == 1 for v in input_hash_splits.values())
    mean = np.mean(train_labels,axis=0)
    for row in predictions:
        np.testing.assert_allclose([float(row[f"baseline_{s}"]) for s in ["L","R"]],mean,rtol=0,atol=1e-12)
    independently_computed = {}
    for (split,model), per_person in errors.items():
        mae = float(np.mean([np.mean(values) for values in per_person.values()]))
        key = "ridge" if model == "ridge" else "training_mean_baseline"
        assert abs(mae-result["metrics"][split][key]["person_macro_MAE"]) < 1e-12
        independently_computed[f"{split}_{model}_person_MAE"] = mae
    source = ROOT / "outputs/gait-public-access-20261015/Stanford/video-gait-v1.zip"
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):
            digest.update(chunk)
    assert digest.hexdigest() == "e839da388ce2429397e095d1eb141d9e38553f42152db83d1062f438e2ba381e"
    audit = {"status":"passed", "label_sides_matched_to_original_table":len(predictions)*2,
             "people_in_multiple_splits":0,"exact_input_duplicates_across_splits":0,
             "baseline_recomputed_from_training_only":True,
             "MAE_recomputed_from_predictions":independently_computed,
             "source_archive_sha256_unchanged":True}
    (OUT / "verification.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps(audit,indent=2))


if __name__ == "__main__":
    main()
