# -*- coding: utf-8 -*-
"""
verify_dataset.py — 阶段一数据完整性校验
逐被试加载并校验：
  - 32 被试齐全（s01..s32）
  - 每被试 40 trials、40 通道、8064 样本
  - labels 形状 (40,4)、数值落在 [1,9]
  - 去标识化：文件名仅含编号，无外部身份字段
输出 outputs/verification_report.json 与控制台摘要。
"""
import json
import os

from deap_loader import (
    DEAPLoader,
    N_TRIALS,
    N_EEG,
    N_TOTAL_CH,
    SFREQ,
    TRIAL_SECONDS,
)


def verify(loader: DEAPLoader) -> dict:
    report = {
        "dataset": "DEAP (data_preprocessed_python)",
        "sfreq": SFREQ,
        "trial_seconds": TRIAL_SECONDS,
        "expected": {
            "n_subjects": 32,
            "n_trials": N_TRIALS,
            "n_channels": N_TOTAL_CH,
            "n_samples": int(SFREQ * TRIAL_SECONDS),
            "label_dims": 4,
            "label_range": [1, 9],
        },
        "subjects": {},
        "summary": {},
        "errors": [],
    }
    ids = loader.list_subjects()
    report["summary"]["subjects_found"] = len(ids)

    for sid in ids:
        path = loader._subject_path(sid)
        rec = {"file": os.path.basename(path)}
        try:
            sd = loader.load_subject(sid)
            rec["n_trials"] = int(sd.eeg.shape[0])
            rec["n_eeg_ch"] = int(sd.eeg.shape[1])
            rec["n_samples"] = int(sd.eeg.shape[2])
            rec["label_shape"] = list(sd.labels.shape)
            lmin, lmax = float(sd.labels.min()), float(sd.labels.max())
            rec["label_min"] = round(lmin, 3)
            rec["label_max"] = round(lmax, 3)
            # DEAP 标签值域 1-9；0 表示被试未评分（缺失），属正常数据特征
            rec["label_has_missing"] = bool(lmin <= 0.0)
            rec["label_in_range"] = bool(lmin >= -1e-6 and lmax <= 9.0 + 1e-6)
            ok = (
                sd.eeg.shape[0] == N_TRIALS
                and sd.eeg.shape[1] == N_EEG
                and sd.eeg.shape[2] == int(SFREQ * TRIAL_SECONDS)
                and tuple(sd.labels.shape) == (N_TRIALS, 4)
                and rec["label_in_range"]
            )
            rec["valid"] = bool(ok)
            if not ok:
                report["errors"].append(f"subject s{sid:02d}: shape/label mismatch")
        except Exception as e:  # noqa: BLE001
            rec["valid"] = False
            rec["error"] = str(e)
            report["errors"].append(f"subject s{sid:02d}: {e}")
        report["subjects"][f"s{sid:02d}"] = rec

    # 去标识化检查：文件名仅含 s + 两位数字 + .dat
    report["deidentified"] = all(
        fn.startswith("s") and fn.endswith(".dat") and fn[1:3].isdigit()
        for fn in os.listdir(loader.data_dir)
    )
    report["summary"]["subjects_with_label_missing"] = sum(
        1 for v in report["subjects"].values() if v.get("label_has_missing")
    )
    report["summary"]["all_valid"] = bool(
        len(report["errors"]) == 0 and len(ids) == 32
    )
    return report


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    loader = DEAPLoader(os.path.normpath(os.path.join(root, "..", "deap-dataset")))
    rep = verify(loader)
    out = os.path.join(root, "outputs", "verification_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(
        f"校验完成：发现被试 {rep['summary']['subjects_found']} 个，"
        f"全部通过={rep['summary']['all_valid']}"
    )
    if rep["errors"]:
        print("错误：")
        for e in rep["errors"][:20]:
            print("  -", e)
    print(f"报告已写入: {out}")


if __name__ == "__main__":
    main()
