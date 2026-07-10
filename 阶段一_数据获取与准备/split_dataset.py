# -*- coding: utf-8 -*-
"""
split_dataset.py — 阶段一 训练/验证/测试集划分
策略 A（被试级划分，推荐用于跨被试泛化）：
  将 32 被试按比例 7:2:1 划分（约 22 / 6 / 3）—— 与项目计划书一致
策略 B（试验级划分，用于被试内建模）：
  每被试 40 trial 按 7:2:1 随机划分（约 28 / 8 / 4）
输出 outputs/data_split.json（含每个 trial 的三元组归属与音乐刺激路径）。
"""
import json
import os
import random

from deap_loader import DEAPLoader, N_TRIALS


def split_by_subject(loader: DEAPLoader, val: float = 0.2, test: float = 0.1, seed: int = 42) -> dict:
    random.seed(seed)
    ids = loader.list_subjects()
    random.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(n * test)))
    n_val = max(1, int(round(n * val)))
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test : n_test + n_val])
    train_ids = set(ids[n_test + n_val :])
    return {
        "strategy": "subject",
        "ratios": {"train": 1 - val - test, "val": val, "test": test},
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }


def split_by_trial(loader: DEAPLoader, val: float = 0.2, test: float = 0.1, seed: int = 42) -> dict:
    random.seed(seed)
    split = {"strategy": "trial", "train": [], "val": [], "test": []}
    for sid in loader.list_subjects():
        trials = list(range(N_TRIALS))
        random.shuffle(trials)
        n = len(trials)
        n_test = max(1, int(round(n * test)))
        n_val = max(1, int(round(n * val)))
        for t in trials[:n_test]:
            split["test"].append([sid, t])
        for t in trials[n_test : n_test + n_val]:
            split["val"].append([sid, t])
        for t in trials[n_test + n_val :]:
            split["train"].append([sid, t])
    return split


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    loader = DEAPLoader(os.path.normpath(os.path.join(root, "..", "deap-dataset")))
    subj = split_by_subject(loader)
    trial = split_by_trial(loader)
    out = {
        "subject_level": subj,
        "trial_level": trial,
        "note": "trial 索引为 0-based；音乐刺激路径由 deap_loader.music_stimulus_path(trial) 获取",
    }
    op = os.path.join(root, "outputs", "data_split.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("被试级划分 ->", {k: len(v) for k, v in subj.items() if isinstance(v, list)})
    print("试验级划分 ->", {k: len(v) for k, v in trial.items() if isinstance(v, list)})
    print("划分已写入:", op)


if __name__ == "__main__":
    main()
