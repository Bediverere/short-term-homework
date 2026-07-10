# -*- coding: utf-8 -*-
"""
deap_loader.py
==============
DEAP 数据集（EEG × Music）加载与访问接口 —— 阶段一：数据获取与准备

DEAP 预处理版本（data_preprocessed_python）结构说明：
  - 每个被试一个 .dat 文件（s01.dat ... s32.dat），pickle 序列化（latin1 编码）
  - 文件内含两个键：
      'data'   : numpy.ndarray, shape (40, 40, 8064)
                 axis0 = 40 个 trial（试验）
                 axis1 = 40 个通道（前 32 为 EEG，后 8 为外围生理信号）
                 axis2 = 8064 个采样点（128 Hz × 63 s）
      'labels' : numpy.ndarray, shape (40, 4)
                 每行对应一个 trial 的自我评测：valence, arousal, dominance, liking（评分 1–9）

本模块提供：
  - 标准 10-20 电极命名与通道索引映射
  - 逐被试 / 按需加载（避免一次性将 3.1 GB 数据载入内存）
  - trial ↔ 音乐 MIDI 刺激的映射（exp_id_{i+1}.mid，对所有被试一致）
  - 采样率 / 时长等元信息
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

# ----------------------------------------------------------------------------
# DEAP 通道定义（与官方 data_preprocessed_python 顺序严格一致）
# ----------------------------------------------------------------------------
# 前 32 通道：EEG（按 Biosemi 32 导标准 10-20 布局顺序）
EEG_CHANNELS: List[str] = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
]

# 后 8 通道：外围生理信号（GSR / 呼吸 / 血容量 / 温度 / 3×EOG / EMG）
PERIPH_CHANNELS: List[str] = [
    "GSR", "Resp", "Plet", "Temp", "EOG1", "EOG2", "EOG3", "EMG",
]

# 标签列名（自我评测维度）
LABEL_NAMES: List[str] = ["valence", "arousal", "dominance", "liking"]

# 关键元信息
SFREQ: int = 128                      # 预处理版本采样率 (Hz)
TRIAL_SECONDS: float = 63.0          # 每个 trial 时长 (s) -> 128*63 = 8064
N_TRIALS: int = 40                   # 每被试 trial 数
N_EEG: int = 32                      # EEG 通道数
N_PERIPH: int = 8                    # 外围通道数
N_TOTAL_CH: int = N_EEG + N_PERIPH   # 总通道数 = 40
N_SUBJECTS: int = 32                 # 被试总数
N_SAMPLES: int = int(SFREQ * TRIAL_SECONDS)  # 8064


@dataclass
class SubjectData:
    """单个被试的 DEAP 数据容器（已解包为标准结构）。"""
    subject_id: int
    eeg: np.ndarray                 # (40, 32, 8064)  trials × EEG-channels × samples
    peripheral: np.ndarray          # (40, 8, 8064)   trials × peripheral × samples
    labels: np.ndarray              # (40, 4)         trials × [valence,arousal,dominance,liking]
    sfreq: int = SFREQ
    trial_seconds: float = TRIAL_SECONDS

    @property
    def n_trials(self) -> int:
        return self.eeg.shape[0]


class DEAPLoader:
    """DEAP 数据集加载器（数据获取与准备阶段的核心接口）。"""

    def __init__(self, deap_root: str):
        self.deap_root = os.path.abspath(deap_root)
        self.data_dir = os.path.join(self.deap_root, "data_preprocessed_python")
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"未找到预处理数据目录: {self.data_dir}")

    # --- 被试枚举 -----------------------------------------------------------
    def list_subjects(self) -> List[int]:
        ids: List[int] = []
        for fn in os.listdir(self.data_dir):
            if fn.startswith("s") and fn.endswith(".dat"):
                try:
                    ids.append(int(fn[1:3]))
                except ValueError:
                    pass
        return sorted(ids)

    def _subject_path(self, subject_id: int) -> str:
        return os.path.join(self.data_dir, f"s{subject_id:02d}.dat")

    def exists(self, subject_id: int) -> bool:
        return os.path.isfile(self._subject_path(subject_id))

    # --- 加载单个被试 -------------------------------------------------------
    def load_subject(self, subject_id: int, with_peripheral: bool = True) -> SubjectData:
        path = self._subject_path(subject_id)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"被试文件不存在: {path}")
        with open(path, "rb") as f:
            raw = pickle.load(f, encoding="latin1")
        data = np.asarray(raw["data"], dtype=np.float32)    # (40, 40, 8064)
        labels = np.asarray(raw["labels"], dtype=np.float32)  # (40, 4)
        eeg = data[:, :N_EEG, :]
        periph = data[:, N_EEG:, :] if with_peripheral else np.empty((0, 0, 0), dtype=np.float32)
        return SubjectData(
            subject_id=subject_id,
            eeg=eeg,
            peripheral=periph,
            labels=labels,
        )

    # --- 遍历所有被试（生成器，内存友好）-----------------------------------
    def iterate_subjects(self, subject_ids: Optional[List[int]] = None) -> Iterator[SubjectData]:
        ids = subject_ids or self.list_subjects()
        for sid in ids:
            yield self.load_subject(sid)

    # --- 音乐刺激映射 -------------------------------------------------------
    def music_stimulus_path(self, trial_idx: int, tempo24: bool = False) -> str:
        """trial_idx 为 0-based；DEAP 的 trial 顺序即 experiment_id 顺序。
        返回对应 MIDI 刺激文件路径（所有被试共享同一套刺激）。"""
        sub = "audio_stimuli_MIDI_tempo24" if tempo24 else "audio_stimuli_MIDI"
        d = os.path.join(self.deap_root, sub)
        return os.path.join(d, f"exp_id_{trial_idx + 1:02d}.mid")

    def build_trial_index(self) -> List[Dict[str, Any]]:
        """构建 (subject, trial) -> 音乐刺激的全局索引，供跨模态对齐使用。"""
        idx: List[Dict[str, Any]] = []
        for sid in self.list_subjects():
            for t in range(N_TRIALS):
                idx.append({
                    "subject_id": sid,
                    "trial_idx": t,
                    "music_midi": self.music_stimulus_path(t),
                })
        return idx


def load_deap(deap_root: Optional[str] = None) -> DEAPLoader:
    """便捷构造函数：默认使用与本项目同级的 deap-dataset 目录。"""
    if deap_root is None:
        here = os.path.dirname(os.path.abspath(__file__))
        deap_root = os.path.normpath(os.path.join(here, "..", "deap-dataset"))
    return DEAPLoader(deap_root)


if __name__ == "__main__":
    ld = load_deap()
    print("被试列表:", ld.list_subjects())
    sd = ld.load_subject(1)
    print("s01 eeg shape:", sd.eeg.shape, "| labels shape:", sd.labels.shape)
    print("音乐刺激(trial0):", ld.music_stimulus_path(0))
