# -*- coding: utf-8 -*-
"""
eeg_preprocess.py
=================
阶段二（EEG 信号预处理与特征工程）之 —— EEG 信号预处理流水线

基于 MNE-Python 构建标准化、可复用的 DEAP EEG 预处理流水线：
  1. 带通滤波：0.5–45 Hz 双向零相位 FIR（firwin 设计）
  2. 工频陷波：50 Hz（中国电网）FIR 零相位陷波
  3. ICA 伪迹剔除：自动检测并剔除眼电（EOG）相关成分

设计要点：
  - 逐 trial 处理（每个 trial 63 s @ 128 Hz），内存友好
  - 利用 DEAP 数据集中独立记录的 3 路 EOG（EOG1/2/3，位于外围通道）
    作为 ICA 眼电成分检测的参考，剔除后再丢弃 EOG 通道，仅保留 32 路 EEG
  - 所有参数集中在 PREPROC_CONFIG，便于复现与调参
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import os
import sys

# 使本模块被其它脚本 import 时也能找到阶段一的 deap_loader
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "阶段一_数据获取与准备")))

import mne

# 抑制 MNE 在非交互场景下的冗余日志
mne.set_log_level("WARNING")
warnings.filterwarnings("ignore")

# 复用阶段一加载器中的通道定义
from deap_loader import EEG_CHANNELS, PERIPH_CHANNELS, SFREQ  # noqa: E402

# DEAP 外围通道中 3 路眼电（EOG1/EOG2/EOG3）在 peripheral 数组里的下标
_EOG_PERIPH_IDX = [PERIPH_CHANNELS.index(n) for n in ("EOG1", "EOG2", "EOG3")]
EOG_NAMES = ["EOG1", "EOG2", "EOG3"]

# 滤波/ICA 默认参数
PREPROC_CONFIG = {
    "l_freq": 0.5,        # 带通下限 (Hz)
    "h_freq": 45.0,       # 带通上限 (Hz)
    "notch_freq": 50.0,   # 工频陷波频率 (Hz)
    "ica_n_components": 15,
    "ica_random_state": 42,
    "frontal_eog_channels": ["Fp1", "Fp2"],  # 用额极通道作眼电代理（DEAP 专用 EOG 噪声大、不可靠）
    "frontal_corr_threshold": 0.30,  # ICA 成分与额极通道相关阈值（强额极相关）
    "artifact_kurtosis_threshold": 5.0,  # 源信号峭度阈值：超高斯(>5)≈眨眼/眼电伪迹
    "ica_max_excluded": 3,           # 单 trial 最多剔除的 ICA 成分数，防止过度清洗
}


@dataclass
class PreprocResult:
    """单 trial 预处理结果容器。"""

    raw_clean: "mne.io.Raw"          # 干净 EEG（仅 32 路）
    raw_filtered: "mne.io.Raw"       # 仅滤波+陷波、未做 ICA 的版本（用于对比）
    ica: "mne.preprocessing.ICA"
    excluded_components: List[int] = field(default_factory=list)
    used_eog_reference: bool = False


def build_raw(
    eeg_trial: np.ndarray,
    eog_trial: Optional[np.ndarray] = None,
    sfreq: int = SFREQ,
) -> "mne.io.Raw":
    """由一个 trial 的 EEG（及可选 EOG）构建 MNE Raw 对象。

    Parameters
    ----------
    eeg_trial : np.ndarray  shape (32, n_samples)
    eog_trial : np.ndarray | None  shape (3, n_samples)，对应 EOG1/2/3
    """
    eeg_trial = np.asarray(eeg_trial, dtype=np.float64)
    ch_names = list(EEG_CHANNELS)
    ch_types = ["eeg"] * len(EEG_CHANNELS)

    if eog_trial is not None:
        eog_trial = np.asarray(eog_trial, dtype=np.float64)
        ch_names = ch_names + EOG_NAMES
        ch_types = ch_types + ["eog"] * 3
        data = np.vstack([eeg_trial, eog_trial])
    else:
        data = eeg_trial

    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)
    # 设置标准 10-20 电极位置；EOG 无标准位置，忽略其缺失
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore", verbose=False)
    return raw


def preprocess_trial(
    eeg_trial: np.ndarray,
    eog_trial: Optional[np.ndarray] = None,
    sfreq: int = SFREQ,
    cfg: dict = PREPROC_CONFIG,
) -> PreprocResult:
    """对单个 trial 执行完整预处理流水线。"""
    raw = build_raw(eeg_trial, eog_trial, sfreq)

    # 1) 零相位 FIR 带通滤波
    raw_f = raw.copy().filter(
        l_freq=cfg["l_freq"], h_freq=cfg["h_freq"],
        method="fir", phase="zero", fir_design="firwin", verbose=False,
    )
    # 2) 工频陷波
    raw_f = raw_f.notch_filter(
        freqs=cfg["notch_freq"], method="fir", phase="zero", verbose=False,
    )

    # 3) ICA 眼电伪迹剔除
    ica = mne.preprocessing.ICA(
        n_components=cfg["ica_n_components"],
        random_state=cfg["ica_random_state"],
        max_iter="auto", verbose=False,
    )
    ica.fit(raw_f)

    excluded: List[int] = []
    used_eog = True
    try:
        bad, scores = _frontal_bad_components(
            ica, raw_f,
            cfg["frontal_eog_channels"],
            cfg["frontal_corr_threshold"],
            cfg["artifact_kurtosis_threshold"],
        )
        # 限制最多剔除数量，避免过度清洗（眼电/眨眼通常仅占 1–3 个成分）
        if len(bad) > cfg["ica_max_excluded"]:
            order = np.argsort(-np.array(scores, dtype=float))
            bad = [int(order[i]) for i in range(cfg["ica_max_excluded"])]
        excluded = [int(b) for b in bad]
    except Exception as e:  # 退化为不剔除
        warnings.warn(f"眼电成分检测失败，跳过 ICA 剔除: {e}")
        excluded = []

    ica.exclude = excluded
    raw_clean = ica.apply(raw_f.copy())
    # 剔除后仅保留 32 路 EEG
    if eog_trial is not None:
        raw_clean = raw_clean.drop_channels(EOG_NAMES)

    return PreprocResult(
        raw_clean=raw_clean,
        raw_filtered=raw_f.copy().drop_channels(EOG_NAMES) if eog_trial is not None else raw_f,
        ica=ica,
        excluded_components=excluded,
        used_eog_reference=used_eog,
    )


def _frontal_bad_components(
    ica: "mne.preprocessing.ICA",
    raw_f: "mne.io.Raw",
    frontal_names: List[str],
    corr_threshold: float,
    kurt_threshold: float,
) -> Tuple[List[int], List[float]]:
    """双判据 ICA 眼电/眨眼成分检测（规避 MNE.find_bads_eog 的多通道异常）。

    判据：
      (1) 额极强相关 —— 成分源信号与 Fp1/Fp2 的最大 |Pearson 相关| > corr_threshold
          （眨眼在双侧额极产生强偏转）
      (2) 超高斯峭度 —— 源信号峭度 > kurt_threshold（≈5）
          （眨眼/眼电呈尖锐脉冲，峭度远高于平滑的神经振荡）

    两者同时满足才判为伪迹，可避免把「额极相关高但峭度低」的正常额叶神经
    成分（如额叶 alpha/theta）误删。

    Returns
    -------
    bad : 命中双判据的成分下标
    scores : 每个成分的峭度（用于排序截断，峭度越高越像伪迹）
    """
    from scipy.stats import kurtosis

    sources = ica.get_sources(raw_f).get_data()          # (n_components, n_samples)
    fron_data = raw_f.copy().pick(frontal_names).get_data()  # (n_frontal, n_samples)
    n_comp = sources.shape[0]
    bad: List[int] = []
    scores: List[float] = []
    for c in range(n_comp):
        # (1) 额极最大相关
        cors = []
        for e in range(fron_data.shape[0]):
            try:
                r = np.corrcoef(sources[c], fron_data[e])[0, 1]
            except Exception:
                r = 0.0
            cors.append(0.0 if np.isnan(r) else abs(r))
        max_corr = float(max(cors))
        # (2) 峭度
        try:
            k = float(kurtosis(sources[c]))
        except Exception:
            k = 0.0
        scores.append(k)
        if max_corr > corr_threshold and k > kurt_threshold:
            bad.append(c)
    return bad, scores


def preprocess_subject(
    eeg: np.ndarray,
    peripheral: Optional[np.ndarray] = None,
    sfreq: int = SFREQ,
    cfg: dict = PREPROC_CONFIG,
) -> Tuple[np.ndarray, dict]:
    """对单个被试全部 trial 执行预处理。

    Parameters
    ----------
    eeg : np.ndarray  shape (n_trials, 32, n_samples)
    peripheral : np.ndarray | None  shape (n_trials, 8, n_samples)

    Returns
    -------
    clean_eeg : np.ndarray  shape (n_trials, 32, n_samples)
    meta : dict  每个 trial 的 ICA 剔除信息等
    """
    n_trials = eeg.shape[0]
    clean_eeg = np.empty_like(eeg, dtype=np.float64)
    meta = {"n_trials": n_trials, "excluded_per_trial": [], "used_eog": []}

    for t in range(n_trials):
        eog = peripheral[t, _EOG_PERIPH_IDX, :] if peripheral is not None else None
        res = preprocess_trial(eeg[t], eog, sfreq, cfg)
        clean_eeg[t] = res.raw_clean.get_data()  # (32, n_samples)
        meta["excluded_per_trial"].append(res.excluded_components)
        meta["used_eog"].append(res.used_eog_reference)

    meta["total_excluded"] = sum(len(x) for x in meta["excluded_per_trial"])
    return clean_eeg, meta


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "阶段一_数据获取与准备")))
    from deap_loader import load_deap

    loader = load_deap()
    sd = loader.load_subject(1)
    res = preprocess_trial(sd.eeg[0], sd.peripheral[0, _EOG_PERIPH_IDX, :])
    print("clean EEG shape:", res.raw_clean.get_data().shape)
    print("excluded ICA components:", res.excluded_components)
