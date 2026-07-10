# -*- coding: utf-8 -*-
"""
eeg_features.py
===============
阶段二（EEG 信号预处理与特征工程）之 —— EEG 特征工程

从预处理后的干净 EEG 提取面向「跨模态情感对齐」的多尺度特征：

  A. 频域特征
     - 频段功率 (Band Power)：delta/theta/alpha/beta/gamma
       采用 Welch 周期图估计 PSD，在五个频带内积分平均
     - 微分熵 (Differential Entropy, DE)：
       对各频带滤波信号的方差应用 DE = 0.5*ln(2πe·σ²)，
       是 DEAP/DREAMER 情感识别任务中验证有效的紧凑特征

  B. 时域特征（逐通道逐 trial）
     - 均值、标准差、偏度、峰度
     - Hjorth 活动度/移动性/复杂性
     - 过零率

所有特征均以 (n_trials, n_channels, ... ) 形式组织，可直接供阶段三
的 Cross-Attention Transformer 作为 EEG 侧 token 序列使用。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

# 标准频段定义（Hz）
FREQ_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
BAND_NAMES = list(FREQ_BANDS.keys())

TIME_FEATURE_NAMES = [
    "mean", "std", "skewness", "kurtosis",
    "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
    "zero_crossing_rate",
]


# ---------------------------------------------------------------------------
# A. 频域特征
# ---------------------------------------------------------------------------
def _band_power_psd(psd: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """在给定频带内对 PSD 积分平均。psd: (n_ch, n_freq)。"""
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return np.zeros(psd.shape[0])
    return psd[:, mask].mean(axis=1)


def compute_band_power(
    data: np.ndarray, sfreq: int, bands: dict = FREQ_BANDS
) -> np.ndarray:
    """计算频段功率。

    Parameters
    ----------
    data : np.ndarray  shape (n_ch, n_samples)
    Returns
    -------
    bp : np.ndarray  shape (n_ch, n_bands)
    """
    nperseg = min(int(sfreq * 2), data.shape[1] // 2)
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=1)
    bp = np.stack(
        [_band_power_psd(psd, freqs, lo, hi) for (lo, hi) in bands.values()],
        axis=1,
    )
    return bp  # (n_ch, n_bands)


def compute_differential_entropy(band_power: np.ndarray) -> np.ndarray:
    """由频段功率估算微分熵：DE = 0.5 * ln(2πe · power)。

    Parameters
    ----------
    band_power : np.ndarray  shape (n_ch, n_bands)
    Returns
    -------
    de : np.ndarray  shape (n_ch, n_bands)
    """
    eps = 1e-12
    return 0.5 * np.log(2 * np.pi * np.e * (band_power + eps))


def compute_faa(band_power: np.ndarray, left_idx: int, right_idx: int,
                band: str = "alpha", eps: float = 1e-12) -> float:
    """前额叶 Alpha 不对称性 (Frontal Alpha Asymmetry, FAA)。

    计划书要求「计算前额叶 Alpha 不对称性（FAA）作为效价(Valence)标志物」。
    以 Russell 情感模型，FAA = ln(α_R) − ln(α_L)（右减左）：正值表示右侧前额
    叶 alpha 更强，通常与负性/退缩情绪相关，左侧更强则与趋近/正性情绪相关。

    Parameters
    ----------
    band_power : np.ndarray  shape (n_ch, n_bands)  频段功率（或 DE，二者对数差等价）
    left_idx, right_idx : int   左/右前额叶通道索引（如 Fp1 / Fp2）
    band : str  用于计算不对称性的频段，默认 alpha
    Returns
    -------
    faa : float  该 trial 的前额叶 alpha 不对称性
    """
    bi = BAND_NAMES.index(band)
    rp = float(band_power[right_idx, bi]) + eps
    lp = float(band_power[left_idx, bi]) + eps
    return float(np.log(rp) - np.log(lp))


# ---------------------------------------------------------------------------
# B. 时域特征
# ---------------------------------------------------------------------------
def _hjorth(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hjorth 参数：活动度、移动性、复杂性。data: (n_ch, n_samples)。"""
    diff1 = np.diff(data, axis=1)
    diff2 = np.diff(diff1, axis=1)
    var0 = data.var(axis=1)
    var1 = diff1.var(axis=1)
    var2 = diff2.var(axis=1)
    activity = var0
    mobility = np.sqrt(var1 / (var0 + 1e-12))
    complexity = np.sqrt((var2 / (var1 + 1e-12)) / (mobility + 1e-12))
    return activity, mobility, complexity


def compute_time_features(data: np.ndarray) -> np.ndarray:
    """逐通道时域统计。data: (n_ch, n_samples) -> (n_ch, 8)。"""
    n_ch = data.shape[0]
    mean = data.mean(axis=1)
    std = data.std(axis=1)
    # 偏度 / 峰度（Fisher 定义）
    mu = mean.reshape(-1, 1)
    s = std.reshape(-1, 1)
    z = (data - mu) / (s + 1e-12)
    skewness = np.mean(z ** 3, axis=1)
    kurt = np.mean(z ** 4, axis=1) - 3.0
    act, mob, comp = _hjorth(data)
    # 过零率
    signs = np.sign(data)
    zc = np.mean((signs[:, 1:] * signs[:, :-1]) < 0, axis=1)
    out = np.stack([mean, std, skewness, kurt, act, mob, comp, zc], axis=1)
    return out  # (n_ch, 8)


# ---------------------------------------------------------------------------
# 汇总：单 trial 特征提取
# ---------------------------------------------------------------------------
def extract_trial_features(clean_data: np.ndarray, sfreq: int,
                          frontal_left: int = None, frontal_right: int = None) -> dict:
    """对单个 trial 的干净 EEG (n_ch, n_samples) 提取全部特征。

    Parameters
    ----------
    frontal_left, frontal_right : int, optional
      左/右前额叶通道索引（如 Fp1 / Fp2）。若同时提供，额外计算
      前额叶 Alpha 不对称性 FAA 作为效价标志物。
    Returns
    -------
    dict:
      band_power   : (n_ch, 5)
      de           : (n_ch, 5)
      time_feat    : (n_ch, 8)
      psd_freqs    : (n_freq,)
      psd          : (n_ch, n_freq)   该 trial 的 PSD（供地形图/可视化）
      faa          : float, optional  前额叶 Alpha 不对称性（仅当提供 frontal 索引时）
    """
    nperseg = min(int(sfreq * 2), clean_data.shape[1] // 2)
    freqs, psd = welch(clean_data, fs=sfreq, nperseg=nperseg, axis=1)
    bp = compute_band_power(clean_data, sfreq)
    de = compute_differential_entropy(bp)
    tf = compute_time_features(clean_data)
    out = {
        "band_power": bp,
        "de": de,
        "time_feat": tf,
        "psd_freqs": freqs,
        "psd": psd,
    }
    if frontal_left is not None and frontal_right is not None:
        out["faa"] = compute_faa(bp, frontal_left, frontal_right, band="alpha")
    return out


def extract_subject_features(clean_eeg: np.ndarray, sfreq: int,
                            frontal_left: int = None, frontal_right: int = None) -> dict:
    """对单个被试全部 trial 提取特征。

    Parameters
    ----------
    clean_eeg : np.ndarray  shape (n_trials, n_ch, n_samples)
    frontal_left, frontal_right : int, optional
      左/右前额叶通道索引；若提供，额外返回逐 trial 的 FAA 序列。
    Returns
    -------
    dict:
      band_power : (n_trials, n_ch, 5)
      de         : (n_trials, n_ch, 5)
      time_feat  : (n_trials, n_ch, 8)
      psd_mean   : (n_ch, n_freq)   跨 trial 平均 PSD（供地形图）
      psd_freqs  : (n_freq,)
      faa        : (n_trials,), optional  逐 trial 前额叶 Alpha 不对称性
    """
    n_trials = clean_eeg.shape[0]
    bp_list, de_list, tf_list, psd_list, freqs = [], [], [], [], None
    faa_list = [] if (frontal_left is not None and frontal_right is not None) else None
    for t in range(n_trials):
        f = extract_trial_features(clean_eeg[t], sfreq, frontal_left, frontal_right)
        bp_list.append(f["band_power"])
        de_list.append(f["de"])
        tf_list.append(f["time_feat"])
        psd_list.append(f["psd"])
        freqs = f["psd_freqs"]
        if faa_list is not None:
            faa_list.append(f["faa"])

    bp = np.stack(bp_list, axis=0)      # (n_trials, n_ch, 5)
    de = np.stack(de_list, axis=0)      # (n_trials, n_ch, 5)
    tf = np.stack(tf_list, axis=0)      # (n_trials, n_ch, 8)
    psd_stack = np.stack(psd_list, axis=0)  # (n_trials, n_ch, n_freq)
    psd_mean = psd_stack.mean(axis=0)   # (n_ch, n_freq)

    out = {
        "band_power": bp,
        "de": de,
        "time_feat": tf,
        "psd_mean": psd_mean,
        "psd_freqs": freqs,
    }
    if faa_list is not None:
        out["faa"] = np.array(faa_list, dtype=np.float64)  # (n_trials,)
    return out


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "阶段一_数据获取与准备")))
    from deap_loader import load_deap
    from eeg_preprocess import preprocess_trial, _EOG_PERIPH_IDX

    loader = load_deap()
    sd = loader.load_subject(1)
    res = preprocess_trial(sd.eeg[0], sd.peripheral[0, _EOG_PERIPH_IDX, :])
    feats = extract_trial_features(res.raw_clean.get_data(), sd.sfreq)
    print("band_power:", feats["band_power"].shape)
    print("de:", feats["de"].shape)
    print("time_feat:", feats["time_feat"].shape)
    print("alpha DE mean:", round(float(feats["de"][:, BAND_NAMES.index("alpha")].mean()), 4))
