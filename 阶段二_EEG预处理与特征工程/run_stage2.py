# -*- coding: utf-8 -*-
"""
run_stage2.py
=============
阶段二主流程：串联 EEG 预处理 + 特征工程，生成可视化与交付报告。

默认对 SUBJECTS 中指定的被试（演示取前 2 名，可改 N 扩展到全部 32 名）
执行：
  1. 逐 trial 预处理（带通 + 陷波 + ICA 眼电剔除）
  2. 逐 trial 特征提取（频段功率 / 微分熵 / 时域统计）
  3. 保存特征（.npz）与元数据（.json）
  4. 生成 5 张可视化图（滤波前后对比 / ICA 成分 / 频段地形图 / PSD / 特征分布）
  5. 输出 stage2_summary.md 交付报告

运行：
  D:\\conda_envs\\pytorch\\python.exe run_stage2.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# --------------------------- 字体与绘图后端 -------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# --------------------------- 路径：复用阶段一加载器 -------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
STAGE1_DIR = os.path.normpath(os.path.join(HERE, "..", "阶段一_数据获取与准备"))
sys.path.insert(0, STAGE1_DIR)

from deap_loader import load_deap, EEG_CHANNELS, SFREQ  # noqa: E402
from eeg_preprocess import preprocess_subject, preprocess_trial, _EOG_PERIPH_IDX  # noqa: E402
from eeg_features import (  # noqa: E402
    extract_subject_features, BAND_NAMES, TIME_FEATURE_NAMES, FREQ_BANDS,
)

# 前额叶 Alpha 不对称性(FAA) 所用左右前额叶通道
FAA_LEFT = EEG_CHANNELS.index("Fp1")
FAA_RIGHT = EEG_CHANNELS.index("Fp2")

OUT_DIR = os.path.join(HERE, "outputs")
FEAT_DIR = os.path.join(OUT_DIR, "features")
os.makedirs(FEAT_DIR, exist_ok=True)

# 演示被试（可改为 range(1, 33) 跑全量 32 名）
SUBJECTS = [1, 2]


# ===========================================================================
# 可视化
# ===========================================================================
def plot_raw_comparison(raw_orig, raw_clean, ch_idx, out_path):
    """原始 vs 干净 EEG 时域对比（选取若干通道）。"""
    fig, axes = plt.subplots(len(ch_idx), 1, figsize=(11, 2.2 * len(ch_idx)), sharex=True)
    if len(ch_idx) == 1:
        axes = [axes]
    t = np.arange(raw_orig.get_data().shape[1]) / raw_orig.info["sfreq"]
    for ax, ci in zip(axes, ch_idx):
        ax.plot(t, raw_orig.get_data()[ci], color="#888888", lw=0.6, label="原始 (含伪迹/工频)")
        ax.plot(t, raw_clean.get_data()[ci], color="#1f77b4", lw=0.8, label="预处理后")
        ax.set_ylabel(EEG_CHANNELS[ci], fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(-80, 80)
    axes[-1].set_xlabel("时间 (s)")
    fig.suptitle("单 trial 原始 EEG 与预处理后对比（前/后：带通+陷波+ICA）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_ica_components(ica, info, out_path, n=6):
    """绘制 ICA 成分头皮拓扑（用于人工核验眼电成分）。"""
    try:
        n = min(n, ica.n_components_)
        fig = ica.plot_components(inst=info, picks=range(n), show=False, sphere="auto")
        if isinstance(fig, list):
            fig = fig[0]
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"  [warn] ICA 成分图生成失败: {e}")
        return False


def plot_band_topomap(band_power, info, out_path):
    """跨 trial 平均频段功率头皮地形图。band_power: (n_ch, 5)。"""
    try:
        import mne
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
        for i, name in enumerate(BAND_NAMES):
            im, _ = mne.viz.plot_topomap(
                band_power[:, i], info, axes=axes[i], show=False,
                cmap="Reds", sphere="auto",
            )
            axes[i].set_title(f"{name} 频段功率", fontsize=10)
        fig.suptitle("跨 trial 平均频段功率头皮地形图（s01）", fontsize=12)
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"  [warn] 频段地形图失败: {e}")
        return False


def plot_psd(freqs, psd_orig, psd_clean, ch_name, out_path):
    """单通道 PSD 对比（原始 vs 预处理后），叠加频段边界。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(freqs, psd_orig, color="#888888", lw=1.0, label="原始")
    ax.semilogy(freqs, psd_clean, color="#1f77b4", lw=1.0, label="预处理后")
    for (lo, hi) in FREQ_BANDS.values():
        ax.axvspan(lo, hi, color="orange", alpha=0.06)
    ax.axvline(50, color="red", ls="--", lw=0.8, label="50Hz 工频")
    ax.set_xlim(0, 60)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("功率谱密度 (V²/Hz)")
    ax.set_title(f"通道 {ch_name} PSD：滤波前后对比")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_feature_distribution(de_all, out_path):
    """微分熵（各频段）跨全部 trial×通道分布直方图。de_all: (N, 5)。"""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, name in enumerate(BAND_NAMES):
        axes[i].hist(de_all[:, i], bins=40, color="#2ca02c", alpha=0.8)
        axes[i].set_title(f"{name} 微分熵", fontsize=10)
        axes[i].set_xlabel("DE")
        axes[i].set_ylabel("频次")
    fig.suptitle("微分熵（DE）各频段分布（全部处理 trial×通道）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 主流程
# ===========================================================================
def main():
    loader = load_deap()
    print(f"阶段二开始：对被试 {SUBJECTS} 执行 EEG 预处理与特征工程")
    print(f"环境 SFREQ={SFREQ} Hz, EEG 通道数={len(EEG_CHANNELS)}")

    agg = {
        "subjects": [],
        "band_power_mean_per_subject": [],
        "de_mean_per_subject": [],
        "faa_mean_per_subject": [],
        "total_excluded_components": 0,
        "n_trials_total": 0,
    }
    de_collect = []  # 收集所有 DE 用于分布图

    for sid in SUBJECTS:
        sd = loader.load_subject(sid)
        print(f"\n>>> 被试 s{sid:02d}：{sd.n_trials} trials, eeg={sd.eeg.shape}")

        # 1) 预处理（传入完整 8 通道外围信号，由预处理函数内部切出 EOG 三路）
        eog = sd.peripheral if sd.peripheral.size else None
        clean_eeg, meta = preprocess_subject(sd.eeg, eog, sd.sfreq)
        if sid == SUBJECTS[0]:
            ref_clean, ref_sd = clean_eeg, sd
        print(f"    完成预处理 | ICA 共剔除成分数={meta['total_excluded']} | "
              f"每 trial 剔除={[len(x) for x in meta['excluded_per_trial']]}")

        # 2) 特征提取（含前额叶 Alpha 不对称性 FAA 效价标志物）
        feats = extract_subject_features(
            clean_eeg, sd.sfreq,
            frontal_left=FAA_LEFT, frontal_right=FAA_RIGHT,
        )

        # 3) 保存
        np.savez(
            os.path.join(FEAT_DIR, f"s{sid:02d}_features.npz"),
            band_power=feats["band_power"],     # (T,32,5)
            de=feats["de"],                     # (T,32,5)
            time_feat=feats["time_feat"],       # (T,32,8)
            psd_mean=feats["psd_mean"],         # (32,F)
            psd_freqs=feats["psd_freqs"],       # (F,)
            faa=feats["faa"],                   # (T,) 前额叶 Alpha 不对称性
        )

        # 4) 统计
        bp_mean = feats["band_power"].mean(axis=(0, 1))   # (5,)
        de_mean = feats["de"].mean(axis=(0, 1))           # (5,)
        agg["subjects"].append(sid)
        agg["band_power_mean_per_subject"].append(bp_mean.tolist())
        agg["de_mean_per_subject"].append(de_mean.tolist())
        agg["faa_mean_per_subject"].append(float(np.mean(feats["faa"])))
        agg["total_excluded_components"] += meta["total_excluded"]
        agg["n_trials_total"] += sd.n_trials
        de_collect.append(feats["de"].reshape(-1, len(BAND_NAMES)))
        print(f"    频段功率均值={dict(zip(BAND_NAMES, np.round(bp_mean,2)))}")

    de_all = np.vstack(de_collect)  # (N, 5)

    # ---------------- 参考 trial 可视化（s01 trial0）----------------------
    print("\n>>> 生成可视化 ...")
    sd0 = loader.load_subject(SUBJECTS[0])
    eog0 = sd0.peripheral[0, _EOG_PERIPH_IDX, :]
    res = preprocess_trial(sd0.eeg[0], eog0, sd0.sfreq)
    info = res.raw_clean.info

    # 原始 raw（未滤波）
    from eeg_preprocess import build_raw
    raw_orig = build_raw(sd0.eeg[0], eog0, sd0.sfreq).drop_channels(["EOG1", "EOG2", "EOG3"])
    plot_raw_comparison(raw_orig, res.raw_clean, ch_idx=[16, 0, 15, 7],  # Fz/Fp1/Pz/C3
                        out_path=os.path.join(OUT_DIR, "fig_raw_vs_clean.png"))
    print("  [ok] fig_raw_vs_clean.png")

    if plot_ica_components(res.ica, info, os.path.join(OUT_DIR, "fig_ica_components.png")):
        print("  [ok] fig_ica_components.png")

    # 频段地形图（参考被试 跨 trial 平均，复用循环已处理的 clean_eeg）
    feats0 = extract_subject_features(
        ref_clean, ref_sd.sfreq,
        frontal_left=FAA_LEFT, frontal_right=FAA_RIGHT,
    )
    bp_mean_topo = feats0["band_power"].mean(axis=0)  # (32,5)
    plot_band_topomap(bp_mean_topo, info, os.path.join(OUT_DIR, "fig_band_topomap.png"))
    print("  [ok] fig_band_topomap.png")

    # PSD 对比（取 Fz=ch16）
    f_orig, p_orig = _psd(sd0.eeg[0][16], sd0.sfreq)
    f_clean, p_clean = _psd(res.raw_clean.get_data()[16], sd0.sfreq)
    plot_psd(f_orig, p_orig, p_clean, "Fz", os.path.join(OUT_DIR, "fig_psd.png"))
    print("  [ok] fig_psd.png")

    plot_feature_distribution(de_all, os.path.join(OUT_DIR, "fig_de_distribution.png"))
    print("  [ok] fig_de_distribution.png")

    # ---------------- 元数据与报告 ----------------
    agg["band_power_global_mean"] = np.mean(agg["band_power_mean_per_subject"], axis=0).tolist()
    agg["de_global_mean"] = np.mean(agg["de_mean_per_subject"], axis=0).tolist()
    agg["faa_global_mean"] = float(np.mean(agg["faa_mean_per_subject"]))
    agg["band_names"] = BAND_NAMES
    agg["time_feature_names"] = TIME_FEATURE_NAMES
    agg["n_eeg_channels"] = len(EEG_CHANNELS)
    agg["sfreq"] = SFREQ
    agg["de_all_shape"] = list(de_all.shape)
    with open(os.path.join(OUT_DIR, "stage2_features_meta.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    _write_summary(agg, OUT_DIR)
    print("\n阶段二完成。交付物位于:", OUT_DIR)


def _psd(data1d, sfreq):
    from scipy.signal import welch
    nperseg = min(int(sfreq * 2), len(data1d) // 2)
    f, p = welch(data1d, fs=sfreq, nperseg=nperseg)
    return f, p


def _write_summary(agg, out_dir):
    bp = dict(zip(agg["band_names"], [round(x, 3) for x in agg["band_power_global_mean"]]))
    de = dict(zip(agg["band_names"], [round(x, 4) for x in agg["de_global_mean"]]))
    lines = [
        "# 阶段二交付报告：EEG 信号预处理与特征工程",
        "",
        f"- 处理被试：{agg['subjects']}",
        f"- 处理 trial 总数：{agg['n_trials_total']}",
        f"- EEG 通道数：{agg['n_eeg_channels']}，采样率：{agg['sfreq']} Hz",
        f"- ICA 眼电成分共剔除：{agg['total_excluded_components']} 个",
        "",
        "## 预处理流水线",
        "1. 带通滤波：0.5–45 Hz 双向零相位 FIR（firwin）",
        "2. 工频陷波：50 Hz FIR 零相位",
        "3. ICA 伪迹剔除：以 3 路 EOG 为参考自动检测眼电成分并剔除",
        "",
        "## 特征工程",
        "- **频段功率**（Welch PSD 积分）：delta/theta/alpha/beta/gamma",
        "- **微分熵 DE** = 0.5·ln(2πe·σ²)，各频段",
        "- **前额叶 Alpha 不对称性 FAA** = ln(α_Fp2) − ln(α_Fp1)：计划书要求作为效价(Valence)标志物"
        "（Russell 模型，正值→右侧 alpha 更强/退缩负性，负值→左侧更强/趋近正性）",
        "- **时域统计**：均值/标准差/偏度/峰度/Hjorth(活动度·移动性·复杂性)/过零率",
        f"- 全部特征维度：band_power & DE → (trial, 32, 5)；time_feat → (trial, 32, 8)",
        "",
        "## 全局统计",
        f"- 跨被试平均频段功率：{bp}",
        f"- 跨被试平均微分熵：{de}",
        f"- 跨被试平均 FAA（效价标志物）：{round(agg['faa_global_mean'], 4)} "
        f"（正值→右侧 alpha 更强/退缩负性；负值→左侧更强/趋近正性）",
        "",
        "## 产出文件",
        "- `features/sXX_features.npz`：每被试特征（band_power / de / time_feat / psd_mean / faa）",
        "- `stage2_features_meta.json`：汇总元数据",
        "- `fig_raw_vs_clean.png`：原始 vs 预处理时域对比",
        "- `fig_ica_components.png`：ICA 成分头皮拓扑",
        "- `fig_band_topomap.png`：频段功率头皮地形图",
        "- `fig_psd.png`：滤波前后 PSD 对比",
        "- `fig_de_distribution.png`：微分熵分布",
        "",
        "> 注：默认演示处理前 2 名被试；将 `run_stage2.py` 中 `SUBJECTS` 改为 "
        "`list(range(1,33))` 即可扩展到全部 32 名被试。",
    ]
    with open(os.path.join(out_dir, "stage2_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
