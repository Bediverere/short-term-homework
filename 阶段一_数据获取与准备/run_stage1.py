# -*- coding: utf-8 -*-
"""
run_stage1.py — 阶段一主流程：加载 → 校验 → 划分 → 概览报告 + 可视化
生成：
  outputs/dataset_overview.md         数据集说明与统计
  outputs/verification_report.json    逐被试校验明细（亦由 verify_dataset 生成）
  outputs/data_split.json             训练/验证/测试划分
  outputs/fig_label_distribution.png  标签(valence/arousal)分布与四象限
  outputs/fig_sample_eeg.png          单试次 EEG 多通道波形
  outputs/fig_channel_stats.png       各通道均值/标准差
  stage1_summary.md                   阶段一交付总结
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 配置中文字体（Windows 优先微软雅黑，回退到黑体/系统字体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from deap_loader import (
    DEAPLoader,
    EEG_CHANNELS,
    N_TRIALS,
    N_EEG,
    SFREQ,
    TRIAL_SECONDS,
)
from verify_dataset import verify
from split_dataset import split_by_subject, split_by_trial

ROOT = os.path.dirname(os.path.abspath(__file__))
DEAP_ROOT = os.path.normpath(os.path.join(ROOT, "..", "deap-dataset"))
OUT = os.path.join(ROOT, "outputs")


# ----------------------------------------------------------------------------
# 统计与可视化
# ----------------------------------------------------------------------------
def collect_label_stats(loader: DEAPLoader):
    all_v, all_a = [], []
    for sd in loader.iterate_subjects():
        all_v.append(sd.labels[:, 0])
        all_a.append(sd.labels[:, 1])
    return np.concatenate(all_v), np.concatenate(all_a)


def fig_label_distribution(V, A, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(V, bins=20, color="#4C72B0", alpha=0.8)
    axes[0].set_title("Valence 分布")
    axes[0].set_xlabel("valence (1-9)")
    axes[0].set_ylabel("trial 数")
    axes[1].hist(A, bins=20, color="#DD8452", alpha=0.8)
    axes[1].set_title("Arousal 分布")
    axes[1].set_xlabel("arousal (1-9)")
    axes[2].scatter(V, A, s=8, alpha=0.3, color="#55A868")
    axes[2].axvline(5, color="gray", ls="--", lw=0.8)
    axes[2].axhline(5, color="gray", ls="--", lw=0.8)
    axes[2].set_xlabel("valence")
    axes[2].set_ylabel("arousal")
    axes[2].set_title("Valence-Arousal 四象限 (阈值=5)")
    axes[2].set_xlim(0, 9)
    axes[2].set_ylim(0, 9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_sample_eeg(loader: DEAPLoader, path, n_ch=8, trial=0, subject=1):
    sd = loader.load_subject(subject)
    eeg = sd.eeg[trial]  # (32, 8064)
    pick = EEG_CHANNELS[:n_ch]
    t = np.arange(eeg.shape[1]) / SFREQ
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, ch in enumerate(pick):
        ax.plot(t, eeg[i] + i * 50, label=ch, lw=0.6)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("幅值 (μV, 偏移堆叠)")
    ax.set_title(f"被试 s{subject:02d} trial {trial} — 前 {n_ch} 通道 EEG 波形")
    ax.set_xlim(0, TRIAL_SECONDS)
    ax.legend(ncol=8, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_channel_stats(loader: DEAPLoader, path):
    ids = loader.list_subjects()[:8]
    means = np.zeros((len(ids), N_EEG))
    stds = np.zeros((len(ids), N_EEG))
    for k, sid in enumerate(ids):
        sd = loader.load_subject(sid)
        flat = sd.eeg.reshape(-1, N_EEG)
        means[k] = flat.mean(0)
        stds[k] = flat.std(0)
    m, s = means.mean(0), stds.mean(0)
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(N_EEG)
    ax.bar(x, m, color="#4C72B0", alpha=0.7, label="平均均值")
    ax.errorbar(x, m, yerr=s, fmt="none", ecolor="#C44E52", capsize=2, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(EEG_CHANNELS, rotation=90, fontsize=7)
    ax.set_title("各 EEG 通道平均信号水平与离散度（前 8 被试）")
    ax.set_ylabel("μV")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 报告生成
# ----------------------------------------------------------------------------
def build_overview_md(loader, rep, subj, trial, V, A):
    lines = []
    lines.append("# DEAP 数据集概览（阶段一：数据获取与准备）\n")
    lines.append("## 1. 数据集简介")
    lines.append("- 名称：DEAP（Database for Emotion Analysis using Physiological signals）")
    lines.append("- 模态：EEG（32 导）× 音乐视频刺激，附自我情绪评测")
    lines.append(f"- 被试：{rep['summary']['subjects_found']} 人（编号 s01–s32，已去标识化）")
    lines.append(f"- 每被试试验数：{N_TRIALS}（与 40 段音乐刺激一一对应）")
    lines.append(f"- 采样率：{SFREQ} Hz；单试验时长：{TRIAL_SECONDS:.0f} s（{SFREQ * int(TRIAL_SECONDS)} 采样点）")
    lines.append(f"- EEG 通道：{N_EEG}（标准 10-20 布局）；外围通道：8（GSR/Resp/Plet/Temp/EOG×3/EMG）")
    lines.append("- 标签：valence / arousal / dominance / liking（自评 1–9）\n")
    lines.append("## 2. 跨模态对齐索引")
    lines.append("- 第 i 个 trial（0-based）对应音乐刺激 `audio_stimuli_MIDI/exp_id_{i+1:02d}.mid`（所有被试共享同一套刺激）")
    lines.append("- 后续阶段将以 (被试, trial) 三元组对齐 EEG × 音乐特征 × 情绪标签。\n")
    lines.append("## 3. 标签分布统计")
    lines.append(f"- Valence：均值 {V.mean():.2f}，标准差 {V.std():.2f}，范围 [{V.min():.2f}, {V.max():.2f}]")
    lines.append(f"- Arousal：均值 {A.mean():.2f}，标准差 {A.std():.2f}，范围 [{A.min():.2f}, {A.max():.2f}]")
    lines.append(f"- 高唤醒高愉悦（V>5 且 A>5）试验占比：{100 * ((V > 5) & (A > 5)).mean():.1f}%")
    lines.append(f"- 低唤醒低愉悦（V<5 且 A<5）试验占比：{100 * ((V < 5) & (A < 5)).mean():.1f}%\n")
    lines.append("## 4. 数据完整性")
    lines.append(f"- 全部通过校验：{rep['summary']['all_valid']}")
    lines.append(f"- 去标识化：{rep.get('deidentified')}\n")
    lines.append("## 5. 产出文件")
    lines.append("- verification_report.json：逐被试校验明细")
    lines.append("- data_split.json：训练/验证/测试划分")
    lines.append("- fig_label_distribution.png / fig_sample_eeg.png / fig_channel_stats.png：可视化")
    return "\n".join(lines) + "\n"


def build_summary_md(rep, subj, trial):
    lines = []
    lines.append("# 阶段一交付总结：数据获取与准备\n")
    lines.append(f"- 数据集：DEAP（data_preprocessed_python）")
    lines.append(f"- 被试数：{rep['summary']['subjects_found']} / 32")
    lines.append(f"- 校验结果：{'全部通过' if rep['summary']['all_valid'] else '存在异常'}")
    lines.append(f"- 去标识化：{rep.get('deidentified')}\n")
    lines.append("## 数据集划分")
    lines.append(f"- 被试级（推荐，跨被试泛化）：训练 {len(subj['train'])} / 验证 {len(subj['val'])} / 测试 {len(subj['test'])} 被试")
    lines.append(f"- 试验级（被试内建模）：训练 {len(trial['train'])} / 验证 {len(trial['val'])} / 测试 {len(trial['test'])} trial\n")
    lines.append("## 交付物清单")
    lines.append("- deap_loader.py：可复用数据加载与访问接口")
    lines.append("- verify_dataset.py：完整性校验")
    lines.append("- split_dataset.py：训练/验证/测试划分")
    lines.append("- run_stage1.py：主流程与可视化")
    lines.append("- outputs/：校验报告、划分清单、概览与三张图")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    loader = DEAPLoader(DEAP_ROOT)

    print("[1/5] 校验数据集 ...")
    rep = verify(loader)
    with open(os.path.join(OUT, "verification_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"     被试 {rep['summary']['subjects_found']} 个, 全部通过={rep['summary']['all_valid']}")

    print("[2/5] 划分数据集 ...")
    subj = split_by_subject(loader)
    trial = split_by_trial(loader)
    split_out = {"subject_level": subj, "trial_level": trial}
    with open(os.path.join(OUT, "data_split.json"), "w", encoding="utf-8") as f:
        json.dump(split_out, f, ensure_ascii=False, indent=2)

    print("[3/5] 统计标签分布 ...")
    V, A = collect_label_stats(loader)

    print("[4/5] 生成可视化 ...")
    fig_label_distribution(V, A, os.path.join(OUT, "fig_label_distribution.png"))
    fig_sample_eeg(loader, os.path.join(OUT, "fig_sample_eeg.png"))
    fig_channel_stats(loader, os.path.join(OUT, "fig_channel_stats.png"))

    print("[5/5] 撰写概览报告 ...")
    overview = build_overview_md(loader, rep, subj, trial, V, A)
    with open(os.path.join(OUT, "dataset_overview.md"), "w", encoding="utf-8") as f:
        f.write(overview)
    summary = build_summary_md(rep, subj, trial)
    with open(os.path.join(ROOT, "stage1_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary)
    print("完成。产出见 outputs/ 与 stage1_summary.md")


if __name__ == "__main__":
    main()
