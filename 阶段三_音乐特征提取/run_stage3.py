# -*- coding: utf-8 -*-
"""
阶段三主流程：音乐特征提取与结构分析
=====================================

串联：MIDI 渲染 -> Librosa 声学特征 -> 与 EEG 同步的滑窗对齐 ->
可视化与交付报告。

产出（全部存于 阶段三_音乐特征提取/outputs/）：
  * features/exp_id_{t}.npz   每个刺激的逐窗口 32 维特征 + 对齐时间戳
  * music_features_meta.json  全量统计与情绪坐标
  * 6 张可视化 PNG
  * stage3_summary.md         阶段三交付报告
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 复用阶段一加载器读取 DEAP 标签（用于跨模态相关性分析）
_STAGE1 = os.path.normpath(os.path.join(os.getcwd(), "..", "阶段一_数据获取与准备"))
if _STAGE1 not in sys.path:
    sys.path.insert(0, _STAGE1)

from midi_render import midi_to_audio, get_midi_tempo
from music_features import extract_music_features

# ----------------------------- 配置 -----------------------------
BASE = r"D:\短学期"
MIDI_DIR = os.path.join(BASE, "deap-dataset", "audio_stimuli_MIDI")
OUT_DIR = os.path.join(os.getcwd(), "outputs")
FEAT_DIR = os.path.join(OUT_DIR, "features")
os.makedirs(FEAT_DIR, exist_ok=True)

WIN_SEC, HOP_SEC = 2.0, 1.0          # 与 EEG 窗口严格一致的滑窗
TRIAL_IDS = list(range(1, 41))       # DEAP 40 个刺激
REP_ID = 1                           # 用于单样本可视化的代表刺激


# ---------------------------------------------------------------------------
# 1) 批量提取
# ---------------------------------------------------------------------------
def run_extraction():
    print(">>> 阶段三：音乐特征提取（%d 个 MIDI 刺激）" % len(TRIAL_IDS))
    records = {}
    global_val = []
    global_aro = []
    tempos = []
    for i, tid in enumerate(TRIAL_IDS, 1):
        path = os.path.join(MIDI_DIR, f"exp_id_{tid}.mid")
        y, sr, dur = midi_to_audio(path)
        bpm = get_midi_tempo(path)
        res = extract_music_features(y, sr, dur, WIN_SEC, HOP_SEC, midi_tempo=bpm)

        # 保存每个刺激的逐窗口特征
        np.savez(
            os.path.join(FEAT_DIR, f"exp_id_{tid}.npz"),
            win_feat=res["win_feat"],          # (n_win, 32)
            win_starts=res["win_starts"],
            win_ends=res["win_ends"],
            tempo=res["tempo"],
            global_valence=res["global_valence"],
            global_arousal=res["global_arousal"],
            feat_names=np.array(res["feat_names"]),
        )
        records[tid] = {
            "duration": round(dur, 2),
            "n_win": int(res["win_feat"].shape[0]),
            "tempo": round(res["tempo"], 2),
            "valence": round(res["global_valence"], 3),
            "arousal": round(res["global_arousal"], 3),
        }
        global_val.append(res["global_valence"])
        global_aro.append(res["global_arousal"])
        tempos.append(bpm)
        if i % 10 == 0 or i == 1:
            print(f"  [{i:02d}/{len(TRIAL_IDS)}] exp_id_{tid}: "
                  f"{dur:5.1f}s BPM={res['tempo']:5.1f} V={res['global_valence']:.2f} "
                  f"A={res['global_arousal']:.2f} win={res['win_feat'].shape[0]}")

    global_val = np.array(global_val)
    global_aro = np.array(global_aro)
    tempos = np.array(tempos)
    meta = {
        "n_stimuli": len(TRIAL_IDS),
        "win_sec": WIN_SEC,
        "hop_sec": HOP_SEC,
        "feat_dim": 32,
        "feat_names": ["rms", "zcr", "centroid", "rolloff"]
                      + [f"mfcc_{i}" for i in range(13)]
                      + [f"chroma_{i}" for i in range(12)]
                      + ["tempo", "valence", "arousal"],
        "mean_tempo": round(float(tempos.mean()), 2),
        "std_tempo": round(float(tempos.std()), 2),
        "mean_valence": round(float(global_val.mean()), 3),
        "std_valence": round(float(global_val.std()), 3),
        "mean_arousal": round(float(global_aro.mean()), 3),
        "std_arousal": round(float(global_aro.std()), 3),
        "records": records,
    }
    with open(os.path.join(OUT_DIR, "music_features_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(">>> 提取完成，已写入 music_features_meta.json 与 features/*.npz")
    return records, global_val, global_aro, tempos


# ---------------------------------------------------------------------------
# 2) 可视化
# ---------------------------------------------------------------------------
def plot_waveform_chroma_mfcc(tid=REP_ID):
    path = os.path.join(MIDI_DIR, f"exp_id_{tid}.mid")
    y, sr, dur = midi_to_audio(path)
    res = extract_music_features(y, sr, dur, WIN_SEC, HOP_SEC)
    fr = res["frames"]
    t = fr["frame_times"]

    # 波形
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.plot(np.linspace(0, dur, len(y)), y, color="#2c7fb8", linewidth=0.4)
    ax.set_title(f"代表刺激 exp_id_{tid} 渲染波形（MIDI→WAV，{sr}Hz）")
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("振幅")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_waveform.png"), dpi=130); plt.close(fig)

    # Chroma 色度图
    fig, ax = plt.subplots(figsize=(12, 3.4))
    im = ax.imshow(fr["chroma"], aspect="auto", origin="lower", cmap="viridis",
                   extent=[t[0], t[-1], 0, 12])
    ax.set_title(f"exp_id_{tid} Chroma 色度图（12 音级随时间）")
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("音级")
    fig.colorbar(im, ax=ax, label="强度")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_chroma.png"), dpi=130); plt.close(fig)

    # MFCC
    fig, ax = plt.subplots(figsize=(12, 3.4))
    im = ax.imshow(fr["mfcc"], aspect="auto", origin="lower", cmap="magma",
                   extent=[t[0], t[-1], 0, 13])
    ax.set_title(f"exp_id_{tid} MFCC 倒谱系数（13 维随时间）")
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("MFCC 维")
    fig.colorbar(im, ax=ax, label="系数")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_mfcc.png"), dpi=130); plt.close(fig)
    print("  [ok] fig_waveform / fig_chroma / fig_mfcc")


def plot_emotion_scatter(val, aro, tempos):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sc = ax.scatter(val, aro, c=tempos, cmap="plasma", s=70, edgecolor="k", linewidth=0.5)
    # 四象限分隔（DEAP 1-9 中心为 5）
    ax.axhline(5, color="gray", ls="--", lw=1)
    ax.axvline(5, color="gray", ls="--", lw=1)
    ax.text(5.4, 8.4, "高唤醒·高效价(兴奋)", fontsize=9, color="#c0392b")
    ax.text(1.6, 8.4, "高唤醒·低效价(紧张)", fontsize=9, color="#8e44ad")
    ax.text(1.6, 1.6, "低唤醒·低效价(悲伤)", fontsize=9, color="#2980b9")
    ax.text(5.4, 1.6, "低唤醒·高效价(平静)", fontsize=9, color="#27ae60")
    for i, tid in enumerate(range(1, len(val) + 1)):
        ax.annotate(str(tid), (val[i - 1], aro[i - 1]), fontsize=6,
                    xytext=(2, 2), textcoords="offset points")
    ax.set_xlim(1, 9); ax.set_ylim(1, 9)
    ax.set_xlabel("Valence 效价 (声学估计, 1-9)")
    ax.set_ylabel("Arousal 唤醒 (声学估计, 1-9)")
    ax.set_title("40 个音乐刺激的声学情绪分布 (Valence-Arousal)")
    fig.colorbar(sc, ax=ax, label="BPM 节拍率")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_emotion_scatter.png"), dpi=130); plt.close(fig)
    print("  [ok] fig_emotion_scatter")


def plot_feature_distributions(val, aro, tempos, records):
    # 提取每个刺激的 RMS / 频谱质心均值做分布
    rms_list, cen_list = [], []
    for tid in range(1, len(records) + 1):
        d = np.load(os.path.join(FEAT_DIR, f"exp_id_{tid}.npz"), allow_pickle=True)
        wf = d["win_feat"]
        rms_list.append(wf[:, 0].mean()); cen_list.append(wf[:, 2].mean())
    rms_list, cen_list = np.array(rms_list), np.array(cen_list)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    data = [tempos, val, aro, rms_list, cen_list,
            np.array([records[t]["duration"] for t in range(1, len(records) + 1)])]
    titles = ["BPM 节拍率", "Valence 效价", "Arousal 唤醒",
              "平均短时能量 RMS", "平均频谱质心", "刺激时长 (s)"]
    colors = ["#e67e22", "#c0392b", "#8e44ad", "#2c7fb8", "#16a085", "#34495e"]
    for ax, d, ti, c in zip(axes.flat, data, titles, colors):
        ax.hist(d, bins=10, color=c, edgecolor="white")
        ax.axvline(d.mean(), color="k", ls="--", lw=1.2, label=f"均值 {d.mean():.2f}")
        ax.set_title(ti, fontsize=11); ax.legend(fontsize=8)
    fig.suptitle("40 个音乐刺激的关键声学特征分布", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, "fig_feature_dist.png"), dpi=130); plt.close(fig)
    print("  [ok] fig_feature_dist")


# ---------------------------------------------------------------------------
# 3) 跨模态相关性：声学情绪 vs DEAP 人工评分
# ---------------------------------------------------------------------------
def crossmodal_analysis(val, aro):
    """比较「音乐声学情绪」与「DEAP 32 被试人工评分均值」。"""
    try:
        from deap_loader import load_deap
    except Exception as e:
        print(f"  [skip] 跨模态分析失败（无法加载 DEAP 标签）: {e}")
        return None
    loader = load_deap()
    n_sub = 32
    deap_val = []; deap_aro = []
    for tid in range(1, len(val) + 1):
        sub_vals = []; sub_aros = []
        for s in range(1, n_sub + 1):
            sd = loader.load_subject(s)
            sub_vals.append(sd.labels[tid - 1, 0])   # valence
            sub_aros.append(sd.labels[tid - 1, 1])   # arousal
        deap_val.append(np.mean(sub_vals)); deap_aro.append(np.mean(sub_aros))
    deap_val = np.array(deap_val); deap_aro = np.array(deap_aro)

    from scipy.stats import pearsonr
    rv = pearsonr(val, deap_val)[0]
    ra = pearsonr(aro, deap_aro)[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, a, b, lab, r in [
        (axes[0], val, deap_val, "Valence", rv),
        (axes[1], aro, deap_aro, "Arousal", ra),
    ]:
        ax.scatter(a, b, c="#2c7fb8", edgecolor="k", s=40)
        ax.set_xlabel(f"声学估计 {lab} (1-9)")
        ax.set_ylabel(f"DEAP 人工均值 {lab} (1-9)")
        ax.set_title(f"{lab}: 声学 vs 人工 (r={r:.2f})")
        ax.axhline(5, color="gray", ls=":", lw=0.8); ax.axvline(5, color="gray", ls=":", lw=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_crossmodal.png"), dpi=130); plt.close(fig)
    print(f"  [ok] fig_crossmodal (V r={rv:.2f}, A r={ra:.2f})")
    return {"pearson_v": round(float(rv), 3), "pearson_a": round(float(ra), 3),
            "deap_val_mean": round(float(deap_val.mean()), 3),
            "deap_aro_mean": round(float(deap_aro.mean()), 3)}


# ---------------------------------------------------------------------------
# 4) 报告
# ---------------------------------------------------------------------------
def write_report(meta, cross):
    lines = []
    lines.append("# 阶段三 · 音乐特征提取与结构分析 · 交付报告\n")
    lines.append("## 1. 数据来源")
    lines.append("- 刺激素材：`deap-dataset/audio_stimuli_MIDI/exp_id_1..40.mid`（DEAP 40 段音乐刺激）")
    lines.append("- 因 DEAP 提供的是 MIDI 而非音频，先用**纯 numpy 加法合成**渲染为 22050Hz 单声道 WAV（无需 fluidsynth/soundfont），再交 Librosa 分析。\n")
    lines.append("## 2. 方法与技术路线")
    lines.append("- **时域**：BPM 节拍率（`librosa.feature.rhythm.tempo`）、短时能量 RMS、过零率 ZCR")
    lines.append("- **时频域**：MFCC(13)（`librosa.feature.mfcc`）、频谱质心、频谱滚降度、Chroma 色度图(12)（`librosa.feature.chroma_stft`）")
    lines.append("- **高层声学情绪**：基于情感计算理论，速度/能量→唤醒(Arousal)，调式(大/小调)+明亮度+音区→效价(Valence)，输出 1-9 连续坐标（与 DEAP 标签同尺度）")
    lines.append(f"- **窗口对齐（关键设计）**：高分辨率帧特征(~43Hz, hop=512)聚合到 WIN_SEC={WIN_SEC}s / HOP_SEC={HOP_SEC}s 的窗口，与 EEG(128Hz, 63s 试次)窗口**严格一致**，两类异构信号在时域上统一锚定，直接支撑阶段四 Cross-Attention 时序对准。")
    lines.append(f"- **特征维度**：每个窗口 32 维 = [RMS, ZCR, centroid, rolloff] + 13×MFCC + 12×Chroma + [tempo, valence, arousal]。\n")
    lines.append("## 3. 全量统计（40 刺激）")
    lines.append(f"- 平均节拍率：{meta['mean_tempo']} ± {meta['std_tempo']} BPM")
    lines.append(f"- 声学效价 Valence：{meta['mean_valence']} ± {meta['std_valence']}（1-9）")
    lines.append(f"- 声学唤醒 Arousal：{meta['mean_arousal']} ± {meta['std_arousal']}（1-9）\n")
    if cross:
        lines.append("## 4. 跨模态相关性（声学情绪 vs DEAP 人工评分）")
        lines.append(f"- Valence  Pearson r = {cross['pearson_v']}")
        lines.append(f"- Arousal  Pearson r = {cross['pearson_a']}")
        lines.append(f"- DEAP 人工均值：Valence={cross['deap_val_mean']}, Arousal={cross['deap_aro_mean']}")
        lines.append("- 说明：声学情绪与人工评分存在相关趋势，但尚未达到强相关，正是阶段四跨模态融合模型要补强的部分（用 EEG 生理信号校正纯声学估计的偏差）。\n")
    lines.append("## 5. 交付物清单（outputs/）")
    lines.append("- `features/exp_id_{1..40}.npz`：每个刺激的逐窗口 32 维特征 + 对齐时间戳 + 情绪坐标")
    lines.append("- `music_features_meta.json`：全量统计")
    lines.append("- `fig_waveform.png` / `fig_chroma.png` / `fig_mfcc.png`：代表刺激波形/色度/倒谱")
    lines.append("- `fig_emotion_scatter.png`：40 刺激情绪 V-A 分布")
    lines.append("- `fig_feature_dist.png`：关键声学特征分布")
    lines.append("- `fig_crossmodal.png`：声学情绪 vs DEAP 人工评分相关性\n")
    lines.append("## 6. 下一步（阶段四）")
    lines.append("- 将本阶段音乐窗口特征(32维) 与阶段二 EEG 微分熵窗口特征(18维/通道) 在统一时间锚定下，输入 Cross-Attention Transformer 做跨模态时序对准与融合。")
    with open(os.path.join(OUT_DIR, "stage3_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(">>> 报告已写入 stage3_summary.md")


# ---------------------------------------------------------------------------
def main():
    records, val, aro, tempos = run_extraction()
    plot_waveform_chroma_mfcc(REP_ID)
    plot_emotion_scatter(val, aro, tempos)
    plot_feature_distributions(val, aro, tempos, records)
    cross = crossmodal_analysis(val, aro)
    meta = json.load(open(os.path.join(OUT_DIR, "music_features_meta.json"), encoding="utf-8"))
    if cross:
        meta["crossmodal"] = cross
        json.dump(meta, open(os.path.join(OUT_DIR, "music_features_meta.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    write_report(meta, cross)
    print("\n>>> 阶段三全部完成。")


if __name__ == "__main__":
    main()
