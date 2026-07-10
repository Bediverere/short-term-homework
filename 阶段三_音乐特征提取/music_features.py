# -*- coding: utf-8 -*-
"""
音乐声学特征提取（阶段三核心）
================================

基于 Librosa 对渲染后的音乐波形做全方位声学特征提取：

1. 时域层面：BPM 节拍率、短时能量(RMS)、过零率(ZCR)
2. 时频域层面：MFCC(13)、频谱质心、频谱滚降度、Chroma 色度图(12)
3. 高层声学情绪效价：基于情感计算理论，由速度/能量->唤醒(Arousal)、
   由调式(大/小调)/明亮度/音高->效价(Valence)，输出 1-9 连续坐标
   （与 DEAP 的 valence/arousal 标签同尺度，便于阶段四跨模态对齐）

关键设计
--------
「音乐滑窗步长与 EEG 窗口严格一致」：EEG 为 128Hz、试次 63s。
本模块将高分辨率帧特征（~43Hz）聚合到与 EEG 同步的窗口
（默认 WIN_SEC=2.0s / HOP_SEC=1.0s），两类异构信号在时域上获得
统一锚定，直接支撑阶段四的 Cross-Attention 时序对准。
"""

from __future__ import annotations

import numpy as np
import librosa

N_MFCC = 13
N_CHROMA = 12

# 大调 / 自然小调音阶（相对根音的半音偏移）
_MAJOR = np.array([0, 2, 4, 5, 7, 9, 11])
_MINOR = np.array([0, 2, 3, 5, 7, 8, 10])


# ----------------------------------------------------------------------------
# 1) 帧级（高分辨率）特征
# ----------------------------------------------------------------------------
def extract_frame_features(y: np.ndarray, sr: int) -> dict:
    """提取 ~43Hz 帧级特征（hop=512 @22050Hz ≈ 23.2ms/帧）。"""
    S = librosa.stft(y, n_fft=2048, hop_length=512, win_length=2048)
    S_mag = np.abs(S)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=512)        # (13, T)
    chroma = librosa.feature.chroma_stft(S=S_mag, sr=sr, hop_length=512)          # (12, T)
    rms = librosa.feature.rms(y=y, hop_length=512, frame_length=2048)             # (1, T)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=512, frame_length=2048)  # (1, T)
    centroid = librosa.feature.spectral_centroid(S=S_mag, sr=sr, hop_length=512)  # (1, T)
    rolloff = librosa.feature.spectral_rolloff(S=S_mag, sr=sr, hop_length=512,
                                               roll_percent=0.85)                 # (1, T)

    frame_times = librosa.frames_to_time(
        np.arange(mfcc.shape[1]), sr=sr, hop_length=512)

    return {
        "mfcc": mfcc,            # (13, T)
        "chroma": chroma,        # (12, T)
        "rms": rms[0],           # (T,)
        "zcr": zcr[0],           # (T,)
        "centroid": centroid[0], # (T,)
        "rolloff": rolloff[0],   # (T,)
        "frame_times": frame_times,
    }


# ----------------------------------------------------------------------------
# 2) 高层声学情绪效价（Valence / Arousal）
# ----------------------------------------------------------------------------
def estimate_key_mode(mean_chroma: np.ndarray):
    """基于平均 Chroma 估计调式（大调/小调），返回 (root, mode, score)。"""
    c = mean_chroma / (mean_chroma.sum() + 1e-9)
    best = (0, "major", -1.0)
    for root in range(12):
        maj = np.zeros(12); maj[_MAJOR] = 1.0
        minr = np.zeros(12); minr[_MINOR] = 1.0
        # 循环移位到 root
        maj_r = np.roll(maj, root); minr_r = np.roll(minr, root)
        s_maj = float(np.dot(c, maj_r)); s_min = float(np.dot(c, minr_r))
        if s_maj > best[2]:
            best = (root, "major", s_maj)
        if s_min > best[2]:
            best = (root, "minor", s_min)
    return best


def compute_emotion(mean_chroma: np.ndarray, mean_centroid: float,
                    mean_rms: float, tempo: float, max_rms: float) -> tuple[float, float]:
    """基于情感计算理论估计 (Valence, Arousal)，均映射到 DEAP 同尺度 1-9。

    理论依据（音乐情感计算主流启发式）：
      * Arousal(唤醒) ~ 速度(BPM) + 能量(RMS)：快/响 -> 高唤醒
      * Valence(效价) ~ 调式(大调=正/小调=负) + 明亮度(频谱质心) + 音区高低
    """
    # --- 唤醒 Arousal ---
    tempo_norm = float(np.clip((tempo - 60.0) / (180.0 - 60.0), 0.0, 1.0))
    energy_norm = float(np.clip(mean_rms / (max_rms + 1e-9), 0.0, 1.0))
    arousal_norm = 0.6 * tempo_norm + 0.4 * energy_norm          # 0..1

    # --- 效价 Valence ---
    _, mode, _ = estimate_key_mode(mean_chroma)
    mode_sign = 1.0 if mode == "major" else -1.0
    # 明亮度：频谱质心对数归一化到 500-4000Hz
    bright = float(np.clip((np.log(max(mean_centroid, 1.0)) - np.log(500.0)) /
                           (np.log(4000.0) - np.log(500.0)), 0.0, 1.0))
    # 音区：chroma 高音区权重
    pitch_reg = float(np.dot(mean_chroma, np.arange(12)) /
                      (12.0 * (mean_chroma.sum() + 1e-9)))
    valence_raw = 0.5 * mode_sign + 0.3 * (bright - 0.5) * 2.0 + 0.2 * (pitch_reg - 0.5) * 2.0
    valence_norm = float(np.clip(valence_raw, -1.0, 1.0))

    # 映射到 1-9（与 DEAP 标签同尺度）
    valence_9 = 5.0 + 4.0 * valence_norm
    arousal_9 = 5.0 + 4.0 * (arousal_norm - 0.5) * 2.0
    return float(valence_9), float(arousal_9)


# ----------------------------------------------------------------------------
# 3) 与 EEG 窗口严格一致的滑窗聚合
# ----------------------------------------------------------------------------
def aggregate_to_windows(frames: dict, duration: float,
                         win_sec: float = 2.0, hop_sec: float = 1.0) -> dict:
    """把高分辨率帧特征聚合到 (win_sec, hop_sec) 窗口，与 EEG 窗口同步。"""
    ft = frames["frame_times"]
    n_win = max(1, int(np.ceil((duration - win_sec) / hop_sec)) + 1)
    win_starts = np.arange(n_win) * hop_sec
    win_ends = win_starts + win_sec

    feats = []
    win_val = []
    win_aro = []
    tempo = 0.0  # 由调用方回填
    for ws, we in zip(win_starts, win_ends):
        mask = (ft >= ws) & (ft < we)
        if mask.sum() == 0:
            # 该窗口无帧，用最近帧填充
            idx = int(np.clip(np.searchsorted(ft, ws), 0, len(ft) - 1))
            mask = np.zeros(len(ft), dtype=bool); mask[idx] = True
        rms_m = float(frames["rms"][mask].mean())
        zcr_m = float(frames["zcr"][mask].mean())
        cen_m = float(frames["centroid"][mask].mean())
        rol_m = float(frames["rolloff"][mask].mean())
        mfcc_m = frames["mfcc"][:, mask].mean(axis=1)          # (13,)
        chroma_m = frames["chroma"][:, mask].mean(axis=1)      # (12,)
        vec = np.concatenate([
            [rms_m, zcr_m, cen_m, rol_m], mfcc_m, chroma_m, [tempo, 0.0, 0.0]
        ])  # tempo/val/aro 占位，下面回填
        feats.append(vec)
        # 先存 chroma/centroid/energy 供情绪计算
        win_val.append((chroma_m, cen_m, rms_m))
    return {
        "n_win": n_win,
        "win_starts": win_starts,
        "win_ends": win_ends,
        "feat_matrix": np.array(feats),   # (n_win, 32) 含占位
        "_agg": win_val,                  # 用于情绪回填
    }


# ----------------------------------------------------------------------------
# 4) 总入口
# ----------------------------------------------------------------------------
def extract_music_features(y: np.ndarray, sr: int, duration: float,
                           win_sec: float = 2.0, hop_sec: float = 1.0,
                           midi_tempo: float | None = None) -> dict:
    """端到端：渲染波形 -> 帧特征 -> 窗口对齐特征 -> 情绪坐标。

    midi_tempo: 若提供（来自 MIDI 文件内嵌速度），优先作为 BPM，
                否则回退到 Librosa 音频节拍检测。
    """
    # 帧级特征
    frames = extract_frame_features(y, sr)

    # 全局情绪（用全曲平均）
    mean_chroma = frames["chroma"].mean(axis=1)
    mean_centroid = float(frames["centroid"].mean())
    mean_rms = float(frames["rms"].mean())
    max_rms = float(frames["rms"].max()) + 1e-9
    if midi_tempo is not None:
        tempo = float(midi_tempo)
    else:
        try:
            tempo = float(librosa.feature.rhythm.tempo(y=y, sr=sr)[0])
        except Exception:
            tempo = 120.0

    g_val, g_aro = compute_emotion(mean_chroma, mean_centroid, mean_rms, tempo, max_rms)

    # 窗口聚合 + 逐窗口情绪
    agg = aggregate_to_windows(frames, duration, win_sec, hop_sec)
    fm = agg["feat_matrix"].copy()
    for i, (chroma_m, cen_m, rms_m) in enumerate(agg["_agg"]):
        v, a = compute_emotion(chroma_m, cen_m, rms_m, tempo, max_rms)
        fm[i, -3] = tempo     # tempo
        fm[i, -2] = v         # valence
        fm[i, -1] = a         # arousal
    del agg["_agg"]

    # 特征维度说明
    feat_names = (["rms", "zcr", "centroid", "rolloff"]
                  + [f"mfcc_{i}" for i in range(N_MFCC)]
                  + [f"chroma_{i}" for i in range(N_CHROMA)]
                  + ["tempo", "valence", "arousal"])

    return {
        "frames": frames,
        "tempo": tempo,
        "global_valence": g_val,
        "global_arousal": g_aro,
        "win_feat": fm,                 # (n_win, 32)
        "win_starts": agg["win_starts"],
        "win_ends": agg["win_ends"],
        "win_valence": fm[:, -2],
        "win_arousal": fm[:, -1],
        "feat_names": feat_names,
        "win_sec": win_sec,
        "hop_sec": hop_sec,
    }


if __name__ == "__main__":
    import os
    from midi_render import midi_to_audio
    d = r"D:\短学期\deap-dataset\audio_stimuli_MIDI"
    y, sr, dur = midi_to_audio(os.path.join(d, "exp_id_1.mid"))
    res = extract_music_features(y, sr, dur)
    print(f"duration={dur:.1f}s  tempo={res['tempo']:.1f}BPM  "
          f"V={res['global_valence']:.2f} A={res['global_arousal']:.2f}")
    print(f"win_feat shape={res['win_feat'].shape}  (n_win, {len(res['feat_names'])})")
    print("feat_names:", res["feat_names"])
