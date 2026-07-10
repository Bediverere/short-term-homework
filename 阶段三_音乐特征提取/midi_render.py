# -*- coding: utf-8 -*-
"""
MIDI -> WAV 渲染器
====================

DEAP 的刺激素材是 MIDI 文件（audio_stimuli_MIDI/exp_id_*.mid），
而非现成音频。Librosa 只能分析波形，因此本模块先把 MIDI 渲染为
单声道 WAV（22050 Hz），再交给 music_features.py 做声学特征提取。

设计要点
--------
* 不依赖 fluidsynth / soundfont（本机未安装），完全用 numpy 加法合成：
  每个音符 = 基频 + 若干谐波（含 ADSR 包络），符合乐音的谐波结构，
  因此 Chroma / 音高类特征依然准确。
* 多乐器：分别合成各 Instrument 的音符并叠加。
* 打击乐（is_drum）：用指数衰减的带通噪声脉冲近似，避免产生虚假音高。
* 输出归一化到 [-0.9, 0.9]，防止削波。

对外接口
--------
    y, sr, duration = midi_to_audio(path)
"""

from __future__ import annotations

import numpy as np
import pretty_midi

# 默认采样率：与 Librosa 常用 22050 一致，足够覆盖音乐基频与谐波
DEFAULT_SR = 22050

# 谐波幅度配比（基频 + 4 次泛音），模拟钢琴/合成器类音色
DEFAULT_HARMONICS = (1.0, 0.55, 0.32, 0.18, 0.09)

# ADSR 包络（占音符时长的比例）：attack / decay / sustain / release
DEFAULT_ADSR_RATIO = (0.08, 0.12, 0.70, 0.10)
DEFAULT_SUSTAIN_LEVEL = 0.75


def _synth_note(f0: float, length: int, sr: int, velocity: float,
                harmonics=DEFAULT_HARMONICS,
                adsr=DEFAULT_ADSR_RATIO, sustain_level=DEFAULT_SUSTAIN_LEVEL) -> np.ndarray:
    """合成单个乐音：谐波叠加 + ADSR 幅度包络。"""
    if length <= 1:
        return np.zeros(1, dtype=np.float64)
    t = np.arange(length) / sr
    sig = np.zeros(length, dtype=np.float64)
    for i, h in enumerate(harmonics):
        sig += h * np.sin(2.0 * np.pi * f0 * (i + 1) * t)
    # ADSR 包络
    a = max(1, int(adsr[0] * length))
    d = max(1, int(adsr[1] * length))
    r = max(1, int(adsr[3] * length))
    s = max(0, length - a - d - r)
    env = np.concatenate([
        np.linspace(0.0, 1.0, a),
        np.linspace(1.0, sustain_level, d),
        np.full(s, sustain_level),
        np.linspace(sustain_level, 0.0, r),
    ])[:length]
    return sig * env * velocity


def _synth_drum(f0: float, length: int, sr: int, velocity: float) -> np.ndarray:
    """合成单个打击乐：带通噪声脉冲 + 指数衰减。"""
    if length <= 1:
        return np.zeros(1, dtype=np.float64)
    noise = np.random.RandomState(int(f0 * 1000) % 2**31).randn(length)
    # 粗略带通：用差分近似高通，再用滑动平均近似低通
    hp = noise - np.concatenate([[0.0], noise[:-1]])
    t = np.arange(length) / sr
    decay = np.exp(-t * (30.0 + f0 / 20.0))
    return hp * decay * velocity * 0.6


def midi_to_audio(midi_path: str, sr: int = DEFAULT_SR,
                  harmonics=DEFAULT_HARMONICS) -> tuple[np.ndarray, int, float]:
    """将 MIDI 文件渲染为单声道波形。

    Returns
    -------
    y : np.ndarray(float32)   归一化后的波形
    sr : int                  采样率
    duration : float          音频时长（秒）
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    duration = float(pm.get_end_time())
    if duration <= 0:
        duration = 5.0
    n = int(duration * sr) + 1
    y = np.zeros(n, dtype=np.float64)

    for inst in pm.instruments:
        for note in inst.notes:
            start = int(note.start * sr)
            end = int(note.end * sr)
            if end <= start:
                end = start + 1
            seg_len = end - start
            f0 = float(pretty_midi.note_number_to_hz(note.pitch))
            vel = max(0.05, note.velocity / 127.0)
            if inst.is_drum:
                seg = _synth_drum(f0, seg_len, sr, vel)
            else:
                seg = _synth_note(f0, seg_len, sr, vel, harmonics=harmonics)
            # 防止越界
            if start + seg_len > n:
                seg = seg[: n - start]
            y[start: start + seg.shape[0]] += seg

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak * 0.9
    return y.astype(np.float32), sr, duration


def get_midi_tempo(midi_path: str) -> float:
    """读取 MIDI 文件内嵌的真实速度（BPM），比音频节拍检测更可靠。"""
    pm = pretty_midi.PrettyMIDI(midi_path)
    return float(pm.estimate_tempo())


def render_all(midi_dir: str, trial_ids, sr: int = DEFAULT_SR,
               out_dir: str | None = None) -> dict:
    """批量渲染一组 MIDI（exp_id_{id}.mid）。返回 {trial_id: (y, sr, duration)}。

    若 out_dir 不为 None，同时把 WAV 存为 out_dir/exp_id_{id}.wav（scipy）。
    """
    import os
    from scipy.io import wavfile

    results = {}
    for tid in trial_ids:
        path = os.path.join(midi_dir, f"exp_id_{tid}.mid")
        if not os.path.exists(path):
            print(f"  [warn] 找不到 {path}，跳过")
            continue
        y, sr_out, dur = midi_to_audio(path, sr=sr)
        results[tid] = (y, sr_out, dur)
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            wavfile.write(os.path.join(out_dir, f"exp_id_{tid}.wav"), sr_out, y)
    return results


if __name__ == "__main__":
    # 冒烟测试：渲染单个 MIDI 并打印时长
    import os
    d = r"D:\短学期\deap-dataset\audio_stimuli_MIDI"
    p = os.path.join(d, "exp_id_1.mid")
    y, sr, dur = midi_to_audio(p)
    print(f"exp_id_1: y.shape={y.shape}, sr={sr}, duration={dur:.2f}s, "
          f"peak={float(np.abs(y).max()):.3f}")
    print(f"exp_id_1 MIDI 速度估计: {get_midi_tempo(p):.1f} BPM")
