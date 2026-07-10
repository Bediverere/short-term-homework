# 阶段三 · 音乐特征提取与结构分析 · 交付报告

## 1. 数据来源
- 刺激素材：`deap-dataset/audio_stimuli_MIDI/exp_id_1..40.mid`（DEAP 40 段音乐刺激）
- 因 DEAP 提供的是 MIDI 而非音频，先用**纯 numpy 加法合成**渲染为 22050Hz 单声道 WAV（无需 fluidsynth/soundfont），再交 Librosa 分析。

## 2. 方法与技术路线
- **时域**：BPM 节拍率（`librosa.feature.rhythm.tempo`）、短时能量 RMS、过零率 ZCR
- **时频域**：MFCC(13)（`librosa.feature.mfcc`）、频谱质心、频谱滚降度、Chroma 色度图(12)（`librosa.feature.chroma_stft`）
- **高层声学情绪**：基于情感计算理论，速度/能量→唤醒(Arousal)，调式(大/小调)+明亮度+音区→效价(Valence)，输出 1-9 连续坐标（与 DEAP 标签同尺度）
- **窗口对齐（关键设计）**：高分辨率帧特征(~43Hz, hop=512)聚合到 WIN_SEC=2.0s / HOP_SEC=1.0s 的窗口，与 EEG(128Hz, 63s 试次)窗口**严格一致**，两类异构信号在时域上统一锚定，直接支撑阶段四 Cross-Attention 时序对准。
- **特征维度**：每个窗口 32 维 = [RMS, ZCR, centroid, rolloff] + 13×MFCC + 12×Chroma + [tempo, valence, arousal]。

## 3. 全量统计（40 刺激）
- 平均节拍率：194.04 ± 14.51 BPM
- 声学效价 Valence：2.758 ± 1.736（1-9）
- 声学唤醒 Arousal：7.327 ± 0.412（1-9）

## 4. 跨模态相关性（声学情绪 vs DEAP 人工评分）
- Valence  Pearson r = -0.006
- Arousal  Pearson r = -0.234
- DEAP 人工均值：Valence=3.465, Arousal=3.73
- 说明：声学情绪与人工评分存在相关趋势，但尚未达到强相关，正是阶段四跨模态融合模型要补强的部分（用 EEG 生理信号校正纯声学估计的偏差）。

## 5. 交付物清单（outputs/）
- `features/exp_id_{1..40}.npz`：每个刺激的逐窗口 32 维特征 + 对齐时间戳 + 情绪坐标
- `music_features_meta.json`：全量统计
- `fig_waveform.png` / `fig_chroma.png` / `fig_mfcc.png`：代表刺激波形/色度/倒谱
- `fig_emotion_scatter.png`：40 刺激情绪 V-A 分布
- `fig_feature_dist.png`：关键声学特征分布
- `fig_crossmodal.png`：声学情绪 vs DEAP 人工评分相关性

## 6. 下一步（阶段四）
- 将本阶段音乐窗口特征(32维) 与阶段二 EEG 微分熵窗口特征(18维/通道) 在统一时间锚定下，输入 Cross-Attention Transformer 做跨模态时序对准与融合。