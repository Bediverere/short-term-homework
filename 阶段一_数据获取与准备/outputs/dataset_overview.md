# DEAP 数据集概览（阶段一：数据获取与准备）

## 1. 数据集简介
- 名称：DEAP（Database for Emotion Analysis using Physiological signals）
- 模态：EEG（32 导）× 音乐视频刺激，附自我情绪评测
- 被试：32 人（编号 s01–s32，已去标识化）
- 每被试试验数：40（与 40 段音乐刺激一一对应）
- 采样率：128 Hz；单试验时长：63 s（8064 采样点）
- EEG 通道：32（标准 10-20 布局）；外围通道：8（GSR/Resp/Plet/Temp/EOG×3/EMG）
- 标签：valence / arousal / dominance / liking（自评 1–9）

## 2. 跨模态对齐索引
- 第 i 个 trial（0-based）对应音乐刺激 `audio_stimuli_MIDI/exp_id_{i+1:02d}.mid`（所有被试共享同一套刺激）
- 后续阶段将以 (被试, trial) 三元组对齐 EEG × 音乐特征 × 情绪标签。

## 3. 标签分布统计
- Valence：均值 3.47，标准差 2.01，范围 [0.00, 9.00]
- Arousal：均值 3.73，标准差 1.98，范围 [0.00, 9.00]
- 高唤醒高愉悦（V>5 且 A>5）试验占比：0.0%
- 低唤醒低愉悦（V<5 且 A<5）试验占比：54.6%

## 4. 数据完整性
- 全部通过校验：True
- 去标识化：True

## 5. 产出文件
- verification_report.json：逐被试校验明细
- data_split.json：训练/验证/测试划分
- fig_label_distribution.png / fig_sample_eeg.png / fig_channel_stats.png：可视化
