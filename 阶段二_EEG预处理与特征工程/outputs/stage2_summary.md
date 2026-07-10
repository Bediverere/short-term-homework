# 阶段二交付报告：EEG 信号预处理与特征工程

- 处理被试：[1, 2]
- 处理 trial 总数：80
- EEG 通道数：32，采样率：128 Hz
- ICA 眼电成分共剔除：84 个

## 预处理流水线
1. 带通滤波：0.5–45 Hz 双向零相位 FIR（firwin）
2. 工频陷波：50 Hz FIR 零相位
3. ICA 伪迹剔除：以 3 路 EOG 为参考自动检测眼电成分并剔除

## 特征工程
- **频段功率**（Welch PSD 积分）：delta/theta/alpha/beta/gamma
- **微分熵 DE** = 0.5·ln(2πe·σ²)，各频段
- **前额叶 Alpha 不对称性 FAA** = ln(α_Fp2) − ln(α_Fp1)：计划书要求作为效价(Valence)标志物（Russell 模型，正值→右侧 alpha 更强/退缩负性，负值→左侧更强/趋近正性）
- **时域统计**：均值/标准差/偏度/峰度/Hjorth(活动度·移动性·复杂性)/过零率
- 全部特征维度：band_power & DE → (trial, 32, 5)；time_feat → (trial, 32, 8)

## 全局统计
- 跨被试平均频段功率：{'delta': 0.426, 'theta': 3.622, 'alpha': 1.919, 'beta': 0.483, 'gamma': 0.214}
- 跨被试平均微分熵：{'delta': 0.4099, 'theta': 1.6483, 'alpha': 1.5958, 'beta': 0.8801, 'gamma': 0.2549}
- 跨被试平均 FAA（效价标志物）：-0.2675 （正值→右侧 alpha 更强/退缩负性；负值→左侧更强/趋近正性）

## 产出文件
- `features/sXX_features.npz`：每被试特征（band_power / de / time_feat / psd_mean / faa）
- `stage2_features_meta.json`：汇总元数据
- `fig_raw_vs_clean.png`：原始 vs 预处理时域对比
- `fig_ica_components.png`：ICA 成分头皮拓扑
- `fig_band_topomap.png`：频段功率头皮地形图
- `fig_psd.png`：滤波前后 PSD 对比
- `fig_de_distribution.png`：微分熵分布

> 注：默认演示处理前 2 名被试；将 `run_stage2.py` 中 `SUBJECTS` 改为 `list(range(1,33))` 即可扩展到全部 32 名被试。