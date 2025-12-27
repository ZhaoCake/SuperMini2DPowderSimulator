// 理论与计算方法汇总
// 摘自：机器视觉转鼓法检测粉体流动性预研究报告

#set page(
  paper: "a4",
  margin: (top: 22mm, bottom: 22mm, left: 22mm, right: 22mm),
)
#set text(
  font: "Maple Mono",
  size: 10.5pt,
  lang: "zh",
)
#set heading(numbering: "1.")
#set par(leading: 1.25em)

#align(center)[
  #text(size: 16pt, weight: "bold")[转鼓粉体视觉检测：理论与计算公式汇总]
]

#v(12pt)

= 1. 流态无量纲描述

工程上常用 Froude 数表征转鼓转速强度，用于界定流态（Slipping / Cascading / Cataracting）：
$ "Fr" = omega^2 R / g $
其中：
- $omega$：角速度（rad/s）
- $R$：转鼓半径（m）
- $g$：重力加速度

= 2. 核心几何特征计算

== 2.1 动态倾角 (Dynamic Angle of Repose)

对每一帧图像提取自由表面点集 ${(x_i, y_i)}_(i=1)^N$，对其进行鲁棒直线拟合：
$ y = k(t) dot x + b(t) $

动态倾角 $theta(t)$ 定义为拟合直线的斜率角度：
$ theta(t) = arctan(k(t)) dot 180 / pi $

统计指标：
- 平均倾角：$bar(theta) = "mean"_t thin theta(t)$
- 角度波动：$sigma_theta = "std"_t thin theta(t)$
- 变化率（帧差近似）：$dot(theta)(t) approx theta(t) - theta(t-1)$

== 2.2 表面形态粗糙度 (Surface Roughness)

=== 2.2.1 残差 RMS
基于拟合直线预测值 $hat(y)_i$，计算残差 $r_i = y_i - hat(y)_i$。表面粗糙度定义为残差的均方根（RMS）：
$ R_q(t) = sqrt(1/N sum_(i=1)^N r_i^2) $

=== 2.2.2 峰-谷幅值
$ A(t) = max_x y(x,t) - min_x y(x,t) $

= 3. 动力学特征计算

== 3.1 流动层厚度 (Flow Layer Thickness)
基于稠密光流场 $arrow(v)(x,y,t)$，在自由表面法向方向向下搜索，直到速度幅值衰减至阈值 $tau_v$ 以下。
$ h(t) = "mean"_x { min_d thick |arrow(v)| (x, y_s (x,t)+d, t) < tau_v } $
其中 $y_s(x,t)$ 为表面位置，$d$ 为向下深度（像素）。

== 3.2 事件检测 (Event Detection)
用于识别雪崩（Avalanche）或塌落（Slumping）事件的判据：
1.  **倾角突变**：$|dot(theta)(t)| > tau_theta$ 且持续时间 $> m$ 帧。
2.  **粗糙度尖峰**：$R_q(t)$ 超过特定阈值。
3.  **动能脉冲**：平均速度能量 $E_v(t)="mean"(|arrow(v)|)$ 出现脉冲。

= 4. 频域特征
对时序信号（如 $theta(t)$）进行 FFT 或 Welch 谱估计，提取：
- 主频 $f_"peak"$
- 能量占比（低频 vs 高频）

= 5. 评估指标 (Evaluation Metrics)

假设真值为 $theta_"gt"(t)$，算法估计值为 $theta_"hat"(t)$：

- **平均绝对误差 (MAE)**:
  $ "MAE" = "mean"_t thin |theta_"hat"(t) - theta_"gt"(t)| $

- **均方根误差 (RMSE)**:
  $ "RMSE" = sqrt("mean"_t thin (theta_"hat"(t) - theta_"gt"(t))^2) $

- **95% 分位误差 (P95)**:
  $ "P95" = "percentile"_(95)(|theta_"hat" - theta_"gt"|) $

= 6. DEM 真值构造理论

为了验证视觉算法，利用离散元仿真（DEM）数据构造真值：

1.  **自由表面真值 $y_"DEM"(x,t)$**：
    将粒子投影到 2D 截面，对 $x$ 轴分箱（Binning），取每个分箱内粒子 $y$ 坐标的上分位数（如 95%）。

2.  **倾角真值 $theta_"DEM"(t)$**：
    对 $y_"DEM"(x,t)$ 进行直线拟合得到。

3.  **流动层厚度真值 $h_"DEM"(t)$**：
    基于 DEM 速度场或剪切率场，定义 $|v| > tau$ 或 $dot(gamma) > tau$ 的区域厚度。
