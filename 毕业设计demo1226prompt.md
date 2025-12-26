## 开发 Prompt

你是资深 Python/CV 工程师。请在一个新目录 `drum2d/` 中实现一个可运行的最小项目，用于“2D 转鼓端面透明窗粉体截面”的视频合成、视觉检测与差分评估。要求：仅依赖 `numpy opencv-python scipy matplotlib tqdm`（可选 `pandas`），不要用深度学习框架。代码需可复现、可一键运行，输出 mp4 和评估图。按以下 spec 实现。

### 1) 总体目标与一键入口
实现 `main.py`，支持命令行：
- `python main.py simulate --out outputs/case_001 --seed 0 --seconds 8 --fps 30 --rpm 25 --fill 0.55 --noise 8 --blur 1.0 --glare 0.15 --regime auto`
- `python main.py detect --case outputs/case_001`
- `python main.py eval --case outputs/case_001`
- `python main.py demo --out outputs/demo_run --n_cases 12 --seconds 6`
  `demo` 要批量生成多工况数据→跑检测→做参数扫描→生成总报告 `outputs/demo_run/summary.png` 和 `summary.csv`（若不想引 pandas 就写 csv）。

目录结构建议（必须创建）：
```
drum2d/
  main.py
  simulate.py
  detect.py
  evaluate.py
  utils.py
  outputs/  (运行后生成)
```

### 2) 仿真合成器 simulate.py
实现 `simulate_case(out_dir, cfg)`，生成：
- `video.mp4`：合成视频（BGR，512x512，fps 可配）
- `gt.npz`：真值（numpy savez）
- `meta.json`：配置与可复现信息

#### 2.1 几何与画面
- 图像 W=512,H=512（可配）
- 圆筒截面窗口：圆心 `(cx,cy)=(W/2,H/2)`，半径 `R=220`（可配）
- 圆外为黑色；圆内为场景

#### 2.2 工况参数 cfg（必须支持）
- `seconds, fps, rpm, fill, noise_sigma, blur_sigma, glare_strength`
- `regime`: `"auto"|"slipping"|"cascading"|"cataracting"`
  - auto：根据 rpm 映射：<15 slipping; 15-35 cascading; >=35 cataracting

#### 2.3 自由表面真值
对每帧 t（秒）生成自由表面曲线 `y = f(x,t)`：
- 基准直线由动态安息角 `theta_gt(t)` 决定：
  - `theta0` 根据 fill 设一个合理值（例如 20°~40°范围）+ regime 偏置
  - `theta_gt(t) = theta0 + osc + small_noise`
  - `osc`: regime 控制振幅与频率（slipping 小、cataracting 大）
- 线：`y_line(x,t) = y0 + (x-cx)*tan(theta_gt(t))`
- 波动项：`delta(x,t) = A * sin(2π*k*x/W + phi(t)) + smooth_random(x,t)`
  - `smooth_random`：用随机噪声经 GaussianBlur 形成
- 事件脉冲（仅 cataracting，且可配置开关）：在若干 `t_i` 触发局部高斯形变，记录 `events`
- 曲线定义域：只取圆内 x 范围；将超出圆边界的点裁剪或丢弃，保证返回的 surface 点都在圆内

输出真值字段（gt.npz 必须包含）：
- `theta_gt` shape (T,) 单位：deg（明确）
- `surface_xy` shape (T,M,2) float32：每帧采样 M 个点（建议 M=200），像素坐标 (x,y)
- `regime_id` shape (T,) int（0/1/2）
- `events` shape (K,) float32（秒），若无则空数组
- `cfg_json` 存一份配置字符串也行（或放 meta.json）

#### 2.4 颗粒纹理与运动（只需“像”，不求物理）
生成粉体区域：圆内且 `y >= f(x,t)` 的区域为粉体（注意图像 y 向下）。
- 纹理：多尺度噪声 + 随机小圆点颗粒（1~3px）
- 运动：让靠近自由表面的薄层动得快
  - 计算每像素到表面的垂直距离近似 `dist = y - f(x,t)`（只在粉体区）
  - `v = v0 * exp(-dist/d_flow)`，方向近似沿表面切线（用 theta）
  - 用 `cv2.remap` 对上一帧粉体纹理做平移/warp，得到新纹理（保证速度场简单可实现：例如只做沿切线方向的平移量 `dx,dy` 随 dist 衰减）
- 最终图像：空气区亮一些、粉体区暗一些，叠加颗粒纹理。

#### 2.5 成像退化
- 高斯噪声：`noise_sigma`
- 模糊：`GaussianBlur` sigma=blur_sigma
- 反光：在圆内叠加一条弧形/椭圆高光（固定位置），强度 `glare_strength`
- 轻微亮度漂移（可选）

#### 2.6 视频写入
用 `cv2.VideoWriter` 写 `mp4v` 或 `avc1`（优先 mp4v，兼容性好）。
同时保存一张 `preview.png`：一帧上画出真值表面曲线与圆边界。

### 3) 检测器 detect.py（OpenCV 基线，要求稳）
实现 `detect_case(case_dir, params)`，输入 `video.mp4` 与 meta，输出：
- `detect.npz`：`theta_hat (T,) deg`, `line_kb (T,2)`, `surface_points (T,N,2)` 可选
- `detect_preview.mp4`：把检测线叠加在视频上，方便展示

检测流程（建议但可调整）：
1. 读取视频逐帧
2. 圆 ROI mask
3. 灰度 + blur（参数化）
4. 分割粉体/空气：Otsu 或自适应阈值（参数化）
5. 对每个 x（在圆内）找粉体最上沿 y，得到候选表面点
6. RANSAC 拟合直线，得到 `theta_hat = atan(k)`（deg）
7. 时序平滑（EMA 或 SavGol，参数化）

`params` 至少包含：
- `blur_ksize` (odd int)
- `thresh_method` in {otsu, adaptive}
- `ransac_resid_thresh`（像素）
- `ema_alpha`

### 4) 评估 evaluate.py
实现：
- `evaluate_case(case_dir)`：读取 `gt.npz` 与 `detect.npz`，输出 `report.png`（单 case）
  - 子图1：一帧叠加（真值曲线/检测线）
  - 子图2：theta_gt vs theta_hat 曲线
  - 子图3：误差 |hat-gt| 曲线 + MAE/RMSE/95%分位
  - 子图4：可选 roughness 指标对比（如果实现）
- `grid_search(root_out_dir, cases, param_grid)`：对一组 case 做参数扫描，输出：
  - `summary.csv`
  - `summary.png`：参数组合 vs 平均 MAE 的条形图或热力图

误差指标：
- `MAE_theta`, `RMSE_theta`, `P95_abs_err`

### 5) demo 模式 main.py demo
`python main.py demo --out outputs/demo_run --n_cases 12 --seconds 6`
行为：
1. 随机采样 n_cases 个 cfg（rpm/fill/noise/blur/glare）
2. 对每个 case 调 simulate、detect、evaluate
3. 做一个小的参数 grid search（例如 3x2x3 = 18 组）
4. 输出总览图 `summary.png`，并在命令行打印最优参数

### 6) 代码质量与可复现
- 每个模块函数要有 docstring
- meta.json 记录所有参数与 seed
- 运行失败要有清晰报错
- 默认参数能在普通笔记本 1-2 分钟内跑完 demo（n_cases=12, seconds=6, 512x512, fps=30）

### 7) 可视化要求（非常重要）
- 所有报告图用 matplotlib 保存为 png
- detect_preview.mp4 叠加信息：
  - 当前帧 theta_hat
  - （可选）theta_gt（从 gt 读，便于展示差分）
  - 画出圆边界、检测线、真值曲线（不同颜色）
- 颜色与字体清晰

### 8) 额外加分（可选但尽量做）
- 输出一个简单“流动性指标”：
  - `theta_std`（角度波动）
  - `surface_roughness_rms`（表面点到拟合线残差 RMS）
- 在 `evaluate_case` 里展示这些指标，并说明 regime/rpm 增大时指标变化趋势

请直接给出所有源码文件内容（main.py/simulate.py/detect.py/evaluate.py/utils.py），确保本地即可运行。并在最后给一段“运行说明”（命令示例）。我使用的包管理工具是uv。但我不希望你的运行过度依赖于uv。你可以使用uv管理虚拟环境但不可依赖运行。
