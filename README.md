# EGO Video Compiler

把第一人称（Egocentric）长视频，编译成结构化、可直接用于训练的**操作片段数据集**。

输入一段几十分钟的工厂作业 / 家务 / 操作类长视频，输出逐片段的**实体标注 + 七模式物理常识问答**，用于 VLA（视觉-语言-动作）模型、世界模型或具身智能数据集的构建。

```
长视频 (mp4)  →  操作片段切分  →  证据帧抽取  →  实体标注  →  VQA 生成  →  结构化 JSON
```

---

## 目录

- [核心特性](#核心特性)
- [处理流程](#处理流程)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [三种输入方式](#三种输入方式)
- [命令行参数](#命令行参数)
- [可视化界面](#可视化界面)
- [输出格式](#输出格式)
- [配置说明](#配置说明)
- [目录结构](#目录结构)
- [常见问题](#常见问题)
- [实测数据](#实测数据)

---

## 核心特性

| 特性 | 说明 |
|---|---|
| **零成本切分** | 基于光流能量 + 自相关周期检测，纯 CPU 计算，不调用任何模型。可先看切分效果再决定是否付费编译 |
| **片段级抽帧** | 只抽取有效操作片段的帧（每段 4 帧：起始 / 峰时 / 中点 / 结束），相比全视频固定频率抽帧，成本降低一个数量级 |
| **逐段编译** | 每个片段独立走完「帧描述 → 实体融合 → VQA 生成 → 校验」，避免数百帧一次性输入导致模型输出退化 |
| **七模式问答** | 属性 / 空间 / 力学 / 推理 / 摘要 / 时序 / 轨迹，覆盖物理常识的不同维度 |
| **跨视频词表** | 实体命名落盘共享，保证同一数据集内「扳手」不会在不同视频里叫「银色工具」和「金属钳子」 |
| **幂等可续跑** | 帧图与中间结果都落盘缓存，中断后重跑自动跳过已完成部分 |
| **接口无关** | 走 OpenAI 兼容协议，DashScope / OpenAI / 本地 vLLM / Ollama 都能用，只改 `base_url` 和模型名 |
| **双入口** | 命令行批量处理 + Gradio 界面单视频交互式调试 |

---

## 处理流程

```
┌──────────────┐    ┌────────────────┐    ┌──────────────┐    ┌──────────────┐
│ ① 输入        │ →  │ ② 操作片段切分  │ →  │ ③ 证据帧抽取  │ →  │ ④ 语义编译    │
│ 本地/URL/目录 │    │ 光流+自相关周期 │    │ 每段4帧      │    │ 实体+七模式VQA│
└──────────────┘    └────────────────┘    └──────────────┘    └──────────────┘
                     纯CPU，零成本            OpenCV抽帧           VLM + LLM
```

1. **输入**：本地视频文件、视频目录、或网络 URL
2. **切分**：计算运动能量曲线 → 自适应阈值找活动段 → 自相关检测主周期 → 峰值锚定 + 谷底定界
3. **抽帧**：每个片段抽 4 帧，Laplacian 方差过滤模糊帧，落盘 960px JPEG
4. **编译**：VLM 描述每帧 → LLM 融合跨帧实体 → LLM 生成七类问答 → 校验器检查证据帧引用与格式 → 失败自动修复一轮

---

## 环境要求

- **Python** 3.9+
- **ffmpeg**：可选但强烈建议。未安装时自动回退到 `imageio-ffmpeg` 自带的二进制，只是解码慢一些

```bash
# Windows: 把 ffmpeg 放到 bin/ 目录，或装到系统 PATH
# macOS:   brew install ffmpeg
# Ubuntu:  sudo apt install ffmpeg
```

---

## 快速开始

### 1. 安装依赖

```bash
git clone https://github.com/Long-Miao-Git/egocentric-data-compiler.git
cd egocentric-data-compiler

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 配置模型密钥

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Key：

```ini
DASHSCOPE_API_KEY=sk-your-key-here
```

> 使用其他 OpenAI 兼容服务时，同时修改 `OPENAI_BASE_URL` 与 `config.yaml` 里的模型名。

### 3. 编译你的第一个视频

```bash
python compile.py --input ./your_video.mp4
```

产物在 `output/compiled/your_video.compiled.json`。

### 4. 想先零成本看看切分效果？

```bash
python compile.py --input ./your_video.mp4 --stage segment
```

只跑切分，不调用任何模型，直接产出 `output/compiled/your_video.segments.json`。
确认切分合理（片段数、边界）后再跑完整流程，避免白花钱。

---

## 三种输入方式

### 方式一：本地视频文件

```bash
python compile.py --input ./videos/demo.mp4
python compile.py --input "D:\data\ego_clip.mov"
```

支持格式：`.mp4` `.avi` `.mov` `.mkv` `.webm` `.m4v`

### 方式二：本地目录批量

```bash
# 只扫当前层
python compile.py --input ./videos/

# 递归扫描子目录
python compile.py --input ./videos/ --recursive
```

### 方式三：网络地址（从开源数据集或其他站点获取）

任何可直接下载的 `http/https` 视频地址都可以，工具会先下载到 `output/downloads/` 再编译：

```bash
python compile.py --url https://example.com/ego_clip.mp4
```

也可以一次给多个：

```bash
python compile.py \
  --url https://example.com/a.mp4 \
  --url https://example.com/b.mp4 \
  --input ./local_videos/
```

#### 从 HuggingFace / ModelScope 获取视频的典型用法

这两个平台的数据集多为 tar 分片打包，需要先下载解压再交给本工具：

```bash
# ---- HuggingFace（以 egocentric-10K 为例）----
pip install huggingface_hub
hf download builddotai/Egocentric-10K \
  --repo-type dataset \
  --include "factory_001/workers/worker_001/*" \
  --local-dir ./raw

# 逐个解压 tar
cd ./raw/factory_001/workers/worker_001
for f in *.tar; do tar -xf "$f"; done

# 交给编译器
python compile.py --input ./raw/factory_001/workers/worker_001/ -r
```

```bash
# ---- ModelScope（国内直连，免 token）----
pip install modelscope
modelscope download --dataset builddotai/Egocentric-10K \
  factory_001/workers/worker_001 --local_dir ./raw

# 后续同上：解压后 --input 指向解压目录
```

> **提示**：如果数据集提供了解压后的直链，也可以直接用 `--url` 逐个传入，省去手动解压。

---

## 命令行参数

```
python compile.py --help
```

### 输入

| 参数 | 简写 | 说明 |
|---|---|---|
| `--input` | `-i` | 本地视频文件或目录，可重复指定 |
| `--url` | `-u` | 网络视频地址（http/https），可重复指定 |
| `--recursive` | `-r` | 输入为目录时递归查找子目录 |

### 输出

| 参数 | 简写 | 默认 | 说明 |
|---|---|---|---|
| `--output` | `-o` | `./output` | 输出根目录 |
| `--force` | | 关 | 重跑已存在的切分 / 编译结果 |

### 算法参数（不传则读 `config.yaml`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--stage` | `all` | `all` 全流程 / `segment` 仅切分（免费）/ `annotate` 仅标注 |
| `--analysis-fps` | 5.0 | 运动曲线采样帧率。越高越准，越慢 |
| `--min-peak-ratio` | 0.65 | 峰值阈值比例。**调小切得更细，调大切得更粗** |
| `--frames-per-seg` | 4 | 每片段证据帧数。增加会显著提高成本 |
| `--clarity` | 60 | Laplacian 清晰度阈值，低于此值判为模糊帧丢弃。EGO 运动模糊场景可降到 40 |
| `--workers` | `-j` | 4 | 片段级并行线程数。受 API 并发配额限制 |

### 其他

| 参数 | 简写 | 说明 |
|---|---|---|
| `--config` | `-c` | 指定配置文件路径 |
| `--show-config` | | 打印当前生效的全部配置后退出，用于排查 |

### 常用组合

```bash
# 大批量、控制成本
python compile.py -i ./videos/ -r -j 8 --frames-per-seg 3

# 切分太碎，想合并
python compile.py -i ./videos/ --min-peak-ratio 0.8 --force

# 画面很糊（快速移动的 EGO 视角）
python compile.py -i ./videos/ --clarity 40

# 指定到独立数据集目录
python compile.py -i ./videos/ -o ./dataset_v1
```

---

## 可视化界面

```bash
python app.py
# 默认 http://127.0.0.1:7860，自动打开浏览器
```

参数：

```bash
python app.py --port 8080 --host 0.0.0.0   # 局域网可访问
python app.py --share                       # 生成公网临时链接
python app.py --output ./my_output          # 指定输出目录
```

界面包含五个标签页：

| 标签页 | 用途 |
|---|---|
| **编译流程** | 上传单个视频 → 调参 → 一键编译 → 下载 JSON |
| **视频理解** | 可视化查看每个片段的实体、证据帧、画面缩略图 |
| **VQA 预览** | 按模式筛选问答，检查质量 |
| **成本报告** | 本次调用的 token 用量与费用明细 |
| **词汇管理** | 查看 / 添加同义词 / 合并重复实体，修正跨视频命名 |

适合单条视频调参：先在界面上试切分阈值和抽帧数，满意后再用命令行批量跑。

---

## 输出格式

每个视频产出一个 JSON：`output/compiled/<视频名>.compiled.json`

```jsonc
{
  "video_name": "demo.mp4",
  "video_info": { "fps": 30.0, "total_frames": 35996, "width": 1920,
                  "height": 1080, "duration_sec": 1199.87 },

  // 切分结果：包含全部检测到的片段
  "segmentation": {
    "segment_count": 173,
    "period_sec": 2.0,        // 检测到的主操作周期
    "period_corr": 0.495,     // 周期自相关强度
    "effective_ratio": 0.9996,
    "segments": [
      { "index": 1, "start_sec": 0.2, "end_sec": 17.2, "duration_sec": 17.0,
        "peak_time_sec": 15.2, "peak_energy": 18.7, "flag": "long" }
    ],
    "warnings": []
  },

  // 编译单元：只含成功编译的片段，是训练数据主体
  "segment_units": [
    {
      "segment": { "index": 2, "start_sec": 17.2, "end_sec": 23.0, ... },
      "frames": [
        { "frame_id": "s002_f4", "timestamp": 22.7,
          "image_path": ".../frames_cache/demo_s002_f4.jpg", "clarity": 113.9 }
      ],
      "entities": [
        { "canonical_name": "手臂", "category": "other", "synonyms": [],
          "first_seen_frame": "s002_f4", "evidence_frames": ["s002_f4"],
          "attributes": { "color": "肤色，深色纹身", "size": "局部可见", "position": "右侧" } }
      ],
      "vqa_items": [
        { "mode": "attribute", "question": "手臂有什么显著的外观特征？",
          "answer": "手臂为肤色，带有深色纹身。",
          "evidence_frame_ids": ["s002_f4"], "reasoning": "", "difficulty": "easy" }
      ],
      "validation": { "status": "pass", "errors": [], "warnings": [], "repaired_count": 0 }
    }
  ],

  "stats": {
    "segments_total": 173, "segments_compiled": 114, "segments_failed": 0,
    "total_entities": 714, "total_vqa": 1274,
    "vqa_mode_distribution": { "attribute": 308, "spatial": 289, "mechanics": 135,
                               "reasoning": 164, "summary": 119, "temporal": 144,
                               "trajectory": 115 },
    "empty_segments": 0, "empty_rate": 0.0
  },

  "frame_filtering": { "extracted": 198, "filtered_by_clarity": 480 },
  "total_cost_cny": 1.27,
  "cost_records": [ { "module": "vlm_batch_001", "input_tokens": 1584,
                      "output_tokens": 1677, "estimated_cost_cny": 0.0017 } ]
}
```

### 七种问答模式

| 模式 | 考察的能力 | 示例 |
|---|---|---|
| `attribute` | 物体属性识别 | 这个容器是什么颜色？ |
| `spatial` | 空间关系 | 螺丝刀在工件的哪个方位？ |
| `mechanics` | 力学常识 | 为什么需要双手托住托盘底部？ |
| `reasoning` | 因果推理 | 为什么先松开卡扣再取件？ |
| `summary` | 片段概括 | 这一段在完成什么操作？ |
| `temporal` | 时序理解 | 拧紧螺丝发生在测量之前还是之后？ |
| `trajectory` | 运动轨迹 | 手部从哪个方向靠近目标？ |

### 读取示例

```python
import json
from pathlib import Path

data = json.loads(Path("output/compiled/demo.compiled.json").read_text(encoding="utf-8"))

# 只取高置信、非简单的问答
for unit in data["segment_units"]:
    if unit["validation"]["status"] != "pass":
        continue
    for qa in unit["vqa_items"]:
        if qa["difficulty"] in ("medium", "hard"):
            print(f"[{qa['mode']}] {qa['question']} -> {qa['answer']}")
```

> `image_path` 指向本地抽帧缓存。迁移数据集时路径会失效，请按 `timestamp` 从原视频重新抽帧，或保留 `frames_cache/` 目录一并迁移。

---

## 配置说明

所有参数集中在 `config.yaml`，也可被命令行覆盖。

```yaml
api:
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1  # 任意 OpenAI 兼容服务

paths:
  output_root: ""        # 留空用 ./output

sampling:
  analysis_fps: 5.0      # 切分采样率
  min_peak_ratio: 0.65   # 切分灵敏度
  frames_per_seg: 4      # 每段证据帧数
  min_clarity_threshold: 60
  workers: 4
  batch_size: 8          # 单次 VLM 调用最多几张图

vlm:
  model: qwen3.7-flash   # 换成任意视觉模型
  max_tokens: 8192
  temperature: 0.1

llm:
  model: qwen3.8-flash
  max_tokens: 8192
  temperature: 0.3
  enable_thinking: false

vqa:
  min_items_per_segment: 5
  max_items_per_segment: 25

cost:                    # 仅用于费用统计显示，按你的服务价格改
  vlm_input_per_1k: 0.0002
  vlm_output_per_1k: 0.0008
  llm_input_per_1k: 0.0008
  llm_output_per_1k: 0.0027
```

**优先级**：命令行参数 > 环境变量 > `config.yaml` > 内置默认值

查看当前实际生效的配置：

```bash
python compile.py --show-config
```

---

## 目录结构

```
ego-video-compiler/
├── compile.py               # 命令行入口（推荐）
├── app.py                   # Gradio 可视化界面
├── config.yaml              # 参数配置
├── requirements.txt
├── .env.example             # 密钥模板（复制为 .env）
├── src/
│   ├── config.py            # 配置加载
│   ├── segmenter.py         # 切分算法（光流 + 自相关周期）
│   ├── annotator.py         # 单视频编译主流程
│   ├── sampler.py           # 抽帧与视频信息
│   ├── vlm_client.py        # 视觉模型调用
│   ├── llm_client.py        # 语言模型调用（实体融合 / VQA / 修复）
│   ├── validator.py         # 产物校验
│   ├── schemas.py           # 数据结构定义
│   ├── vocab.py             # 跨视频实体词表
│   ├── cost_tracker.py      # token 与费用统计
│   ├── compiler.py          # 编译聚合
│   └── prompts/             # 提示词模板
│       ├── vlm_prompt.py
│       ├── fusion_prompt.py
│       ├── vqa_prompt.py
│       └── repair_prompt.py
├── bin/                     # ffmpeg 存放处（首次运行自动生成，已被 gitignore）
└── output/                  # 运行产物（已 gitignore）
    ├── compiled/            # ★ 最终产物
    ├── frames_cache/        # 抽帧缓存（可删）
    └── downloads/           # 网络视频落盘
```

---

## 常见问题

**Q：切出来的片段太少 / 太碎？**

调 `--min-peak-ratio`。调小（如 0.5）切得更细，调大（如 0.85）合并得更多。
建议先用 `--stage segment` 零成本反复试。

**Q：片段数量对了，但某几段边界不准？**

切分基于运动能量，静止但仍在操作的动作（如缓慢拧螺丝）能量低，可能被切掉。
可以调高 `--analysis-fps` 提高时间分辨率，或用 `--clarity` 放宽抽帧筛选保住证据帧。

**Q：编译出来有些片段实体和问答都是空的？**

说明该片段内模型没识别出有效操作。可能原因：
- 画面过暗或严重模糊
- 片段本身是过渡动作（走路、调整设备）
- 该片段证据帧太少

空片段会在 `stats.empty_segments` 里单独统计，训练前建议直接过滤。

**Q：怎么判断哪些问答质量高？**

看 `validation.status`（`pass` / `warning` / `fail`）和 `difficulty`。
实践上取 `status == "pass"` 且 `difficulty != "easy"` 的样本信息量最高。

**Q：能完全不联网、用本地模型吗？**

可以。把 `base_url` 指向本地服务（vLLM / Ollama / LM Studio），模型名改成对应的视觉模型即可。
注意：VLM 必须支持图片输入，否则帧描述阶段会失败。

**Q：成本大概多少？**

见下方实测数据。控制成本的主要手段是 `--frames-per-seg`（每段帧数）和 `--workers`（并发）。
先用小样本跑通再全量，是最省钱的做法。

**Q：中断了怎么办？**

直接重跑同一条命令。已完成的切分和编译会自动跳过，抽帧缓存也会复用。
需要强制重做某一步时加 `--force`。

---

## 实测数据

用本工具编译 egocentric-10K 数据集的 223 个工厂作业视频（单视频约 20 分钟，1080p / 30fps）：

| 指标 | 数值 |
|---|---|
| 视频数 | 223 |
| 操作片段 | 39,021 |
| 实体标注 | 281,949 |
| VQA 问答对 | 485,992 |
| 每视频均值 | 片段 175 / 实体 1264 / VQA 2179 |
| 单视频成本 | 约 ¥3.0（新档模型口径 ¥2.33） |
| 单片段耗时 | 约 30~40 秒（4 线程） |

问答模式分布均匀（属性 24% / 空间 23% / 推理 13% / 时序 12% / 力学 11% / 摘要 9% / 轨迹 8%），未出现单一模式主导。

---

## 许可

[MIT License](LICENSE)

## 致谢

切分算法参考了第一人称视频中「周期性操作」的视觉特征，问答模式设计参考了物理常识推理的相关工作。
