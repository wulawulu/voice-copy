# 语音管线 (CLI)

这是一个本地离线 CLI，用于：
1) 使用 `pyannote/speaker-diarization-community-1` 做说话人分离
2) 将同一说话人的片段拼接成一个音频文件
3) 使用 Qwen3-ASR 对每个说话人做识别
4) 使用 VoxCPM1.5 复刻声音并合成目标文本

## 环境要求
- Python 3.12+
- PATH 中可用的 `ffmpeg`（用于音频规范化和预览）
- 本地模型运行时（Python 包 + 权重）

## 安装
```bash
pip install -e .
```

如果使用 `uv`：
```bash
uv sync
```

## 快速开始
### 1) 说话人分离 + 拼接
```bash
python main.py diarize \
  --input ./assets/audio.wav \
  --out-dir ./runs/run1 \
  --hf-token YOUR_HF_TOKEN \
  --device cuda
```

输出：
- `runs/run1/audio.wav`（规范化后的 16k 单声道 wav）
- `runs/run1/speakers/speaker_00.wav`, `speaker_01.wav`, ...
- `runs/run1/speakers.json`

### 2) 列出说话人
```bash
python main.py list --run-dir ./runs/run1
```

### 3) 每个说话人 ASR
```bash
python main.py asr \
  --run-dir ./runs/run1
```

可选参数：
- `--asr-attn sdpa`：不使用 FlashAttention2
- `--asr-model-path`：使用本地模型目录

### 4) 复刻并合成
```bash
python main.py synth \
  --run-dir ./runs/run1 \
  --speaker speaker_00 \
  --prompt "你好，请在下午三点开会"
```

### 一键全流程
```bash
python main.py run \
  --input ./assets/audio.wav \
  --out-dir ./runs/run1 \
  --hf-token YOUR_HF_TOKEN \
  --device cuda \
  --speaker speaker_00 \
  --prompt "你好，请在下午三点开会"
```

### 预览某个说话人音频
```bash
python main.py preview --run-dir ./runs/run1 --speaker speaker_00
```

## 说明
- `pyannote/speaker-diarization-community-1` 通常需要 HuggingFace token。
- MP3/WAV 都支持，会被统一规范化为 16k 单声道 WAV。
- 复刻效果依赖每个说话人的可用音频时长与纯净度。
- 离线模式可预先下载权重并设置 `HF_HUB_OFFLINE=1`。
- 未安装 FlashAttention2 时，ASR 会自动降级到 `sdpa`。
- VoxCPM 依赖 `modelscope`，需要 `setuptools==69.5.1` 以提供 `pkg_resources`。
