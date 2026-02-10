# 开发说明

## 架构概览
CLI 为线性管线，支持分步执行：
1) `diarize`: 规范化音频，执行说话人分离并按说话人拼接。
2) `asr`: 对每个说话人的音频做 Qwen3-ASR 并保存文本。
3) `synth`: 使用 VoxCPM（参考音频+文本）进行声音复刻和合成。
4) `preview`: 使用 `ffplay` 预览指定说话人或任意音频文件。

每次运行会生成：
- `audio.wav`：规范化的 16k 单声道音频
- `speakers.json`：说话人元数据与片段
- `speakers/`：每个说话人拼接后的音频
- `transcripts/`：每个说话人的识别文本
- `synth/`：合成输出

## 关键实现细节
- **音频规范化**：用 `ffmpeg` 统一转为 16k 单声道 PCM，保证模型输入一致。
- **说话人分离**：pyannote 输出对象在不同版本略有差异，代码会尝试 `speaker_diarization` / `annotation` 等属性以兼容。
- **拼接**：先切分临时 wav，再用 concat demuxer 拼接，避免重新编码损失。
- **ASR**：直接调用 `Qwen3ASRModel.from_pretrained` 与 `transcribe`。若未安装 FlashAttention2，会自动降级到 `sdpa`。
- **复刻**：VoxCPM 使用 `text` + `prompt_wav_path` + `prompt_text`。输出通过 `soundfile` 写文件。
- **依赖约束**：
  - `numba>=0.63.1` + `llvmlite>=0.46.0` 保证 Python 3.12 兼容。
  - `setuptools==69.5.1` 解决 `modelscope` 依赖的 `pkg_resources` 缺失问题。

## 常见问题
- **Pyannote token**：社区模型需要 HuggingFace token，推荐使用 `HUGGINGFACE_TOKEN` 环境变量。
- **FlashAttention2**：未安装会导致 ASR 初始化失败，可用 `--asr-attn sdpa` 或自动降级。
- **`pkg_resources` 缺失**：若 `setuptools` 过新会出错，需固定为 `69.5.1`。
- **CUDA/CPU**：`diarize` 用 `--device cuda`；ASR 用 `--asr-device` 指定。

## 下一步优化
1) **片段级预览**：支持播放任意时间片段，便于人工核验。
2) **交互式选择**：提供 TUI/交互界面浏览说话人并试听。
3) **并行 ASR**：对多说话人并行转写以提升速度。
4) **模型缓存**：长驻进程复用模型，减少加载时间。
5) **配置文件**：增加 `config.toml/yaml`，简化命令行参数。
6) **分离后处理**：合并短间隔、过滤极短片段，提升合成质量。
