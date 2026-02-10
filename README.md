# Voice Pipeline (CLI)

This is a local, offline CLI to:
1) diarize a recording with `pyannote/speaker-diarization-community-1`
2) concat each speaker's segments into a single audio file
3) run ASR per speaker with Qwen3-ASR
4) clone voice / synthesize text per speaker with VoxCPM1.5

## Requirements
- Python 3.12+
- `ffmpeg` in PATH (for normalization and preview)
- Local model runtimes for ASR and cloning (Python packages + model weights)

## Install
```bash
pip install -e .
```

If you use `uv`:
```bash
uv sync
```

## Quick Start
### 1) Diarize + concat
```bash
python main.py diarize \
  --input ./assets/audio.wav \
  --out-dir ./runs/run1 \
  --hf-token YOUR_HF_TOKEN \
  --device cuda
```

Outputs:
- `runs/run1/audio.wav` (normalized to 16k mono wav)
- `runs/run1/speakers/speaker_00.wav`, `speaker_01.wav`, ...
- `runs/run1/speakers.json`

### 2) List speakers
```bash
python main.py list --run-dir ./runs/run1
```

### 3) ASR per speaker
```bash
python main.py asr \
  --run-dir ./runs/run1
```

Optional flags:
- `--asr-attn sdpa` to avoid FlashAttention2 dependency
- `--asr-model-path` to use a local model directory

### 4) Clone + synth
```bash
python main.py synth \
  --run-dir ./runs/run1 \
  --speaker speaker_00 \
  --prompt "你好，请在下午三点开会"
```

### All in one
```bash
python main.py run \
  --input ./assets/audio.wav \
  --out-dir ./runs/run1 \
  --hf-token YOUR_HF_TOKEN \
  --device cuda \
  --speaker speaker_00 \
  --prompt "你好，请在下午三点开会"
```

### Preview speaker audio
```bash
python main.py preview --run-dir ./runs/run1 --speaker speaker_00
```

## Notes
- `pyannote/speaker-diarization-community-1` typically requires a HuggingFace token.
- MP3/WAV input is supported; it is normalized to 16k mono WAV for consistent processing.
- For best cloning quality, ensure each speaker has enough clean audio after concatenation.
- To force offline mode, pre-download model weights and set `HF_HUB_OFFLINE=1`.
- If FlashAttention2 is not installed, the ASR step will automatically fall back to `sdpa`.
- VoxCPM relies on `modelscope` and requires `setuptools==69.5.1` for `pkg_resources`.
