# Develop Notes

## Architecture
The CLI is a linear pipeline with optional per-step execution:
1) `diarize`: normalize audio, run pyannote diarization, then concat per speaker.
2) `asr`: run Qwen3-ASR on each speaker clip and store transcripts.
3) `synth`: run VoxCPM using (prompt wav + prompt text) to clone and synthesize.
4) `preview`: play a speaker clip or any audio file via `ffplay`.

Artifacts (per run directory):
- `audio.wav`: normalized source audio (16k mono).
- `speakers.json`: speaker metadata, segments, and per-speaker audio path.
- `speakers/`: concatenated clips for each speaker.
- `transcripts/`: per-speaker ASR output.
- `synth/`: generated audio per speaker.

## Key Implementation Details
- **Normalization**: `ffmpeg` converts any input (mp3/wav) to 16k mono PCM for consistent model input.
- **Diarization**: `pyannote` output types vary across versions. The code handles `DiarizeOutput`, `speaker_diarization`, or `annotation` by probing attributes before iterating.
- **Concatenation**: segments are extracted into temp wavs and concatenated with ffmpeg concat demuxer (safe=0).
- **ASR**: `Qwen3ASRModel.from_pretrained` is used directly. FlashAttention2 is optional; the code falls back to `sdpa` if not installed.
- **Voice Clone**: VoxCPM uses `text`, `prompt_wav_path`, `prompt_text` (matching current `voxcpm` signature). Output is written with `soundfile`.
- **Dependencies**:
  - `numba>=0.63.1` + `llvmlite>=0.46.0` ensure Python 3.12 compatibility.
  - `setuptools==69.5.1` is pinned because `modelscope` imports `pkg_resources`.

## Common Failure Modes
- **Pyannote token**: required for the community diarization pipeline. Use `HUGGINGFACE_TOKEN` env var.
- **FlashAttention2**: missing package causes ASR load errors; fallback to `sdpa` or pass `--asr-attn sdpa`.
- **`pkg_resources` missing**: if `setuptools` is too new, VoxCPM fails. Pin `setuptools==69.5.1`.
- **CUDA vs CPU**: pass `--device cuda` for pyannote; ASR uses `--asr-device` if needed.

## Next Optimizations
1) **Segment-level preview**: play a specific segment range for QA before concatenation.
2) **Speaker selection UX**: add an interactive TUI to browse speakers and preview audio.
3) **Batch ASR**: parallelize transcription per speaker for large recordings.
4) **Caching**: reuse model instances across multiple runs in a long-lived process.
5) **Config file**: add a `config.toml` or `yaml` to avoid long CLI flags.
6) **Diarization refinement**: merge short gaps and filter very short segments to reduce artifacts.
