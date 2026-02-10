#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from pyannote.audio import Pipeline


@dataclass(frozen=True)
class Segment:
    speaker: str
    start: float
    end: float


def die(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        die("ffmpeg not found in PATH.")


def ensure_ffplay() -> None:
    if shutil.which("ffplay") is None:
        die("ffplay not found in PATH.")


def run_cmd(cmd: List[str]) -> None:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        die(f"Command failed: {' '.join(cmd)}")


def normalize_audio(input_path: Path, output_path: Path) -> None:
    ensure_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    run_cmd(cmd)


def load_pipeline(model_id: str, token: Optional[str], device: Optional[str]) -> Pipeline:
    try:
        if token:
            pipeline = Pipeline.from_pretrained(model_id, token=token)
        else:
            pipeline = Pipeline.from_pretrained(model_id)
    except TypeError:
        if token:
            pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)
        else:
            pipeline = Pipeline.from_pretrained(model_id)
    if device:
        import torch

        pipeline.to(torch.device(device))
    return pipeline


def diarize_audio(
    audio_path: Path,
    model_id: str,
    token: Optional[str],
    device: Optional[str],
) -> List[Segment]:
    pipeline = load_pipeline(model_id, token, device)
    diarization = pipeline(str(audio_path))
    annotation = diarization
    if hasattr(diarization, "speaker_diarization"):
        annotation = diarization.speaker_diarization
    if hasattr(diarization, "annotation"):
        annotation = diarization.annotation
    segments: List[Segment] = []
    if hasattr(annotation, "itertracks"):
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(Segment(speaker=speaker, start=turn.start, end=turn.end))
    elif hasattr(annotation, "itersegments"):
        for turn in annotation.itersegments():
            speaker = getattr(turn, "label", "speaker")
            segments.append(Segment(speaker=speaker, start=turn.start, end=turn.end))
    else:
        die(f"Unsupported diarization output type: {type(diarization)}")
    return segments


def group_segments(segments: Iterable[Segment]) -> Dict[str, List[Segment]]:
    grouped: Dict[str, List[Segment]] = {}
    for seg in segments:
        grouped.setdefault(seg.speaker, []).append(seg)
    for speaker in grouped:
        grouped[speaker].sort(key=lambda s: s.start)
    return grouped


def concat_segments(
    audio_path: Path,
    segments: List[Segment],
    output_path: Path,
    temp_dir: Path,
) -> None:
    ensure_ffmpeg()
    temp_dir.mkdir(parents=True, exist_ok=True)
    part_files: List[Path] = []
    for idx, seg in enumerate(segments):
        part = temp_dir / f"seg_{idx:04d}.wav"
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-ss",
            f"{seg.start:.3f}",
            "-to",
            f"{seg.end:.3f}",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(part),
        ]
        run_cmd(cmd)
        part_files.append(part)

    concat_list = temp_dir / "concat.txt"
    with concat_list.open("w", encoding="utf-8") as f:
        for part in part_files:
            f.write(f"file '{part.as_posix()}'\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    run_cmd(cmd)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def diarize_cmd(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        die(f"Input not found: {input_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normalized = out_dir / "audio.wav"
    normalize_audio(input_path, normalized)

    token = args.hf_token or os.getenv("HUGGINGFACE_TOKEN")
    segments = diarize_audio(normalized, args.pipeline, token, args.device)
    if not segments:
        die("No segments produced by diarization.")

    grouped = group_segments(segments)

    speakers_dir = out_dir / "speakers"
    temp_root = out_dir / "tmp_segments"
    temp_root.mkdir(parents=True, exist_ok=True)
    speakers_info = []

    for idx, (speaker_label, segs) in enumerate(sorted(grouped.items())):
        speaker_id = f"speaker_{idx:02d}"
        with tempfile.TemporaryDirectory(dir=temp_root) as td:
            concat_segments(
                normalized,
                segs,
                speakers_dir / f"{speaker_id}.wav",
                Path(td),
            )
        duration = sum(s.end - s.start for s in segs)
        speakers_info.append(
            {
                "id": speaker_id,
                "label": speaker_label,
                "duration": duration,
                "segments": [{"start": s.start, "end": s.end} for s in segs],
                "audio_path": str((speakers_dir / f"{speaker_id}.wav").resolve()),
            }
        )

    write_json(out_dir / "segments.json", [seg.__dict__ for seg in segments])
    write_json(out_dir / "speakers.json", speakers_info)

    print(f"OK: diarization done. Speakers: {len(speakers_info)}")
    print(f"Speakers metadata: {out_dir / 'speakers.json'}")
    print(f"Speaker audio: {speakers_dir}")


def resolve_run_dir(path: Path) -> Path:
    if path.is_dir():
        return path
    die(f"Run dir not found: {path}")
    return path


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def list_cmd(args: argparse.Namespace) -> None:
    run_dir = resolve_run_dir(Path(args.run_dir))
    speakers_path = run_dir / "speakers.json"
    if not speakers_path.exists():
        die(f"Missing speakers.json in {run_dir}")
    speakers = read_json(speakers_path)
    transcripts_dir = run_dir / "transcripts"

    for info in speakers:
        duration = format_duration(info["duration"])
        transcript_path = transcripts_dir / f"{info['id']}.txt"
        transcript_status = "missing"
        if transcript_path.exists():
            text = transcript_path.read_text(encoding="utf-8").strip()
            transcript_status = f"{len(text)} chars"
        print(f"{info['id']} ({info['label']}): {duration}, transcript: {transcript_status}")


def parse_torch_dtype(name: Optional[str]):
    if not name:
        return None
    import torch

    if not hasattr(torch, name):
        die(f"Invalid torch dtype: {name}")
    return getattr(torch, name)


def build_qwen3_asr_model(args: argparse.Namespace):
    from qwen_asr import Qwen3ASRModel
    import torch

    device = args.asr_device
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    kwargs = {
        "dtype": parse_torch_dtype(args.asr_dtype),
        "device_map": device,
        "attn_implementation": args.asr_attn,
        "max_inference_batch_size": args.asr_batch_size,
        "max_new_tokens": args.asr_max_new_tokens,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if args.asr_local_files_only:
        kwargs["local_files_only"] = True

    model_id = args.asr_model_path or args.asr_model_id
    try:
        model = Qwen3ASRModel.from_pretrained(model_id, **kwargs)
    except ImportError as exc:
        # Fallback when flash-attn is not installed
        if kwargs.get("attn_implementation") == "flash_attention_2":
            kwargs["attn_implementation"] = "sdpa"
            print(
                "Warning: flash_attention_2 not available, falling back to sdpa.",
                file=sys.stderr,
            )
            model = Qwen3ASRModel.from_pretrained(model_id, **kwargs)
        else:
            raise exc
    except TypeError:
        kwargs.pop("local_files_only", None)
        model = Qwen3ASRModel.from_pretrained(model_id, **kwargs)
    return model


def asr_cmd(args: argparse.Namespace) -> None:
    run_dir = resolve_run_dir(Path(args.run_dir))
    speakers_path = run_dir / "speakers.json"
    if not speakers_path.exists():
        die(f"Missing speakers.json in {run_dir}")

    speakers = read_json(speakers_path)
    if args.speaker:
        speakers = [s for s in speakers if s["id"] == args.speaker]
        if not speakers:
            die(f"Speaker not found: {args.speaker}")

    transcripts_dir = run_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    model = build_qwen3_asr_model(args)
    audio_list = [str(Path(info["audio_path"])) for info in speakers]
    results = model.transcribe(audio=audio_list, language="Chinese")

    for info, result in zip(speakers, results):
        speaker_id = info["id"]
        out_path = transcripts_dir / f"{speaker_id}.txt"
        text = result.text if hasattr(result, "text") else str(result)
        out_path.write_text(text.strip(), encoding="utf-8")
        print(f"OK: {speaker_id} -> {out_path}")


def build_voxcpm_model(args: argparse.Namespace):
    from voxcpm import VoxCPM

    model_id = args.clone_model_path or args.clone_model_id
    kwargs = {
        "local_files_only": args.clone_local_files_only,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        model = VoxCPM.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs.pop("local_files_only", None)
        model = VoxCPM.from_pretrained(model_id, **kwargs)
    return model


def synth_cmd(args: argparse.Namespace) -> None:
    run_dir = resolve_run_dir(Path(args.run_dir))
    speakers_path = run_dir / "speakers.json"
    if not speakers_path.exists():
        die(f"Missing speakers.json in {run_dir}")

    if not args.prompt:
        die("--prompt is required.")

    speakers = read_json(speakers_path)
    if args.speaker:
        speakers = [s for s in speakers if s["id"] == args.speaker]
        if not speakers:
            die(f"Speaker not found: {args.speaker}")

    transcripts_dir = run_dir / "transcripts"
    synth_dir = run_dir / "synth"
    synth_dir.mkdir(parents=True, exist_ok=True)

    model = build_voxcpm_model(args)

    import soundfile as sf

    for info in speakers:
        speaker_id = info["id"]
        audio_path = Path(info["audio_path"])
        transcript_path = transcripts_dir / f"{speaker_id}.txt"
        if not transcript_path.exists():
            die(f"Missing transcript for {speaker_id}: {transcript_path}")
        ref_text = transcript_path.read_text(encoding="utf-8").strip()
        if not ref_text:
            die(f"Empty transcript for {speaker_id}")

        out_path = synth_dir / f"{speaker_id}.wav"
        audio = model.generate(
            text=args.prompt,
            prompt_wav_path=str(audio_path),
            prompt_text=ref_text,
            cfg_value=args.clone_cfg_value,
            inference_timesteps=args.clone_inference_timesteps,
            normalize=args.clone_normalize,
            denoise=args.clone_denoise,
            retry_badcase=args.clone_retry_badcase,
        )
        sample_rate = getattr(model.tts_model, "sample_rate", 44100)
        sf.write(out_path, audio, sample_rate)
        print(f"OK: {speaker_id} -> {out_path}")


def preview_cmd(args: argparse.Namespace) -> None:
    ensure_ffplay()
    path: Optional[Path] = None
    if args.audio:
        path = Path(args.audio)
    elif args.run_dir and args.speaker:
        run_dir = resolve_run_dir(Path(args.run_dir))
        speakers_path = run_dir / "speakers.json"
        if not speakers_path.exists():
            die(f"Missing speakers.json in {run_dir}")
        speakers = read_json(speakers_path)
        for info in speakers:
            if info["id"] == args.speaker:
                path = Path(info["audio_path"])
                break
        if path is None:
            die(f"Speaker not found: {args.speaker}")
    else:
        die("Provide --audio or --run-dir + --speaker.")

    if not path.exists():
        die(f"Audio not found: {path}")
    cmd = ["ffplay", "-autoexit", "-nodisp", "-loglevel", "error", str(path)]
    run_cmd(cmd)


def run_cmd_all(args: argparse.Namespace) -> None:
    diarize_args = argparse.Namespace(
        input=args.input,
        out_dir=args.out_dir,
        pipeline=args.pipeline,
        hf_token=args.hf_token,
        device=args.device,
    )
    diarize_cmd(diarize_args)

    asr_args = argparse.Namespace(
        run_dir=args.out_dir,
        speaker=args.speaker,
        asr_model_id=args.asr_model_id,
        asr_model_path=args.asr_model_path,
        asr_device=args.asr_device,
        asr_dtype=args.asr_dtype,
        asr_attn=args.asr_attn,
        asr_batch_size=args.asr_batch_size,
        asr_max_new_tokens=args.asr_max_new_tokens,
        asr_language=args.asr_language,
        asr_local_files_only=args.asr_local_files_only,
    )
    asr_cmd(asr_args)

    synth_args = argparse.Namespace(
        run_dir=args.out_dir,
        speaker=args.speaker,
        prompt=args.prompt,
        clone_model_id=args.clone_model_id,
        clone_model_path=args.clone_model_path,
        clone_cfg_value=args.clone_cfg_value,
        clone_inference_timesteps=args.clone_inference_timesteps,
        clone_normalize=args.clone_normalize,
        clone_denoise=args.clone_denoise,
        clone_retry_badcase=args.clone_retry_badcase,
        clone_local_files_only=args.clone_local_files_only,
    )
    synth_cmd(synth_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diarize -> concat -> ASR -> voice clone pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    diarize = subparsers.add_parser("diarize", help="run diarization and concat per speaker")
    diarize.add_argument("--input", required=True, help="input wav/mp3 file")
    diarize.add_argument("--out-dir", required=True, help="output directory for this run")
    diarize.add_argument(
        "--pipeline",
        default="pyannote/speaker-diarization-community-1",
        help="pyannote pipeline id",
    )
    diarize.add_argument("--hf-token", help="HuggingFace token (or HUGGINGFACE_TOKEN env)")
    diarize.add_argument("--device", help="torch device, e.g. cuda or cpu")
    diarize.set_defaults(func=diarize_cmd)

    list_cmd_parser = subparsers.add_parser("list", help="list speakers and durations")
    list_cmd_parser.add_argument("--run-dir", required=True)
    list_cmd_parser.set_defaults(func=list_cmd)

    asr = subparsers.add_parser("asr", help="run Qwen3-ASR per speaker")
    asr.add_argument("--run-dir", required=True)
    asr.add_argument("--speaker", help="speaker id like speaker_00")
    asr.add_argument("--asr-model-id", default="Qwen/Qwen3-ASR-0.6B")
    asr.add_argument("--asr-model-path", help="local path to model")
    asr.add_argument("--asr-device", help="torch device, e.g. cuda:0 or cpu")
    asr.add_argument("--asr-dtype", default="bfloat16")
    asr.add_argument("--asr-attn", default="flash_attention_2")
    asr.add_argument("--asr-batch-size", type=int, default=1)
    asr.add_argument("--asr-max-new-tokens", type=int, default=1024)
    asr.add_argument("--asr-language", default="auto")
    asr.add_argument("--asr-local-files-only", action="store_true")
    asr.set_defaults(func=asr_cmd)

    synth = subparsers.add_parser("synth", help="clone voice and synthesize prompt")
    synth.add_argument("--run-dir", required=True)
    synth.add_argument("--speaker", help="speaker id like speaker_00")
    synth.add_argument("--prompt", required=True, help="text to synthesize")
    synth.add_argument("--clone-model-id", default="openbmb/VoxCPM1.5")
    synth.add_argument("--clone-model-path", help="local path to model")
    synth.add_argument("--clone-cfg-value", type=float, default=1.4)
    synth.add_argument("--clone-inference-timesteps", type=int, default=10)
    synth.add_argument("--clone-normalize", action=argparse.BooleanOptionalAction, default=True)
    synth.add_argument("--clone-denoise", action=argparse.BooleanOptionalAction, default=True)
    synth.add_argument("--clone-retry-badcase", action=argparse.BooleanOptionalAction, default=True)
    synth.add_argument("--clone-local-files-only", action="store_true")
    synth.set_defaults(func=synth_cmd)

    preview = subparsers.add_parser("preview", help="play speaker audio or a file")
    preview.add_argument("--run-dir", help="run directory containing speakers.json")
    preview.add_argument("--speaker", help="speaker id like speaker_00")
    preview.add_argument("--audio", help="path to audio file")
    preview.set_defaults(func=preview_cmd)

    run_all = subparsers.add_parser("run", help="diarize + asr + synth")
    run_all.add_argument("--input", required=True)
    run_all.add_argument("--out-dir", required=True)
    run_all.add_argument(
        "--pipeline",
        default="pyannote/speaker-diarization-community-1",
        help="pyannote pipeline id",
    )
    run_all.add_argument("--hf-token", help="HuggingFace token (or HUGGINGFACE_TOKEN env)")
    run_all.add_argument("--device", help="torch device, e.g. cuda or cpu")
    run_all.add_argument("--speaker", help="speaker id like speaker_00")
    run_all.add_argument("--prompt", required=True, help="text to synthesize")
    run_all.add_argument("--asr-model-id", default="Qwen/Qwen3-ASR-0.6B")
    run_all.add_argument("--asr-model-path", help="local path to model")
    run_all.add_argument("--asr-device", help="torch device, e.g. cuda:0 or cpu")
    run_all.add_argument("--asr-dtype", default="bfloat16")
    run_all.add_argument("--asr-attn", default="flash_attention_2")
    run_all.add_argument("--asr-batch-size", type=int, default=1)
    run_all.add_argument("--asr-max-new-tokens", type=int, default=1024)
    run_all.add_argument("--asr-language", default="auto")
    run_all.add_argument("--asr-local-files-only", action="store_true")
    run_all.add_argument("--clone-model-id", default="openbmb/VoxCPM1.5")
    run_all.add_argument("--clone-model-path", help="local path to model")
    run_all.add_argument("--clone-cfg-value", type=float, default=1.4)
    run_all.add_argument("--clone-inference-timesteps", type=int, default=10)
    run_all.add_argument("--clone-normalize", action=argparse.BooleanOptionalAction, default=True)
    run_all.add_argument("--clone-denoise", action=argparse.BooleanOptionalAction, default=True)
    run_all.add_argument("--clone-retry-badcase", action=argparse.BooleanOptionalAction, default=True)
    run_all.add_argument("--clone-local-files-only", action="store_true")
    run_all.set_defaults(func=run_cmd_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
