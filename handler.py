"""RunPod Serverless worker for LTX-2.3.
# cache-bust: force COPY re-run

Two modes:
  - "a2v":   talking-head avatar from audio + image + prompt (A2VidPipelineTwoStage).
  - "video": text-/image-to-video with optional IC-LoRAs (DistilledPipeline).

Pipeline cached in-process; rebuilt only when mode or LoRA set changes.
Weights live on the attached RunPod Network Volume at /runpod-volume.
"""
from __future__ import annotations

import base64
import gc
import json
import logging
import os
import struct
import tempfile
import threading
from pathlib import Path
from typing import Any

import requests
import runpod
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
MODELS_ROOT = Path(os.environ.get("LTX_MODELS_ROOT", "/runpod-volume/models"))
LORAS_DIR = Path(os.environ.get("LTX_LORAS_DIR", "/runpod-volume/loras"))

DISTILLED_CHECKPOINT = MODELS_ROOT / "ltx-2.3-22b-distilled-1.1.safetensors"
SPATIAL_UPSAMPLER = MODELS_ROOT / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT = MODELS_ROOT / "gemma-3-12b-it-qat-q4_0-unquantized"
# Stage-2 LoRA used by A2VidPipelineTwoStage (per Lightricks blog).
STAGE2_DISTILLED_LORA = MODELS_ROOT / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"

DEFAULT_FRAME_RATE = 24.0
MAX_SEED = (1 << 31) - 1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("ltx-a2v")


# ─────────────────────────────────────────────────────────────────────────────
# Runtime patches (mirror the Lightricks HF Space)
# ─────────────────────────────────────────────────────────────────────────────
def _apply_runtime_patches() -> None:
    # xformers attention into ltx_core
    try:
        from ltx_core.model.transformer import attention as _attn_mod
        from xformers.ops import memory_efficient_attention as _mea
        _attn_mod.memory_efficient_attention = _mea
        log.info("xformers attention patched")
    except Exception as e:  # noqa: BLE001
        log.warning("xformers patch skipped: %s", e)

    try:
        from xformers.ops.fmha import _set_use_fa3
        _set_use_fa3(False)
        log.info("FA3 dispatch disabled")
    except Exception as e:  # noqa: BLE001
        log.warning("FA3 disable skipped: %s", e)

    try:
        from ltx_core.loader.primitives import StateDict
        from ltx_core.loader.sft_loader import SafetensorsStateDictLoader

        dtype_map = {
            "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
            "BF16": torch.bfloat16,
            "F8_E5M2": torch.float8_e5m2, "F8_E4M3": torch.float8_e4m3fn,
            "I64": torch.int64, "I32": torch.int32, "I16": torch.int16,
            "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
        }

        def _patched_load(self, path, sd_ops, device=None):
            sd, size = {}, 0
            dtype = set()
            device = device or torch.device("cpu")
            for shard_path in (path if isinstance(path, list) else [path]):
                with open(shard_path, "rb") as f:
                    header_len = struct.unpack("<Q", f.read(8))[0]
                    header = json.loads(f.read(header_len).decode("utf-8"))
                    base_off = 8 + header_len
                    for name, meta in header.items():
                        if name == "__metadata__":
                            continue
                        ek = name if sd_ops is None else sd_ops.apply_to_key(name)
                        if ek is None:
                            continue
                        a, b = meta["data_offsets"]
                        f.seek(base_off + a)
                        buf = f.read(b - a)
                        t = torch.frombuffer(bytearray(buf), dtype=dtype_map[meta["dtype"]]).reshape(meta["shape"])
                        t = t.to(device=device, non_blocking=True, copy=False)
                        kvs = (((ek, t),) if sd_ops is None else sd_ops.apply_to_key_value(ek, t))
                        for k, v in kvs:
                            size += v.nbytes
                            dtype.add(v.dtype)
                            sd[k] = v
            return StateDict(sd=sd, device=device, size=size, dtype=dtype)

        SafetensorsStateDictLoader.load = _patched_load
        log.info("safetensors chunked-read loader installed")
    except Exception as e:  # noqa: BLE001
        log.warning("safetensors patch skipped: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline cache
# ─────────────────────────────────────────────────────────────────────────────
_active: dict | None = None  # {"key": str, "pipeline": ...}
_lock = threading.Lock()
_patched = False


def _ensure_patched() -> None:
    global _patched
    if not _patched:
        _apply_runtime_patches()
        _patched = True


def _evict() -> None:
    global _active
    if _active is None:
        return
    log.info("evicting pipeline key=%s", _active["key"])
    _active = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _lora_specs(loras: list[dict]):
    from ltx_core.loader import LoraPathStrengthAndSDOps
    from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
    out = []
    for entry in loras or []:
        name = entry["name"]
        path = LORAS_DIR / f"{name}.safetensors"
        if not path.exists():
            raise FileNotFoundError(f"LoRA not found: {name} ({path})")
        strength = float(entry.get("strength", 1.0))
        out.append(LoraPathStrengthAndSDOps(str(path), strength, LTXV_LORA_COMFY_RENAMING_MAP))
    return out


def _key_for(mode: str, loras: list[dict]) -> str:
    if not loras:
        return f"{mode}:none"
    parts = sorted(f"{e['name']}@{float(e.get('strength', 1.0))}" for e in loras)
    return f"{mode}:" + "|".join(parts)


def get_pipeline(mode: str, loras: list[dict]):
    """mode = 'a2v' | 'video'. Rebuilds only on mode/LoRA change."""
    global _active
    key = _key_for(mode, loras)
    with _lock:
        if _active is not None and _active["key"] == key:
            return _active["pipeline"]

        _ensure_patched()
        _evict()

        from ltx_core.quantization import QuantizationPolicy

        if mode == "a2v":
            from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage
            from ltx_core.loader import LoraPathStrengthAndSDOps
            from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP

            if not STAGE2_DISTILLED_LORA.exists():
                raise FileNotFoundError(f"missing distilled-384 LoRA: {STAGE2_DISTILLED_LORA}")

            distilled_lora = [LoraPathStrengthAndSDOps(
                str(STAGE2_DISTILLED_LORA), 0.8, LTXV_LORA_COMFY_RENAMING_MAP,
            )]
            log.info("loading A2VidPipelineTwoStage")
            pipe = A2VidPipelineTwoStage(
                checkpoint_path=str(DISTILLED_CHECKPOINT),
                distilled_lora=distilled_lora,
                spatial_upsampler_path=str(SPATIAL_UPSAMPLER),
                gemma_root=str(GEMMA_ROOT),
                loras=[],
                quantization=QuantizationPolicy.fp8_cast(),
            )
        elif mode == "video":
            from ltx_pipelines.distilled import DistilledPipeline
            log.info("loading DistilledPipeline (loras=%d)", len(loras))
            pipe = DistilledPipeline(
                distilled_checkpoint_path=str(DISTILLED_CHECKPOINT),
                spatial_upsampler_path=str(SPATIAL_UPSAMPLER),
                gemma_root=str(GEMMA_ROOT),
                loras=_lora_specs(loras),
                quantization=QuantizationPolicy.fp8_cast(),
            )
        else:
            raise ValueError(f"unknown mode: {mode}")

        # Preload components so first request doesn't pay it.
        # Only DistilledPipeline exposes `model_ledger`; A2VidPipelineTwoStage
        # structures its components as direct attributes (stage_1, stage_2, etc.)
        # and lazy-loads them on first call, so we skip preload there.
        ledger = getattr(pipe, "model_ledger", None)
        if ledger is not None:
            for fn in ("transformer", "video_encoder", "video_decoder", "audio_decoder",
                       "vocoder", "spatial_upsampler", "text_encoder",
                       "gemma_embeddings_processor"):
                try:
                    getattr(ledger, fn)()
                except AttributeError:
                    pass

        _active = {"key": key, "pipeline": pipe}
        log.info("pipeline ready (key=%s)", key)
        return pipe


# ─────────────────────────────────────────────────────────────────────────────
# Asset fetchers
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_url(url: str, dest: Path) -> Path:
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    with dest.open("wb") as f:
        for chunk in r.iter_content(64 * 1024):
            f.write(chunk)
    return dest


def _write_b64(b64: str, dest: Path) -> Path:
    dest.write_bytes(base64.b64decode(b64))
    return dest


def _resolve_asset(d: dict, key_url: str, key_b64: str, dest: Path) -> Path | None:
    if d.get(key_url):
        return _fetch_url(d[key_url], dest)
    if d.get(key_b64):
        return _write_b64(d[key_b64], dest)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Runners
# ─────────────────────────────────────────────────────────────────────────────
def _encode_video_and_audio(video, audio, output_path: str, fps: float, num_frames: int) -> None:
    from ltx_core.model.video_vae import get_video_chunks_number, TilingConfig
    from ltx_pipelines.utils.media_io import encode_video
    chunks = get_video_chunks_number(num_frames, TilingConfig.default())
    encode_video(
        video=video, fps=int(round(fps)), audio=audio,
        output_path=output_path, video_chunks_number=chunks,
    )


def _run_a2v(inp: dict) -> dict:
    """A2V: audio_url + image_url + prompt → talking avatar."""
    from ltx_core.components.guiders import MultiModalGuiderParams

    work = Path(tempfile.mkdtemp(prefix="a2v_"))
    audio_path = _resolve_asset(inp, "audio_url", "audio_b64", work / "audio.wav")
    image_path = _resolve_asset(inp, "image_url", "image_b64", work / "image.jpg")
    if not audio_path:
        raise ValueError("audio_url or audio_b64 required")
    if not image_path:
        raise ValueError("image_url or image_b64 required")

    prompt = inp.get("prompt", "")
    seed = int(inp.get("seed", 42))
    height = int(inp.get("height", 1024))
    width = int(inp.get("width", 768))
    enhance_prompt = bool(inp.get("enhance_prompt", False))
    fps = float(inp.get("fps", DEFAULT_FRAME_RATE))

    # Frame count derived from audio length if not provided.
    if inp.get("num_frames"):
        num_frames = int(inp["num_frames"])
    else:
        import av  # already pulled by ltx-pipelines deps
        with av.open(str(audio_path)) as container:
            stream = container.streams.audio[0]
            dur = float(stream.duration * stream.time_base) if stream.duration else 5.0
        num_frames = int(dur * fps) + 1
    num_frames = ((num_frames - 1 + 7) // 8) * 8 + 1  # snap to 8k+1

    pipe = get_pipeline("a2v", [])
    # A2V's audio VAE expects stereo (2-channel) mel-spec input. Force-convert
    # via ffmpeg in case the input is mono (InfiniteTalk previews often are).
    stereo_audio = work / "audio_stereo.wav"
    import subprocess as _sp
    _sp.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(audio_path),
         "-ac", "2", "-ar", "44100", str(stereo_audio)],
        check=True,
    )
    audio_path = stereo_audio
    # Signature type hint says tuple but internal code uses img.path — use ImageConditioningInput.
    from ltx_pipelines.utils.args import ImageConditioningInput
    images = [ImageConditioningInput(path=str(image_path), frame_idx=0, strength=1.0)]

    with torch.inference_mode():
        video, audio = pipe(
            prompt=prompt,
            negative_prompt=inp.get("negative_prompt", ""),
            seed=seed,
            height=height, width=width,
            num_frames=num_frames, frame_rate=fps,
            num_inference_steps=int(inp.get("num_inference_steps", 8)),
            video_guider_params=MultiModalGuiderParams(),
            images=images,
            audio_path=str(audio_path),
            enhance_prompt=enhance_prompt,
        )

    out = work / "out.mp4"
    _encode_video_and_audio(video, audio, str(out), fps, num_frames)
    return {"video_b64": base64.b64encode(out.read_bytes()).decode("ascii"),
            "num_frames": num_frames, "fps": fps, "seed": seed}


def _run_video(inp: dict) -> dict:
    """DistilledPipeline: t2v/i2v with optional IC-LoRAs."""
    from ltx_pipelines.utils.args import ImageConditioningInput

    work = Path(tempfile.mkdtemp(prefix="vid_"))
    prompt = inp["prompt"]
    duration = float(inp.get("duration", 3.0))
    seed = int(inp.get("seed", 42))
    height = int(inp.get("height", 1024))
    width = int(inp.get("width", 1536))
    enhance_prompt = bool(inp.get("enhance_prompt", False))
    loras = inp.get("loras", []) or []
    fps = DEFAULT_FRAME_RATE

    num_frames = int(duration * fps) + 1
    num_frames = ((num_frames - 1 + 7) // 8) * 8 + 1

    image_paths: list[str] = []
    if inp.get("image_url") or inp.get("image_b64"):
        ip = _resolve_asset(inp, "image_url", "image_b64", work / "img_0.jpg")
        if ip:
            image_paths.append(str(ip))
    for i, u in enumerate(inp.get("image_urls", []) or []):
        image_paths.append(str(_fetch_url(u, work / f"img_{i}.jpg")))

    last_image_path = None
    if inp.get("last_image_url") or inp.get("last_image_b64"):
        lp = _resolve_asset(inp, "last_image_url", "last_image_b64", work / "last.jpg")
        if lp:
            last_image_path = str(lp)

    images = [ImageConditioningInput(path=p, frame_idx=0, strength=1.0) for p in image_paths]
    if last_image_path:
        images.append(ImageConditioningInput(path=last_image_path, frame_idx=num_frames - 1, strength=1.0))

    pipe = get_pipeline("video", loras)
    kw = dict(
        prompt=prompt, seed=seed, height=height, width=width,
        num_frames=num_frames, frame_rate=fps, images=images,
        enhance_prompt=enhance_prompt,
    )
    if inp.get("reference_video_url"):
        ref = _fetch_url(inp["reference_video_url"], work / "ref.mp4")
        kw["video_conditioning"] = [(str(ref), float(inp.get("reference_strength", 1.0)))]

    from ltx_core.model.video_vae import TilingConfig
    kw["tiling_config"] = TilingConfig.default()

    with torch.inference_mode():
        video, audio = pipe(**kw)

    out = work / "out.mp4"
    _encode_video_and_audio(video, audio, str(out), fps, num_frames)
    return {"video_b64": base64.b64encode(out.read_bytes()).decode("ascii"),
            "num_frames": num_frames, "fps": fps, "seed": seed}


def _list_loras() -> list[str]:
    if not LORAS_DIR.exists():
        return []
    return sorted(p.stem for p in LORAS_DIR.glob("*.safetensors"))


# ─────────────────────────────────────────────────────────────────────────────
# RunPod handler
# ─────────────────────────────────────────────────────────────────────────────
def handler(event: dict) -> dict:
    inp = event.get("input", {})
    mode = inp.get("mode", "a2v").lower()

    if mode == "list-loras":
        return {"loras": _list_loras()}
    if mode == "a2v":
        return _run_a2v(inp)
    if mode == "video":
        return _run_video(inp)
    return {"error": f"unknown mode: {mode}", "valid": ["a2v", "video", "list-loras"]}


runpod.serverless.start({"handler": handler})
