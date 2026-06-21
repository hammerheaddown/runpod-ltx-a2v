# runpod-ltx-a2v

RunPod Serverless worker for LTX-2.3. Two modes:

- **`a2v`** — talking-head avatar from audio + image + prompt (`A2VidPipelineTwoStage`).
- **`video`** — text-/image-to-video with optional IC-LoRAs (`DistilledPipeline`),
  same surface as the Modal app (`/api/generate`).

Pipeline is cached in-process; only rebuilt when mode or LoRA set changes.

## Architecture

```
ViponScraper  →  RunPod Serverless endpoint
                       │
                       ▼
                  handler.py  (one worker = one container = one warm pipeline)
                       │
                       ▼
         /runpod-volume   ←  Network Volume (weights + 12 LoRAs, ~85 GB)
```

## One-time setup

### 1. Create a Network Volume
- RunPod console → Storage → Network Volumes → New.
- 150 GB, region with H100 / RTX PRO 6000 stock.
- Name: `ltx-a2v-vol`.

### 2. Populate it
Spin up a tiny CPU pod (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`,
0.5 vCPU, $0.05-0.20/hr) with `ltx-a2v-vol` mounted at `/workspace`.

In the pod's web terminal:
```bash
cd /workspace
git clone <YOUR-FORK-OR-PASTE> /tmp/prep
export HF_TOKEN=hf_xxx
bash /tmp/prep/scripts/prepare_volume.sh
```

Or paste `scripts/prepare_volume.sh` into the terminal directly. Takes ~20 min,
costs ~$0.10. Skipped (gated) LoRAs are listed at the end with their license URLs.

### 3. Build + push the image
```bash
cd runpod-ltx-a2v
docker build -t <dockerhub-user>/ltx-a2v:latest .
docker push <dockerhub-user>/ltx-a2v:latest
```

### 4. Create the Serverless endpoint
RunPod console → Serverless → New Endpoint:
- **Container image:** `<dockerhub-user>/ltx-a2v:latest`
- **GPU:** RTX PRO 6000 (96 GB) — H100 also fine.
- **Min workers:** 0 (scale-to-zero), **Max workers:** 1-3.
- **Idle timeout:** 60 s.
- **Execution timeout:** 600 s.
- **Network Volume:** attach `ltx-a2v-vol` at `/runpod-volume`.
- **Container disk:** 30 GB.
- **Active workers:** 0 (set to 1 if you want to eliminate cold starts at a flat cost).

Save → endpoint URL appears. Use it as `<RUNPOD_ENDPOINT>` below.

## Calling it

All calls use RunPod's standard sync/async envelope. Auth via `Authorization: Bearer <RUNPOD_API_KEY>`.

### Talking avatar (A2V)
```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "mode": "a2v",
      "audio_url": "https://example.com/voice.wav",
      "image_url": "https://example.com/portrait.jpg",
      "prompt": "person speaking naturally with expressive eye contact",
      "seed": 42,
      "height": 1024,
      "width": 768
    }
  }'
```

Returns `{ "output": { "video_b64": "...", "num_frames": ..., "fps": 24, "seed": ... } }`.
Decode `video_b64` and save as `.mp4`.

### t2v / i2v / LoRA generation
```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "mode": "video",
      "prompt": "cinematic motion, soft lighting",
      "duration": 3,
      "image_url": "https://example.com/first.jpg",
      "last_image_url": "https://example.com/last.jpg",
      "loras": [{"name": "ltx-2.3-22b-ic-lora-colorization-0.9", "strength": 1.0}]
    }
  }'
```

### List available LoRAs
```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"mode": "list-loras"}}'
```

## Cost model

| Item | Rate |
|---|---|
| RTX PRO 6000 Serverless | ~$0.00058/sec executing |
| 6-sec avatar clip @ 30 sec render | ~$0.02 |
| Cold start (after idle) | adds 60-120 sec the first request after idle |
| Network Volume (150 GB) | ~$0.07/GB/mo = ~$10/mo |
| Image storage on Docker Hub | free |

**Compared to Lightricks's `/v2/audio-to-video` API ($0.10/sec audio):**
10 clips/day × 6 sec audio = $6/day on the LTX API vs ~$0.20-0.40/day on RunPod Serverless.
Pays for itself after ~3 days of moderate use.

## Adding more LoRAs

Append the repo ID to `LORA_REPOS` in `scripts/prepare_volume.sh`, rerun on a temp
pod, and they auto-appear in the `list-loras` response. No code change required.

## Known limits

- **Cold start ~90-150 sec** on this image (22B model load is the dominant cost).
  Pay for one Active Worker if latency matters.
- **Base64 video payload** can hit RunPod's 20 MB response limit on clips >~30 sec.
  Swap `video_b64` for an R2/S3 upload + signed URL when you cross that.
- **`a2v` audio currently expects WAV.** MP3/M4A/OGG should work via `av` decoding;
  test before relying on them.
 
