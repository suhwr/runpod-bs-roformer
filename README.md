# Ultra-Optimized BS-RoFormer Serverless Worker

High-performance, production-ready serverless worker for **BS-RoFormer** (Band-Split RoPE Transformer) and **Mel-Band RoFormer** music source separation, powered by `python-audio-separator` and PyTorch CUDA.

Achieves world-leading Signal-to-Distortion Ratio (**SDR 12.97 dB**) for pure, bleed-free vocal and instrumental isolation.

---

## ⚡ Key Optimizations

1. **Zero Cold-Start Model Baking:**
   The highest-scoring SOTA weights (`model_bs_roformer_ep_317_sdr_12.9755.ckpt` ViperX 12.97 and `mel_band_roformer_kim_ft_unfrozen.ckpt`) are pre-downloaded and baked directly into the Docker image layer. When a new container spins up, inference starts immediately without waiting to download weights.

2. **Native FP16 Autocast Acceleration:**
   Executes model inference in half-precision (`use_autocast=True`) on modern NVIDIA Tensor Cores (RTX 3090, 4090, A10G, L4, etc.), cutting VRAM consumption by 50% and doubling throughput.

3. **Global Warm Loading:**
   Default model weights are loaded into GPU VRAM once during container initialization, ensuring warm invocations execute in seconds.

4. **Automatic Storage Upload Fallback Chain:**
   Avoids payload drop limits (< 10 MB) by converting separated stems to high-quality MP3 (320 kbps) and uploading them to a multi-tier fallback chain:
   $$\text{Uguu} \longrightarrow \text{Catbox} \longrightarrow \text{Uploadcare (Ucarecdn)}$$
   The endpoint returns a tiny JSON response (< 1 KB) with public download URLs.

---

## 📥 API Input Specification

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `audio_url` | `string` | *(Required)* | Direct download URL to the source audio file (MP3, WAV, FLAC, M4A, etc.) |
| `model_name` | `string` | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | Model checkpoint to use (`model_bs_roformer_ep_317_sdr_12.9755.ckpt` or `mel_band_roformer_kim_ft_unfrozen.ckpt`) |
| `output_format`| `string` | `"MP3"` | Audio format for separated stems (`"MP3"`, `"WAV"`, `"FLAC"`) |
| `bitrate` | `string` | `"320k"` | Bitrate for MP3 output (e.g. `"320k"`, `"192k"`) |
| `single_stem` | `string` | `null` | Optional stem filter: `"Vocals"` or `"Instrumental"`. Omit to export both stems |
| `catbox_key` | `string` | `null` | Optional Catbox userhash for persistent file management |

### Example Request:

```json
{
  "input": {
    "audio_url": "https://uguu.se/sample_audio.mp3",
    "model_name": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "output_format": "MP3",
    "bitrate": "320k"
  }
}
```

---

## 📤 API Output Specification

```json
{
  "status": "COMPLETED",
  "model": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
  "execution_time_seconds": 14.85,
  "separation_time_seconds": 12.10,
  "stems": {
    "vocals": {
      "url": "https://h.uguu.se/vocal_stem.mp3",
      "file_name": "input_audio_(Vocals)_model_bs_roformer_ep_317_sdr_12.9755.mp3",
      "size_bytes": 4512340,
      "storage": "uguu"
    },
    "instrumental": {
      "url": "https://h.uguu.se/inst_stem.mp3",
      "file_name": "input_audio_(Instrumental)_model_bs_roformer_ep_317_sdr_12.9755.mp3",
      "size_bytes": 5612340,
      "storage": "uguu"
    }
  }
}
```

---

## 🛠️ Environment Variables

- `FOUR_STEM_ENGINE`: Multi-stem separation engine: `"roformer"` (default) or `"demucs"`.
- `FOUR_STEM_ROFORMER_MODEL`: SOTA RoFormer model for multi-stem separation (defaults to `BS-Roformer-SW.ckpt`).
- `FOUR_STEM_DEMUCS_MODEL`: Demucs model for 4-stem separation (defaults to `htdemucs_ft.yaml`).
- `DEFAULT_MODEL`: Default 2-stem model checkpoint filename (defaults to `model_bs_roformer_ep_317_sdr_12.9755.ckpt`).
- `MAX_CONCURRENCY`: Worker job queue concurrency (defaults to `1` for sequential, crash-free execution).
- `CATBOX_KEY`: Default Catbox userhash (optional).
- `MODEL_DIR`: Directory where pre-baked weights reside (defaults to `/app/models`).
