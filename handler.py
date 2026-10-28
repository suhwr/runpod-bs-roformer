import base64
import os
import pathlib
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import runpod
import torch
from audio_separator.separator import Separator

# ============================================================================ #
# Configuration & Hardware Diagnostics                                         #
# ============================================================================ #

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/output")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
DEFAULT_CATBOX_USER_HASH = os.environ.get("CATBOX_KEY", "")

pathlib.Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("🚀 Initializing BS-RoFormer Serverless Worker")
print(f"📦 PyTorch Version : {torch.__version__}")
print(f"⚡ CUDA Available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU Device      : {torch.cuda.get_device_name(0)}")
    print(f"🧠 Total VRAM      : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
print(f"📂 Model Directory : {MODEL_DIR}")
print(f"🎯 Default Model   : {DEFAULT_MODEL}")
print("=" * 60)

# ============================================================================ #
# Separator Engine Warm Pre-loading                                            #
# ============================================================================ #

separator = Separator(
    model_file_dir=MODEL_DIR,
    output_dir=OUTPUT_DIR,
    output_format="MP3",
    output_bitrate="320k",
    use_autocast=True,  # FP16 (Half-Precision) optimization for maximum speed & lower VRAM
)

CURRENT_LOADED_MODEL: Optional[str] = None

try:
    print(f"⏳ Pre-loading default model: {DEFAULT_MODEL}...")
    separator.load_model(model_filename=DEFAULT_MODEL)
    CURRENT_LOADED_MODEL = DEFAULT_MODEL
    print(f"✅ Pre-loaded {DEFAULT_MODEL} successfully into VRAM!")
except Exception as exc:
    print(f"⚠️ Warning during model pre-load: {exc}")


# ============================================================================ #
# Storage Uploaders with Automatic Fallback Chain                              #
# uguu.se -> catbox.moe -> ucarecdn.com -> base64 (fallback)                  #
# ============================================================================ #

def _upload_uguu(file_path: pathlib.Path) -> str:
    """Upload to Uguu.se (up to 100MB, high speed)."""
    with open(file_path, "rb") as f:
        resp = requests.post(
            "https://uguu.se/upload",
            params={"output": "json"},
            files={"files[]": (file_path.name, f)},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        files = data.get("files")
        if files and len(files) > 0 and files[0].get("url"):
            return files[0]["url"]
        raise ValueError(f"Uguu response invalid: {data}")


def _upload_catbox(file_path: pathlib.Path, user_hash: Optional[str] = None) -> str:
    """Upload to Catbox.moe (up to 200MB)."""
    data = {"reqtype": "fileupload"}
    hash_to_use = user_hash or DEFAULT_CATBOX_USER_HASH
    if hash_to_use:
        data["userhash"] = hash_to_use

    with open(file_path, "rb") as f:
        resp = requests.post(
            "https://catbox.moe/user/api.php",
            data=data,
            files={"fileToUpload": (file_path.name, f)},
            timeout=90,
        )
        resp.raise_for_status()
        url = resp.text.strip()
        if url.startswith("https://") or url.startswith("http://"):
            return url
        raise ValueError(f"Catbox failed with response: {url}")


def _upload_ucarecdn(file_path: pathlib.Path, pub_key: str = "demopublickey") -> str:
    """Upload to Uploadcare (ucarecdn.com)."""
    with open(file_path, "rb") as f:
        resp = requests.post(
            "https://upload.uploadcare.com/base/",
            data={
                "UPLOADCARE_PUB_KEY": pub_key,
                "UPLOADCARE_STORE": "1",
            },
            files={"file": (file_path.name, f)},
            timeout=60,
        )
        resp.raise_for_status()
        file_id = resp.json().get("file")
        if file_id:
            return f"https://ucarecdn.com/{file_id}/{file_path.name}"
        raise ValueError(f"Uploadcare failed: {resp.text}")


def upload_with_fallback(file_path: pathlib.Path, catbox_hash: Optional[str] = None) -> Tuple[str, str]:
    """Fallback chain: uguu -> catbox -> ucarecdn."""
    errors: List[str] = []

    # 1. Try Uguu
    try:
        url = _upload_uguu(file_path)
        return url, "uguu"
    except Exception as exc:
        errors.append(f"uguu: {exc}")

    # 2. Try Catbox
    try:
        url = _upload_catbox(file_path, user_hash=catbox_hash)
        return url, "catbox"
    except Exception as exc:
        errors.append(f"catbox: {exc}")

    # 3. Try Ucarecdn
    try:
        url = _upload_ucarecdn(file_path)
        return url, "ucarecdn"
    except Exception as exc:
        errors.append(f"ucarecdn: {exc}")

    raise RuntimeError("All uploaders failed: " + " | ".join(errors))


# ============================================================================ #
# Serverless Handler Function                                                  #
# ============================================================================ #

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    global CURRENT_LOADED_MODEL

    start_time = time.time()
    job_input = event.get("input", {})

    audio_url = job_input.get("audio_url")
    if not audio_url:
        return {"status": "FAILED", "error": "audio_url is required"}

    model_name = job_input.get("model_name", DEFAULT_MODEL)
    output_format = job_input.get("output_format", "MP3").upper()
    if output_format not in ["MP3", "WAV", "FLAC"]:
        output_format = "MP3"

    bitrate = str(job_input.get("bitrate", "320k"))
    single_stem = job_input.get("single_stem")  # e.g., "Vocals" or "Instrumental"
    catbox_key = job_input.get("catbox_key", DEFAULT_CATBOX_USER_HASH)

    temp_dir = tempfile.mkdtemp()
    temp_dir_path = pathlib.Path(temp_dir)
    input_path = temp_dir_path / "input_audio.mp3"

    try:
        # 1. Download input audio with stream
        print(f"📥 Downloading audio from: {audio_url}")
        resp = requests.get(audio_url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(input_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print(f"💾 Saved input audio ({input_path.stat().st_size} bytes)")

        # 2. Dynamic model switching if requested model is different
        if model_name != CURRENT_LOADED_MODEL:
            print(f"🔄 Switching model from {CURRENT_LOADED_MODEL} -> {model_name}...")
            separator.load_model(model_filename=model_name)
            CURRENT_LOADED_MODEL = model_name

        # 3. Configure separator options
        separator.output_format = output_format
        separator.output_bitrate = bitrate
        separator.output_single_stem = single_stem if single_stem in ["Vocals", "Instrumental"] else None

        # 4. Run separation
        print(f"⚙️ Running separation using {CURRENT_LOADED_MODEL} (autocast=FP16)...")
        sep_start = time.time()
        output_files = separator.separate(str(input_path))
        sep_elapsed = time.time() - sep_start
        print(f"✅ Separation completed in {sep_elapsed:.2f}s! Stems produced: {output_files}")

        # 5. Process stems & upload via fallback chain
        stems_dict: Dict[str, Dict[str, Any]] = {}
        for file_item in output_files:
            file_path = pathlib.Path(OUTPUT_DIR) / file_item
            if not file_path.exists():
                file_path = pathlib.Path(file_item)

            if not file_path.exists():
                print(f"⚠️ Output file not found: {file_item}")
                continue

            file_lower = file_path.name.lower()
            if "vocals" in file_lower:
                stem_key = "vocals"
            elif "instrumental" in file_lower or "instrument" in file_lower:
                stem_key = "instrumental"
            elif "drums" in file_lower or "drum" in file_lower:
                stem_key = "drums"
            elif "bass" in file_lower:
                stem_key = "bass"
            elif "other" in file_lower:
                stem_key = "other"
            else:
                stem_key = file_path.stem

            file_size = file_path.stat().st_size
            print(f"⬆️ Uploading stem '{stem_key}' ({file_size} bytes)...")

            try:
                url, storage = upload_with_fallback(file_path, catbox_hash=catbox_key)
                stems_dict[stem_key] = {
                    "url": url,
                    "file_name": file_path.name,
                    "size_bytes": file_size,
                    "storage": storage,
                }
                print(f"✨ Stem '{stem_key}' uploaded to {storage}: {url}")
            except Exception as up_err:
                # Fallback to base64 only if file is small enough (<7MB)
                if file_size < 7 * 1024 * 1024:
                    with open(file_path, "rb") as bf:
                        b64_data = base64.b64encode(bf.read()).decode("utf-8")
                    stems_dict[stem_key] = {
                        "base64": b64_data,
                        "file_name": file_path.name,
                        "size_bytes": file_size,
                        "storage": "base64",
                    }
                    print(f"⚠️ Upload failed ({up_err}), fallback to base64 for '{stem_key}'")
                else:
                    stems_dict[stem_key] = {
                        "error": str(up_err),
                        "file_name": file_path.name,
                        "size_bytes": file_size,
                    }

            # Unlink output file locally immediately to keep container disk clean
            try:
                file_path.unlink()
            except Exception:
                pass

        total_elapsed = time.time() - start_time
        return {
            "status": "COMPLETED",
            "model": CURRENT_LOADED_MODEL,
            "execution_time_seconds": round(total_elapsed, 2),
            "separation_time_seconds": round(sep_elapsed, 2),
            "stems": stems_dict,
        }

    except Exception as exc:
        print(f"❌ Error during separation: {exc}")
        return {
            "status": "FAILED",
            "error": f"Separation failed: {str(exc)}",
        }

    finally:
        # Clean up temporary input folder
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir_path, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
