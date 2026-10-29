import base64
import hashlib
import mimetypes
import os
import pathlib
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests
import runpod
import torch
from audio_separator.separator import Separator

# ============================================================================ #
# Configuration & Hardware Diagnostics                                         #
# ============================================================================ #

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
DEFAULT_CATBOX_USER_HASH = os.environ.get("CATBOX_KEY", "")
MAX_CONCURRENCY = max(1, int(os.environ.get("MAX_CONCURRENCY", "1")))

# 4-Stem / Multi-Stem Engine Configuration
# FOUR_STEM_ENGINE: "roformer" (default) or "demucs"
raw_engine_env = os.environ.get("FOUR_STEM_ENGINE", os.environ.get("FOUR_STEM_MODEL", "roformer")).strip()
if raw_engine_env.lower() in ["roformer", "demucs"]:
    FOUR_STEM_ENGINE = raw_engine_env.lower()
elif "demucs" in raw_engine_env.lower():
    FOUR_STEM_ENGINE = "demucs"
elif "roformer" in raw_engine_env.lower():
    FOUR_STEM_ENGINE = "roformer"
else:
    FOUR_STEM_ENGINE = "roformer"

FOUR_STEM_ROFORMER_MODEL = os.environ.get("FOUR_STEM_ROFORMER_MODEL", "BS-Roformer-SW.ckpt")
FOUR_STEM_DEMUCS_MODEL = os.environ.get("FOUR_STEM_DEMUCS_MODEL", "htdemucs_ft.yaml")

pathlib.Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

print("=" * 60)
print(f"🚀 Initializing Serverless Audio Separator Worker (Max Concurrency: {MAX_CONCURRENCY})")
print(f"📦 PyTorch Version   : {torch.__version__}")
print(f"⚡ CUDA Available    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU Device        : {torch.cuda.get_device_name(0)}")
    print(f"🧠 Total VRAM        : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
print(f"📂 Model Directory   : {MODEL_DIR}")
print(f"🎯 Default 2-Stem    : {DEFAULT_MODEL}")
print(f"🎛️ 4-Stem Engine     : {FOUR_STEM_ENGINE} (roformer: {FOUR_STEM_ROFORMER_MODEL}, demucs: {FOUR_STEM_DEMUCS_MODEL})")
print("=" * 60)

# ============================================================================ #
# Separator Engine Warm Pre-loading                                            #
# ============================================================================ #

separator = Separator(
    model_file_dir=MODEL_DIR,
    output_dir="/tmp",
    output_format="MP3",
    output_bitrate="320k",
    use_autocast=True,  # FP16 (Half-Precision) optimization
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
# Backblaze B2 (Primary) -> Uguu -> Catbox -> Uploadcare -> Base64             #
# ============================================================================ #

def _upload_b2(file_path: pathlib.Path, folder: str = "tmp/audio") -> str:
    """
    Upload audio stems to Backblaze B2 (boto3 S3 or native REST API).
    Target prefix 'tmp/' allows automatic Lifecycle cleanup rules.
    """
    key_id = os.environ.get("B2_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    app_key = os.environ.get("B2_APPLICATION_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    bucket = os.environ.get("B2_BUCKET_NAME")
    endpoint = os.environ.get("B2_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL")

    if not (key_id and app_key and bucket):
        raise ValueError("Backblaze B2 credentials (B2_KEY_ID, B2_APPLICATION_KEY, B2_BUCKET_NAME) not set")

    clean_folder = folder.strip("/")
    target_key = f"{clean_folder}/{int(time.time())}_{uuid.uuid4().hex[:8]}_{file_path.name}"
    content_type, _ = mimetypes.guess_type(str(file_path))
    if not content_type:
        content_type = "audio/mpeg"

    # Attempt 1: boto3 S3 API if endpoint is set
    if endpoint:
        try:
            import boto3
            from botocore.config import Config

            s3 = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=key_id,
                aws_secret_access_key=app_key,
                config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
            )
            with open(file_path, "rb") as f:
                s3.upload_fileobj(f, bucket, target_key, ExtraArgs={"ContentType": content_type})

            public_prefix = os.environ.get("B2_PUBLIC_URL_PREFIX", "").rstrip("/")
            if public_prefix:
                return f"{public_prefix}/{target_key}"

            try:
                return s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": target_key}, ExpiresIn=86400)
            except Exception:
                parsed = urllib.parse.urlsplit(endpoint)
                return f"{parsed.scheme}://{bucket}.{parsed.netloc}/{target_key}"
        except ImportError:
            pass

    # Attempt 2: B2 Native REST API
    auth_resp = requests.get(
        "https://api.backblazeb2.com/b2api/v3/b2_authorize_account",
        auth=(key_id, app_key),
        timeout=30,
    )
    auth_resp.raise_for_status()
    auth_data = auth_resp.json()
    api_url = auth_data["apiUrl"]
    auth_token = auth_data["authorizationToken"]
    download_url = auth_data["downloadUrl"]

    bucket_id = os.environ.get("B2_BUCKET_ID") or auth_data.get("allowed", {}).get("bucketId")
    if not bucket_id:
        lb_resp = requests.post(
            f"{api_url}/b2api/v3/b2_list_buckets",
            headers={"Authorization": auth_token},
            json={"accountId": auth_data["accountId"], "bucketName": bucket},
            timeout=30,
        )
        lb_resp.raise_for_status()
        buckets = lb_resp.json().get("buckets", [])
        if buckets:
            bucket_id = buckets[0]["bucketId"]
        else:
            raise ValueError(f"B2 bucket '{bucket}' not found")

    up_resp = requests.post(
        f"{api_url}/b2api/v3/b2_get_upload_url",
        headers={"Authorization": auth_token},
        json={"bucketId": bucket_id},
        timeout=30,
    )
    up_resp.raise_for_status()
    up_data = up_resp.json()
    upload_url = up_data["uploadUrl"]
    upload_token = up_data["authorizationToken"]

    sha1 = hashlib.sha1()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha1.update(chunk)
    digest = sha1.hexdigest()

    file_size = file_path.stat().st_size
    with open(file_path, "rb") as f:
        res = requests.post(
            upload_url,
            headers={
                "Authorization": upload_token,
                "X-Bz-File-Name": urllib.parse.quote(target_key),
                "Content-Type": content_type,
                "Content-Length": str(file_size),
                "X-Bz-Content-Sha1": digest,
            },
            data=f,
            timeout=180,
        )
        res.raise_for_status()

    public_prefix = os.environ.get("B2_PUBLIC_URL_PREFIX", "").rstrip("/")
    if public_prefix:
        return f"{public_prefix}/{target_key}"

    return f"{download_url}/file/{bucket}/{target_key}"


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
    """Fallback chain: Backblaze B2 (if configured) -> uguu -> catbox -> ucarecdn."""
    errors: List[str] = []

    # 1. Backblaze B2 (tmp/ auto-expire lifecycle)
    if os.environ.get("B2_APPLICATION_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"):
        try:
            url = _upload_b2(file_path, folder="tmp/audio")
            return url, "backblaze_b2"
        except Exception as exc:
            print(f"⚠️ Backblaze B2 upload failed ({exc}), falling back to other uploaders...")
            errors.append(f"b2: {exc}")

    # 2. Try Uguu
    try:
        url = _upload_uguu(file_path)
        return url, "uguu"
    except Exception as exc:
        errors.append(f"uguu: {exc}")

    # 3. Try Catbox
    try:
        url = _upload_catbox(file_path, user_hash=catbox_hash)
        return url, "catbox"
    except Exception as exc:
        errors.append(f"catbox: {exc}")

    # 4. Try Ucarecdn
    try:
        url = _upload_ucarecdn(file_path)
        return url, "ucarecdn"
    except Exception as exc:
        errors.append(f"ucarecdn: {exc}")

    raise RuntimeError("All uploaders failed: " + " | ".join(errors))

    raise RuntimeError("All uploaders failed: " + " | ".join(errors))


def _download_file(url: str, dest_path: pathlib.Path) -> None:
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


# ============================================================================ #
# Serverless Handler Function (Sequential Execution, Concurrency=1)            #
# ============================================================================ #

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    global CURRENT_LOADED_MODEL

    start_time = time.time()
    job_id = event.get("id", f"job_{int(start_time * 1000)}")
    job_input = event.get("input", {})

    audio_url = job_input.get("audio_url")
    if not audio_url:
        return {"status": "FAILED", "error": "audio_url is required"}

    raw_model_name = str(job_input.get("model_name", "")).strip()
    is_4stem = bool(job_input.get("is_4stem", False))
    force_model = bool(job_input.get("force_model", False))

    # Allow runtime engine override per job, otherwise use FOUR_STEM_ENGINE env
    job_engine = str(job_input.get("four_stem_engine", "")).strip().lower()
    active_engine = job_engine if job_engine in ["roformer", "demucs"] else FOUR_STEM_ENGINE

    # Check if 4-stem / multi-stem separation is requested
    is_multi_request = (
        is_4stem or
        raw_model_name.lower() in [
            "4stems", "4stem", "4", "all", "multi", "multistem",
            "htdemucs_ft.yaml", "htdemucs.yaml", "bs-roformer-sw.ckpt"
        ]
    )

    if is_multi_request:
        if force_model and raw_model_name:
            model_name = raw_model_name
        elif active_engine == "demucs":
            model_name = FOUR_STEM_DEMUCS_MODEL
        else:
            model_name = FOUR_STEM_ROFORMER_MODEL
        print(f"[{job_id}] 🎛️ 4-Stem requested. Selected engine: '{active_engine}' -> model: '{model_name}'")
    elif raw_model_name:
        model_name = raw_model_name
    else:
        model_name = DEFAULT_MODEL
    output_format = job_input.get("output_format", "MP3").upper()
    if output_format not in ["MP3", "WAV", "FLAC"]:
        output_format = "MP3"

    bitrate = str(job_input.get("bitrate", "320k"))
    single_stem = job_input.get("single_stem")  # e.g., "Vocals" or "Instrumental"
    catbox_key = job_input.get("catbox_key", DEFAULT_CATBOX_USER_HASH)

    # Create isolated job workspace to prevent any file collision
    job_temp_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"sep_{job_id}_"))
    job_output_dir = job_temp_dir / "output"
    job_output_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_temp_dir / "input_audio.mp3"

    try:
        # 1. Download input audio
        print(f"[{job_id}] 📥 Downloading audio from: {audio_url}")
        _download_file(audio_url, input_path)
        print(f"[{job_id}] 💾 Saved input audio ({input_path.stat().st_size} bytes)")

        # Quick sanitize pass with ffmpeg to eliminate libmpg123 header junk & decode errors
        sanitized_input = job_temp_dir / "sanitized_input.wav"
        try:
            res = subprocess.run(
                ["ffmpeg", "-y", "-i", str(input_path), "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", str(sanitized_input)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
            if res.returncode == 0 and sanitized_input.exists() and sanitized_input.stat().st_size > 0:
                input_path = sanitized_input
                print(f"[{job_id}] 🧹 Sanitized input audio to clean PCM WAV")
        except Exception as san_err:
            print(f"[{job_id}] ℹ️ Audio sanitize pass skipped: {san_err}")

        # 2. Dynamic model switching if requested model is different
        if model_name != CURRENT_LOADED_MODEL:
            print(f"[{job_id}] 🔄 Switching model from {CURRENT_LOADED_MODEL} -> {model_name}...")
            separator.load_model(model_filename=model_name)
            CURRENT_LOADED_MODEL = model_name

        # Configure separator options for this specific job
        separator.output_dir = str(job_output_dir)
        separator.output_format = output_format
        separator.output_bitrate = bitrate
        separator.output_single_stem = single_stem if single_stem in ["Vocals", "Instrumental"] else None

        # Synchronize output_dir to the underlying model instance if present
        if hasattr(separator, "model_instance") and separator.model_instance:
            separator.model_instance.output_dir = str(job_output_dir)
            if hasattr(separator.model_instance, "output_format"):
                separator.model_instance.output_format = output_format

        # Run GPU separation
        print(f"[{job_id}] ⚙️ Running GPU separation using {CURRENT_LOADED_MODEL} (autocast=FP16)...")
        sep_start = time.time()
        output_files = separator.separate(str(input_path))
        sep_elapsed = time.time() - sep_start
        print(f"[{job_id}] ✅ GPU separation completed in {sep_elapsed:.2f}s! Stems produced: {output_files}")

        # 3. Resolve stem files across multiple candidate locations
        found_stem_files: List[pathlib.Path] = []
        if output_files:
            for file_item in output_files:
                p = pathlib.Path(file_item)
                candidates = [
                    job_output_dir / p.name,
                    p,
                    pathlib.Path("/tmp") / p.name,
                    pathlib.Path("/app") / p.name,
                ]
                for cand in candidates:
                    if cand.exists() and cand.is_file() and cand.stat().st_size > 0:
                        target = job_output_dir / cand.name
                        if cand.resolve() != target.resolve():
                            try:
                                shutil.copy2(cand, target)
                                cand = target
                            except Exception:
                                pass
                        found_stem_files.append(cand)
                        break

        # Fallback filesystem scanning if output_files list resolution didn't locate files
        if not found_stem_files:
            print(f"[{job_id}] 🔍 Scanning {job_output_dir} for output stems...")
            for ext in ("*.mp3", "*.wav", "*.flac"):
                found_stem_files.extend(job_output_dir.glob(ext))

        # Check /tmp fallback scan if still empty
        if not found_stem_files:
            print(f"[{job_id}] 🔍 Fallback scanning /tmp for recently generated stems...")
            for ext in ("*.mp3", "*.wav", "*.flac"):
                for tmp_file in pathlib.Path("/tmp").glob(ext):
                    # Check files modified in the last 2 minutes
                    if time.time() - tmp_file.stat().st_mtime < 120 and tmp_file.stat().st_size > 0:
                        target = job_output_dir / tmp_file.name
                        shutil.copy2(tmp_file, target)
                        found_stem_files.append(target)

        if not found_stem_files:
            return {
                "status": "FAILED",
                "error": "No stems were generated by the separation engine.",
            }

        # 4. Upload stems via fallback chain
        stems_dict: Dict[str, Dict[str, Any]] = {}
        for file_path in found_stem_files:
            file_lower = file_path.name.lower()
            if "vocals" in file_lower or "vocal" in file_lower:
                stem_key = "vocals"
            elif "instrumental" in file_lower or "instrument" in file_lower:
                stem_key = "instrumental"
            elif "drums" in file_lower or "drum" in file_lower:
                stem_key = "drums"
            elif "bass" in file_lower:
                stem_key = "bass"
            elif "guitar" in file_lower:
                stem_key = "guitar"
            elif "piano" in file_lower:
                stem_key = "piano"
            elif "other" in file_lower or "melody" in file_lower or "melodi" in file_lower:
                stem_key = "other"
            else:
                stem_key = file_path.stem

            file_size = file_path.stat().st_size
            print(f"[{job_id}] ⬆️ Uploading stem '{stem_key}' ({file_size} bytes)...")

            try:
                url, storage = upload_with_fallback(file_path, catbox_key)
                stems_dict[stem_key] = {
                    "url": url,
                    "file_name": file_path.name,
                    "size_bytes": file_size,
                    "storage": storage,
                }
                print(f"[{job_id}] ✨ Stem '{stem_key}' uploaded to {storage}: {url}")
            except Exception as up_err:
                print(f"[{job_id}] ⚠️ Upload failed ({up_err}) for '{stem_key}'")
                # Fallback to base64 if small enough (<10MB)
                if file_size < 10 * 1024 * 1024:
                    with open(file_path, "rb") as bf:
                        b64_data = base64.b64encode(bf.read()).decode("utf-8")
                    stems_dict[stem_key] = {
                        "url": "",
                        "base64": b64_data,
                        "file_name": file_path.name,
                        "size_bytes": file_size,
                        "storage": "base64",
                    }
                    print(f"[{job_id}] ⚠️ Base64 fallback active for '{stem_key}'")
                else:
                    stems_dict[stem_key] = {
                        "error": str(up_err),
                        "file_name": file_path.name,
                        "size_bytes": file_size,
                    }

        if not stems_dict or all("url" not in v and "base64" not in v for v in stems_dict.values()):
            return {
                "status": "FAILED",
                "error": "Failed to upload separated stems to storage.",
            }

        total_elapsed = time.time() - start_time
        return {
            "status": "COMPLETED",
            "model": CURRENT_LOADED_MODEL,
            "execution_time_seconds": round(total_elapsed, 2),
            "separation_time_seconds": round(sep_elapsed, 2),
            "stems": stems_dict,
        }

    except Exception as exc:
        print(f"[{job_id}] ❌ Error during separation: {exc}")
        return {
            "status": "FAILED",
            "error": f"Separation failed: {str(exc)}",
        }

    finally:
        # Clean up temporary job folder completely
        if job_temp_dir.exists():
            shutil.rmtree(job_temp_dir, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def concurrency_controller(current_concurrency):
    return MAX_CONCURRENCY


if __name__ == "__main__":
    print(f"🚀 Starting Serverless Worker (Max Concurrency: {MAX_CONCURRENCY})...")
    start_params = {
        "handler": handler,
    }
    if MAX_CONCURRENCY > 1:
        start_params["concurrency_modifier"] = concurrency_controller

    runpod.serverless.start(start_params)
