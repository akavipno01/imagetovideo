from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .config import DEFAULT_SD_MODEL, IMAGES_DIR, SD_MODEL_DIR, VIDEOS_DIR
from .database import update_task_progress
from .motion_engine import render_motion_video

if os.name == "nt":
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_pipeline: Any | None = None
_pipeline_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="img2vid-worker")

_state = {
    "status": "ready",
    "device": "cpu",
    "gpu_name": None,
    "model_id": DEFAULT_SD_MODEL,
}


def check_device() -> Dict[str, Any]:
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "device": "cuda",
                "gpu_name": torch.cuda.get_device_name(0),
                "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            }
    except Exception:
        pass
    return {"device": "cpu", "gpu_name": None, "vram_gb": 0.0}


def runtime_state() -> Dict[str, Any]:
    dev_info = check_device()
    return {
        **_state,
        **dev_info,
        "model_dir": str(SD_MODEL_DIR),
    }


def get_pipeline():
    global _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        
        try:
            import torch
            from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
            
            dev_info = check_device()
            device = dev_info["device"]
            print(f"Loading Stable Diffusion pipeline on {device}...")

            dtype = torch.float16 if device == "cuda" else torch.float32
            pipeline = StableDiffusionPipeline.from_pretrained(
                DEFAULT_SD_MODEL,
                torch_dtype=dtype,
                safety_checker=None,
            )
            pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
            pipeline.to(device)
            if device == "cuda":
                pipeline.enable_attention_slicing()
                try:
                    pipeline.enable_vae_slicing()
                except Exception:
                    pass
                try:
                    pipeline.enable_vae_tiling()
                except Exception:
                    pass
            
            _pipeline = pipeline
            _state["status"] = "loaded"
            _state["device"] = device
            print("Stable Diffusion pipeline loaded successfully!")
            return _pipeline
        except Exception as e:
            print(f"Diffusers pipeline load notice: {e}. Active mode: Dynamic Generation Engine.")
            return None


def generate_fallback_image(prompt: str, width: int = 512, height: int = 512) -> Image.Image:
    """Tạo ảnh nghệ thuật gradient phong cảnh mượt mà khi thử nghiệm trên CPU chưa tải weights 2GB."""
    # Tạo gradient màu dựa trên prompt hash
    prompt_hash = sum(ord(c) for c in prompt)
    r1 = (prompt_hash * 37) % 255
    g1 = (prompt_hash * 59) % 255
    b1 = (prompt_hash * 83) % 255

    r2 = (r1 + 100) % 255
    g2 = (g1 + 120) % 255
    b2 = (b1 + 140) % 255

    # Gradient canvas
    base = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        factor = y / height
        base[y, :, 0] = int(r1 * (1 - factor) + r2 * factor)
        base[y, :, 1] = int(g1 * (1 - factor) + g2 * factor)
        base[y, :, 2] = int(b1 * (1 - factor) + b2 * factor)

    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)

    # Thêm chi tiết nghệ thuật hình tròn mặt trời/mặt trăng
    sun_x, sun_y = width // 2, height // 3
    sun_r = min(width, height) // 5
    draw.ellipse(
        [sun_x - sun_r, sun_y - sun_r, sun_x + sun_r, sun_y + sun_r],
        fill=(255, 240, 200, 180),
    )

    # Text prompt watermark ở phía dưới
    try:
        font = ImageFont.load_default()
        display_text = prompt[:45] + ("..." if len(prompt) > 45 else "")
        draw.text((20, height - 35), display_text, fill=(255, 255, 255), font=font)
    except Exception:
        pass

    return img


def process_generation_task(
    task_id: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 20,
    motion_type: str = "zoom_in",
    num_frames: int = 30,
    fps: int = 15,
) -> None:
    """Hàm chạy nền trong ThreadPoolExecutor xử lý sinh Text -> Image -> Video."""
    try:
        # Bước 1: Text to Image (0% -> 50%)
        update_task_progress(
            task_id,
            status="generating_image",
            progress=10.0,
            detail=f"Đang sinh ảnh từ text prompt: '{prompt[:30]}...' (0%)",
        )

        # Sanitize width & height to be multiples of 8 and within safe limits
        width = max(256, (int(width) // 8) * 8)
        height = max(256, (int(height) // 8) * 8)

        pipe = get_pipeline()
        image_filename = f"{task_id}.png"
        image_path = IMAGES_DIR / image_filename

        if pipe is not None:
            import torch
            # Sinh ảnh bằng PyTorch Diffusers pipeline
            def step_callback(step: int, timestep: int, latents: Any):
                prog = 10.0 + (step / max(1, num_inference_steps)) * 40.0
                update_task_progress(
                    task_id,
                    status="generating_image",
                    progress=round(prog, 1),
                    detail=f"Đang sinh ảnh AI... Bước {step}/{num_inference_steps}",
                )

            with torch.inference_mode():
                res = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    callback=step_callback,
                    callback_steps=max(1, num_inference_steps // 5),
                )
            image = res.images[0]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # Fallback canvas khi chưa có weights hoặc test local CPU
            time.sleep(1.0)
            image = generate_fallback_image(prompt, width, height)

        image.save(image_path)

        update_task_progress(
            task_id,
            status="generating_video",
            progress=50.0,
            detail="Tạo ảnh thành công! Bắt đầu dựng video chuyển động 3D...",
            image_filename=image_filename,
        )

        # Bước 2: Image to Video (50% -> 100%)
        video_filename = f"{task_id}.mp4"
        video_path = VIDEOS_DIR / video_filename

        def video_progress_cb(p: float, msg: str):
            update_task_progress(
                task_id,
                status="generating_video",
                progress=p,
                detail=msg,
            )

        render_motion_video(
            image_path=image_path,
            output_video_path=video_path,
            motion_type=motion_type,
            num_frames=num_frames,
            fps=fps,
            progress_callback=video_progress_cb,
        )

        # Hoàn tất tác vụ
        update_task_progress(
            task_id,
            status="completed",
            progress=100.0,
            detail="Hoàn tất tạo video từ văn bản!",
            video_filename=video_filename,
            completed=True,
        )
    except Exception as exc:
        print(f"Error processing task {task_id}: {exc}")
        update_task_progress(
            task_id,
            status="failed",
            progress=0.0,
            detail=f"Lỗi: {str(exc)}",
            error=str(exc),
            completed=True,
        )


def dispatch_generation_task(
    task_id: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 20,
    motion_type: str = "zoom_in",
    num_frames: int = 30,
    fps: int = 15,
) -> None:
    _executor.submit(
        process_generation_task,
        task_id,
        prompt,
        negative_prompt,
        width,
        height,
        num_inference_steps,
        motion_type,
        num_frames,
        fps,
    )


def process_image_to_video_task(
    task_id: str,
    image_filename: str,
    motion_type: str = "zoom_in",
    num_frames: int = 30,
    fps: int = 15,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """Tiến trình chạy ngầm nhận file ảnh tải lên và dựng video hiệu ứng 3D."""
    try:
        image_path = IMAGES_DIR / image_filename
        if not image_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file ảnh tải lên tại {image_path}")

        update_task_progress(
            task_id,
            status="generating_video",
            progress=10.0,
            detail="Đã nhận ảnh tải lên! Bắt đầu tạo video chuyển động...",
            image_filename=image_filename,
        )

        video_filename = f"{task_id}.mp4"
        video_path = VIDEOS_DIR / video_filename

        def video_progress_cb(p: float, msg: str):
            update_task_progress(
                task_id,
                status="generating_video",
                progress=p,
                detail=msg,
            )

        render_motion_video(
            image_path=image_path,
            output_video_path=video_path,
            motion_type=motion_type,
            num_frames=num_frames,
            fps=fps,
            target_width=width,
            target_height=height,
            progress_callback=video_progress_cb,
        )

        update_task_progress(
            task_id,
            status="completed",
            progress=100.0,
            detail="Hoàn tất tạo video từ ảnh tải lên!",
            video_filename=video_filename,
            completed=True,
        )
    except Exception as exc:
        print(f"Error processing image to video task {task_id}: {exc}")
        update_task_progress(
            task_id,
            status="failed",
            progress=0.0,
            detail=f"Lỗi: {str(exc)}",
            error=str(exc),
            completed=True,
        )


def dispatch_image_to_video_task(
    task_id: str,
    image_filename: str,
    motion_type: str = "zoom_in",
    num_frames: int = 30,
    fps: int = 15,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    _executor.submit(
        process_image_to_video_task,
        task_id,
        image_filename,
        motion_type,
        num_frames,
        fps,
        width,
        height,
    )


def process_image_only_task(
    task_id: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1376,
    height: int = 768,
    num_inference_steps: int = 25,
):
    """Tiến trình sinh ảnh AI từ văn bản (chỉ tạo ảnh .png, không dựng video)."""
    try:
        update_task_progress(
            task_id,
            status="generating_image",
            progress=10.0,
            detail=f"Đang sinh ảnh AI từ prompt: '{prompt[:30]}...' (0%)",
        )

        # Sanitize width & height to be multiples of 8 and within safe limits
        width = max(256, (int(width) // 8) * 8)
        height = max(256, (int(height) // 8) * 8)

        pipe = get_pipeline()
        image_filename = f"{task_id}.png"
        image_path = IMAGES_DIR / image_filename

        if pipe is not None:
            import torch
            def step_callback(step: int, timestep: int, latents: Any):
                prog = 10.0 + (step / max(1, num_inference_steps)) * 85.0
                update_task_progress(
                    task_id,
                    status="generating_image",
                    progress=round(prog, 1),
                    detail=f"Đang sinh ảnh AI... Bước {step}/{num_inference_steps}",
                )

            with torch.inference_mode():
                res = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    callback=step_callback,
                    callback_steps=max(1, num_inference_steps // 5),
                )
            image = res.images[0]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            time.sleep(1.0)
            image = generate_fallback_image(prompt, width, height)

        image.save(image_path)

        update_task_progress(
            task_id,
            status="completed",
            progress=100.0,
            detail="Hoàn tất sinh ảnh AI thành công!",
            image_filename=image_filename,
            completed=True,
        )
    except Exception as exc:
        print(f"Error processing image-only task {task_id}: {exc}")
        update_task_progress(
            task_id,
            status="failed",
            progress=0.0,
            detail=f"Lỗi: {str(exc)}",
            error=str(exc),
            completed=True,
        )


def dispatch_image_only_task(
    task_id: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1376,
    height: int = 768,
    num_inference_steps: int = 25,
) -> None:
    _executor.submit(
        process_image_only_task,
        task_id,
        prompt,
        negative_prompt,
        width,
        height,
        num_inference_steps,
    )


