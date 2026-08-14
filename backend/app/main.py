from __future__ import annotations

import base64
import io
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from PIL import Image
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import APP_NAME, IMAGES_DIR, VIDEOS_DIR, ensure_directories
from .database import (
    create_task,
    delete_task,
    get_task,
    initialize_database,
    list_tasks,
)
from .model_runtime import (
    dispatch_generation_task,
    dispatch_image_only_task,
    dispatch_image_to_video_task,
    runtime_state,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_directories()
    initialize_database()
    print(f"=== {APP_NAME} is ready! ===")
    yield


app = FastAPI(
    title=APP_NAME,
    description="API hệ thống chuyển đổi từ văn bản (Text Prompt) sang Ảnh và Video 3D chuyển động.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Văn bản mô tả nội dung cần tạo ảnh và video", example="A futuristic neon city at sunset, 4k hyperrealistic")
    negative_prompt: str = Field("", description="Các chi tiết không mong muốn trong ảnh", example="blurry, bad quality, distortion")
    width: int = Field(1376, ge=256, le=2048, description="Chiều rộng ảnh")
    height: int = Field(768, ge=256, le=2048, description="Chiều cao ảnh")
    num_inference_steps: int = Field(20, ge=1, le=100, description="Số bước khuếch tán sinh ảnh Stable Diffusion")
    motion_type: str = Field("zoom_in", description="Hiệu ứng chuyển động camera 3D: zoom_in, zoom_out, pan_left, pan_right, 3d_parallax, circle_orbit")
    num_frames: int = Field(30, ge=10, le=2400, description="Tổng số khung hình cho video")
    fps: int = Field(15, ge=5, le=60, description="Tốc độ khung hình (khung hình / giây)")


class ImageToVideoRequest(BaseModel):
    image_base64: str = Field(..., description="Dữ liệu ảnh dạng chuỗi Base64 (hỗ trợ cả dạng raw base64 hoặc data:image/png;base64,...)", example="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
    motion_type: str = Field("zoom_in", description="Hiệu ứng chuyển động camera 3D: zoom_in, zoom_out, pan_left, pan_right, 3d_parallax, circle_orbit")
    num_frames: int = Field(30, ge=10, le=2400, description="Tổng số khung hình cho video")
    fps: int = Field(15, ge=5, le=60, description="Tốc độ khung hình (khung hình / giây)")
    prompt: Optional[str] = Field("Base64 Image Video Task", description="Mô tả hoặc tiêu đề cho tác vụ")
    width: Optional[int] = Field(None, ge=256, le=2048, description="Chiều rộng video đầu ra (nếu bỏ trống sẽ lấy theo ảnh gốc)")
    height: Optional[int] = Field(None, ge=256, le=2048, description="Chiều cao video đầu ra (nếu bỏ trống sẽ lấy theo ảnh gốc)")


class GenerateImageOnlyRequest(BaseModel):
    prompt: str = Field(..., description="Văn bản mô tả ảnh cần sinh AI (Chỉ tạo ảnh, không dựng video)", example="A hyperrealistic cybernetic tiger in a futuristic forest")
    negative_prompt: str = Field("", description="Các chi tiết không mong muốn trong ảnh", example="blurry, low quality")
    width: int = Field(1376, ge=256, le=2048, description="Chiều rộng ảnh")
    height: int = Field(768, ge=256, le=2048, description="Chiều cao ảnh")
    num_inference_steps: int = Field(25, ge=1, le=100, description="Số bước khuếch tán sinh ảnh Stable Diffusion")


@app.get("/")
def index():
    return {
        "app": APP_NAME,
        "version": "1.0.0",
        "endpoints": {
            "generate": "POST /generate",
            "generate_image": "POST /generate-image",
            "generate_from_image": "POST /generate-from-image",
            "status": "GET /status/{task_id}",
            "download": "GET /download/{task_id}",
            "image": "GET /image/{task_id}",
            "tasks": "GET /tasks",
            "runtime": "GET /runtime",
            "health": "GET /health",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "app": APP_NAME, "runtime": runtime_state()}


@app.get("/runtime")
def get_runtime():
    return runtime_state()


@app.post("/generate-image", status_code=202)
@app.post("/generate_image", status_code=202)
def generate_image_only(req: GenerateImageOnlyRequest):
    """API nhận Text Prompt và chỉ khởi chạy tiến trình sinh ảnh AI (không tạo video)."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp nội dung text prompt hợp lệ.")

    task_id = str(uuid.uuid4())
    create_task(
        task_id=task_id,
        prompt=req.prompt.strip(),
        negative_prompt=req.negative_prompt.strip(),
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_inference_steps,
    )

    dispatch_image_only_task(
        task_id=task_id,
        prompt=req.prompt.strip(),
        negative_prompt=req.negative_prompt.strip(),
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_inference_steps,
    )

    return {
        "task_id": task_id,
        "status": "queued",
        "detail": "Tác vụ sinh ảnh đã được tiếp nhận và đang được xử lý...",
        "status_url": f"/status/{task_id}",
        "image_url": f"/image/{task_id}",
    }


@app.post("/generate", status_code=202)
def generate_text_to_video(req: GenerateRequest):
    """API chính nhận Text Prompt và khởi chạy tiến trình sinh Ảnh + Video không đồng bộ."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp nội dung text prompt.")

    task_id = str(uuid.uuid4())
    task = create_task(
        task_id=task_id,
        prompt=req.prompt.strip(),
        negative_prompt=req.negative_prompt.strip(),
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_inference_steps,
        motion_type=req.motion_type,
        num_frames=req.num_frames,
        fps=req.fps,
    )

    dispatch_generation_task(
        task_id=task_id,
        prompt=req.prompt.strip(),
        negative_prompt=req.negative_prompt.strip(),
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_inference_steps,
        motion_type=req.motion_type,
        num_frames=req.num_frames,
        fps=req.fps,
    )

    return {
        "task_id": task_id,
        "status": "queued",
        "detail": "Tác vụ đã được tiếp nhận và đang được xử lý...",
        "status_url": f"/status/{task_id}",
        "download_url": f"/download/{task_id}",
        "image_url": f"/image/{task_id}",
    }


@app.post("/generate-from-image", status_code=202)
def generate_video_from_base64_image(req: ImageToVideoRequest):
    """API nhận dữ liệu ảnh dạng chuỗi Base64 và khởi chạy tiến trình tạo video hiệu ứng 3D."""
    if not req.image_base64.strip():
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp dữ liệu chuỗi Base64 ảnh.")

    base64_str = req.image_base64.strip()
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = img.size
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Dữ liệu ảnh Base64 không hợp lệ: {str(exc)}")

    width = req.width if req.width is not None else orig_w
    height = req.height if req.height is not None else orig_h

    task_id = str(uuid.uuid4())
    image_filename = f"{task_id}.png"
    image_path = IMAGES_DIR / image_filename
    img.save(image_path, format="PNG")

    prompt_text = req.prompt.strip() if req.prompt else "Uploaded Base64 Image"

    task = create_task(
        task_id=task_id,
        prompt=prompt_text,
        negative_prompt="",
        width=width,
        height=height,
        num_inference_steps=0,
        motion_type=req.motion_type,
        num_frames=req.num_frames,
        fps=req.fps,
    )

    dispatch_image_to_video_task(
        task_id=task_id,
        image_filename=image_filename,
        motion_type=req.motion_type,
        num_frames=req.num_frames,
        fps=req.fps,
        width=width,
        height=height,
    )

    return {
        "task_id": task_id,
        "status": "queued",
        "detail": "Đã nhận ảnh Base64 và khởi chạy tiến trình dựng video 3D...",
        "status_url": f"/status/{task_id}",
        "download_url": f"/download/{task_id}",
        "image_url": f"/image/{task_id}",
    }


@app.get("/status/{task_id}")
def check_status(task_id: str):
    """Kiểm tra tiến độ xử lý của tác vụ."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Không tìm thấy task_id yêu cầu.")
    return task


@app.get("/download/{task_id}")
def download_video(task_id: str):
    """Tải xuống file video kết quả (.mp4)."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Không tìm thấy task_id yêu cầu.")
    
    if task["status"] != "completed" or not task["video_filename"]:
        raise HTTPException(
            status_code=400,
            detail=f"Video chưa sẵn sàng. Trạng thái hiện tại: {task['status']} ({task['detail']})"
        )

    video_path = VIDEOS_DIR / task["video_filename"]
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="File video kết quả không tồn tại trên hệ thống.")

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"video_{task_id[:8]}.mp4",
    )


@app.get("/image/{task_id}")
def get_generated_image(task_id: str):
    """Xem hoặc tải ảnh trung gian (.png) được tạo từ text prompt."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Không tìm thấy task_id yêu cầu.")

    if not task["image_filename"]:
        raise HTTPException(status_code=400, detail="Ảnh chưa được khởi tạo.")

    image_path = IMAGES_DIR / task["image_filename"]
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="File ảnh không tồn tại.")

    return FileResponse(
        path=str(image_path),
        media_type="image/png",
        filename=f"image_{task_id[:8]}.png",
    )


@app.get("/tasks")
def list_all_tasks(limit: int = 50):
    """Lấy danh sách các tác vụ gần đây."""
    return list_tasks(limit=limit)


@app.delete("/tasks/{task_id}")
def remove_task(task_id: str):
    """Xóa một tác vụ và các file đính kèm."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Không tìm thấy task_id.")

    if task.get("image_filename"):
        (IMAGES_DIR / task["image_filename"]).unlink(missing_ok=True)
    if task.get("video_filename"):
        (VIDEOS_DIR / task["video_filename"]).unlink(missing_ok=True)

    success = delete_task(task_id)
    return {"ok": success, "task_id": task_id}
