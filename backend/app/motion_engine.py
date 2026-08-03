from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image


def generate_depth_map(img_np: np.ndarray) -> np.ndarray:
    """Tạo depth map xấp xỉ từ độ sáng và gradient ảnh cho hiệu ứng 3D parallax."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Tạo depth gradient (trên xa, dưới gần)
    height, width = gray.shape
    y_indices, x_indices = np.indices((height, width))
    gradient_depth = (y_indices / height) * 255.0

    # Phối hợp giữa độ sáng (luma) và gradient khoảng cách
    depth = (blurred.astype(np.float32) * 0.4 + gradient_depth * 0.6)
    depth = cv2.GaussianBlur(depth, (21, 21), 0)
    
    # Normalize về khoảng [0, 1]
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    return depth_norm


def render_motion_video(
    image_path: Path,
    output_video_path: Path,
    motion_type: str = "zoom_in",
    num_frames: int = 30,
    fps: int = 15,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Path:
    """Tạo video MP4 từ ảnh với các hiệu ứng chuyển động camera 3D / Parallax."""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    img_np = np.array(img)

    depth_map = generate_depth_map(img_np)
    frames = []

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        
        if progress_callback:
            progress_val = 50.0 + (t * 45.0)  # 50% -> 95%
            progress_callback(round(progress_val, 1), f"Đang render khung hình video {i+1}/{num_frames}...")

        if motion_type == "zoom_in":
            scale = 1.0 + 0.15 * t
            dx, dy = 0.0, 0.0
        elif motion_type == "zoom_out":
            scale = 1.15 - 0.15 * t
            dx, dy = 0.0, 0.0
        elif motion_type == "pan_left":
            scale = 1.08
            dx = (t - 0.5) * 0.08 * width
            dy = 0.0
        elif motion_type == "pan_right":
            scale = 1.08
            dx = (0.5 - t) * 0.08 * width
            dy = 0.0
        elif motion_type == "3d_parallax":
            scale = 1.05
            shift_x = math.sin(t * 2 * math.pi) * 12.0
            shift_y = math.cos(t * 2 * math.pi) * 6.0
            
            # Biến dạng ảnh dựa trên depth map
            map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
            map_x = map_x.astype(np.float32) + (depth_map * shift_x).astype(np.float32)
            map_y = map_y.astype(np.float32) + (depth_map * shift_y).astype(np.float32)
            
            frame_warped = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            frames.append(frame_warped)
            continue
        elif motion_type == "circle_orbit":
            scale = 1.06
            angle = t * 2 * math.pi
            dx = math.cos(angle) * 0.03 * width
            dy = math.sin(angle) * 0.03 * height
        else:
            # Mặc định Zoom in
            scale = 1.0 + 0.12 * t
            dx, dy = 0.0, 0.0

        # Áp dụng Matrix Transformation (Scale + Shift)
        center_x, center_y = width / 2.0, height / 2.0
        matrix = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
        matrix[0, 2] += dx
        matrix[1, 2] += dy

        frame_transformed = cv2.warpAffine(
            img_np, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        frames.append(frame_transformed)

    # Xuất ra file MP4 bằng OpenCV VideoWriter hoặc imageio
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    
    written_with_imageio = False
    try:
        import imageio
        writer = imageio.get_writer(str(output_video_path), fps=fps, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        written_with_imageio = True
    except Exception as e:
        print(f"Notice: imageio export unavailable ({e}), using OpenCV VideoWriter fallback.")

    if not written_with_imageio:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        for frame in frames:
            # OpenCV VideoWriter expects BGR format
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()

    if progress_callback:
        progress_callback(99.0, "Đã đóng gói hoàn tất file video MP4!")

    return output_video_path

