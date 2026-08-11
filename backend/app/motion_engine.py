from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional, Tuple

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


def ease_in_out_cubic(t: float) -> float:
    """Nội suy mượt Cubic Smoothstep Ease-In-Out (0.0 -> 1.0) cho camera chuyển động cinematic."""
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def detect_focal_center(img_np: np.ndarray, depth_map: np.ndarray) -> Tuple[float, float]:
    """Tự động xác định tiêu điểm chính (focal center) của bức ảnh.
    Kết hợp giữa Depth map (vùng gần), Sobel Gradient (đường nét nổi bật) và Luma,
    áp dụng spatial Gaussian prior ưu tiên khu vực trung tâm và phía trên.
    Trả về tọa độ pixel (focal_x, focal_y).
    """
    height, width = img_np.shape[:2]
    
    y_indices, x_indices = np.indices((height, width), dtype=np.float32)
    norm_x = x_indices / float(width)
    norm_y = y_indices / float(height)

    # Spatial Gaussian prior ưu tiên khu vực giữa & trên (x=0.5, y=0.45)
    spatial_weights = np.exp(-(((norm_x - 0.5) ** 2) / (2 * 0.28 ** 2) + ((norm_y - 0.45) ** 2) / (2 * 0.28 ** 2)))

    # Sobel Edge Saliency
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(grad_x, grad_y)
    edge_norm = edge_mag / (edge_mag.max() + 1e-6)

    # Combined weight map
    combined_weight = (depth_map * 0.45 + edge_norm * 0.35 + gray * 0.20) * spatial_weights

    # Lấy top 25% giá trị cao nhất để tính Center of Mass tiêu điểm
    threshold = np.percentile(combined_weight, 75)
    mask = combined_weight >= threshold

    if np.sum(mask) > 0:
        focal_x_norm = float(np.sum(norm_x[mask] * combined_weight[mask]) / np.sum(combined_weight[mask]))
        focal_y_norm = float(np.sum(norm_y[mask] * combined_weight[mask]) / np.sum(combined_weight[mask]))
    else:
        focal_x_norm, focal_y_norm = 0.5, 0.45

    # Giới hạn an toàn trong khoảng [0.2, 0.8] tránh quá sát mép ảnh
    focal_x_norm = max(0.2, min(0.8, focal_x_norm))
    focal_y_norm = max(0.2, min(0.8, focal_y_norm))

    return focal_x_norm * width, focal_y_norm * height


def build_camera_matrix(
    width: int,
    height: int,
    scale: float,
    center_x: float,
    center_y: float,
) -> Tuple[np.ndarray, float, float]:
    """Tạo ma trận biến đổi camera (Scale + Translate) với cơ chế tự động Clamp tiêu điểm.
    Đảm bảo 100% hình ảnh luôn nằm hoàn toàn trong khung canvas, không bị lộ viền, trượt mép hay sô ảnh.
    """
    # Margin an toàn scale >= 1.002 ngăn chặn tuyệt đối sô viền mép 1px
    scale = max(1.002, float(scale))

    half_box_w = (width / 2.0) / scale
    half_box_h = (height / 2.0) / scale

    # Clamp center_x & center_y để khung view-box trích xuất từ ảnh gốc luôn nằm trong [0, width] x [0, height]
    clamped_x = max(half_box_w, min(width - half_box_w, float(center_x)))
    clamped_y = max(half_box_h, min(height - half_box_h, float(center_y)))

    # Ma trận Affine: Ánh xạ view-box trung tâm (clamped_x, clamped_y) vào chính giữa canvas (width/2, height/2)
    tx = (width / 2.0) - scale * clamped_x
    ty = (height / 2.0) - scale * clamped_y

    matrix = np.array([
        [scale, 0.0, tx],
        [0.0, scale, ty]
    ], dtype=np.float32)

    return matrix, clamped_x, clamped_y


def fit_image_aspect_fill(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize ảnh theo thuật toán Aspect Fill (phủ kín canvas, giữ nguyên tỉ lệ khung hình) 
    rồi Center Crop về đúng kích thước target_w x target_h.
    Ngăn chặn tuyệt đối vỡ/méo ảnh và loại bỏ hoàn toàn viền đen/padding.
    """
    orig_w, orig_h = img.size
    if (orig_w, orig_h) == (target_w, target_h):
        return img

    ratio_in = orig_w / float(orig_h)
    ratio_out = target_w / float(target_h)

    if abs(ratio_in - ratio_out) < 1e-3:
        return img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    scale = max(target_w / float(orig_w), target_h / float(orig_h))
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    scaled_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    return scaled_img.crop((left, top, right, bottom))


def render_motion_video(
    image_path: Path,
    output_video_path: Path,
    motion_type: str = "zoom_in",
    num_frames: int = 30,
    fps: int = 15,
    zoom_center: Optional[Tuple[float, float]] = None,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Path:
    """Tạo video MP4 từ ảnh với các hiệu ứng chuyển động camera 3D / Parallax mượt mà.
    
    Tính năng nâng cấp:
    - Hỗ trợ giữ chuẩn độ phân giải 16:9 (1376x768) hoặc tùy chỉnh.
    - Ease-In-Out Smoothstep camera transitions.
    - Tự động Aspect Fill + Center Crop loại bỏ vỡ/méo hình và viền đen.
    - Clamping ma trận camera ngăn chặn tuyệt đối lộ viền/gương hay sô ảnh.
    - Subpixel Interpolation bicubic sắc nét.
    """
    img = Image.open(image_path).convert("RGB")

    if target_width is not None and target_height is not None:
        tw = int(target_width)
        th = int(target_height)
        img = fit_image_aspect_fill(img, tw, th)

    width, height = img.size

    # Đảm bảo chiều rộng và chiều cao luôn là số chẵn (yêu cầu chuẩn H.264 codec)
    if width % 2 != 0 or height % 2 != 0:
        width = width - (width % 2)
        height = height - (height % 2)
        img = img.crop((0, 0, width, height))

    img_np = np.array(img)
    depth_map = generate_depth_map(img_np)
    
    # Xác định tiêu điểm zoom (tọa độ pixel)
    if zoom_center is not None:
        zx, zy = zoom_center
        focal_x = max(0.1, min(0.9, float(zx))) * width
        focal_y = max(0.1, min(0.9, float(zy))) * height
    else:
        focal_x, focal_y = detect_focal_center(img_np, depth_map)

    center_orig_x, center_orig_y = width / 2.0, height / 2.0
    frames = []

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        e = ease_in_out_cubic(t)

        if progress_callback:
            progress_val = 50.0 + (t * 45.0)  # 50% -> 95%
            progress_callback(round(progress_val, 1), f"Đang render khung hình video {i+1}/{num_frames}...")

        if motion_type == "zoom_in":
            # Phóng to mượt mà từ 1.0 -> 1.22, hướng dần về tiêu điểm (focal_x, focal_y)
            scale = 1.0 + 0.22 * e
            cx = center_orig_x + (focal_x - center_orig_x) * e
            cy = center_orig_y + (focal_y - center_orig_y) * e
            matrix, _, _ = build_camera_matrix(width, height, scale, cx, cy)
            frame_transformed = cv2.warpAffine(
                img_np, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

        elif motion_type == "zoom_out":
            # Thu nhỏ mượt mà từ 1.22 (tại tiêu điểm) -> 1.0 (toàn cảnh gốc 100%)
            scale = 1.22 - 0.22 * e
            cx = focal_x + (center_orig_x - focal_x) * e
            cy = focal_y + (center_orig_y - focal_y) * e
            matrix, _, _ = build_camera_matrix(width, height, scale, cx, cy)
            frame_transformed = cv2.warpAffine(
                img_np, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

        elif motion_type == "pan_left":
            # Lia camera mượt từ phải qua trái trong khoảng an toàn
            scale = 1.15
            half_w = (width / 2.0) / scale
            min_cx = half_w
            max_cx = width - half_w
            cx = max_cx - (max_cx - min_cx) * e
            cy = focal_y
            matrix, _, _ = build_camera_matrix(width, height, scale, cx, cy)
            frame_transformed = cv2.warpAffine(
                img_np, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

        elif motion_type == "pan_right":
            # Lia camera mượt từ trái qua phải trong khoảng an toàn
            scale = 1.15
            half_w = (width / 2.0) / scale
            min_cx = half_w
            max_cx = width - half_w
            cx = min_cx + (max_cx - min_cx) * e
            cy = focal_y
            matrix, _, _ = build_camera_matrix(width, height, scale, cx, cy)
            frame_transformed = cv2.warpAffine(
                img_np, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

        elif motion_type == "3d_parallax":
            # Kết hợp Zoom 1.12 an toàn và depth displacement mượt
            scale = 1.12
            matrix, _, _ = build_camera_matrix(width, height, scale, focal_x, focal_y)
            base_img = cv2.warpAffine(img_np, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            base_depth = cv2.warpAffine(depth_map, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            shift_x = math.sin(e * 2 * math.pi) * 8.0
            shift_y = math.cos(e * 2 * math.pi) * 4.0
            
            map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
            map_x = map_x.astype(np.float32) + (base_depth * shift_x).astype(np.float32)
            map_y = map_y.astype(np.float32) + (base_depth * shift_y).astype(np.float32)
            
            frame_transformed = cv2.remap(base_img, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        elif motion_type == "circle_orbit":
            # Camera quay chuyển động tròn quanh tiêu điểm focal
            scale = 1.14
            radius_x = 0.04 * width
            radius_y = 0.04 * height
            angle = e * 2 * math.pi
            cx = focal_x + math.cos(angle) * radius_x
            cy = focal_y + math.sin(angle) * radius_y
            matrix, _, _ = build_camera_matrix(width, height, scale, cx, cy)
            frame_transformed = cv2.warpAffine(
                img_np, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

        else:
            # Default Zoom in
            scale = 1.0 + 0.22 * e
            cx = center_orig_x + (focal_x - center_orig_x) * e
            cy = center_orig_y + (focal_y - center_orig_y) * e
            matrix, _, _ = build_camera_matrix(width, height, scale, cx, cy)
            frame_transformed = cv2.warpAffine(
                img_np, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
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


