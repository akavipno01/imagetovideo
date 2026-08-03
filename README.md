# Text to Image & Video Generator (Text-to-Video API)

Hệ thống API và Google Colab Notebook cho phép chuyển đổi từ **Văn bản (Text Prompt)** sang **Ảnh (PNG)** và **Video chuyển động 3D / Parallax (MP4)**.

Mô hình kiến trúc được thiết kế theo cấu trúc dự án `voicecolab`, tích hợp các thuật toán sinh ảnh từ `Stable Diffusion / Diffusers` và công nghệ dựng hiệu ứng camera chuyển động 3D từ `3d-photo-inpainting`.

---

## 🚀 Tính năng nổi bật

- 📝 **Text to Image**: Tự động chuyển văn bản tiếng Anh/tiếng Việt thành bức ảnh phong cảnh, nhân vật hoặc nghệ thuật bằng mô hình Diffusion.
- 🎬 **Image to 3D Video**: Tự động dựng video MP4 chuyển động camera 3D chuyên nghiệp (Zoom In, Zoom Out, Pan Left/Right, 3D Parallax, Circle Orbit).
- ⚡ **Hàng chờ không đồng bộ (Async Queue)**: Nhận yêu cầu ngay lập tức, trả về `task_id` và cập nhật phần trăm tiến độ `%` chi tiết.
- ☁️ **Google Colab GPU & Cloudflare Tunnel**: Khởi chạy dễ dàng trên GPU miễn phí của Google Colab và mở cổng HTTPS công khai không cần mở port mạng cá nhân.
- 💾 **SQLite Database**: Lưu trữ lịch sử tất cả các tác vụ sinh video, thời gian tạo, đường dẫn file và trạng thái.

---

## 🛠 Hướng dẫn chạy trên Google Colab GPU

1. Tải notebook `Image_To_Video_Colab.ipynb` lên **Google Colab** (hoặc mở trực tiếp qua GitHub).
2. Vào **Runtime** -> **Change runtime type** -> Chọn **GPU**.
3. Chọn **Run all (Chạy tất cả)**.
4. Chờ đường truyền **Cloudflare Tunnel** in ra liên kết dạng:
   `https://xxx.trycloudflare.com`

---

## 💻 Hướng dẫn chạy Backend ở máy cá nhân (Local)

### 1. Cài đặt môi trường
```bash
cd d:/Thang/imagetovideo/backend
pip install -r requirements.txt
```

### 2. Khởi chạy Server
```bash
python run.py
```
Server sẽ chạy mặc định tại: `http://127.0.0.1:3930`

---

## 📌 Hướng dẫn gọi API (API Endpoints)

### 1. Gửi yêu cầu sinh Video từ Text Prompt
- **Endpoint**: `POST /generate`
- **Content-Type**: `application/json`
- **Body Request**:
```json
{
  "prompt": "A majestic glowing dragon flying over misty mountains at sunset, 4k cinematic fantasy",
  "negative_prompt": "blurry, low quality, distortion",
  "width": 512,
  "height": 512,
  "num_inference_steps": 20,
  "motion_type": "zoom_in",
  "num_frames": 30,
  "fps": 15
}
```
- **Response**:
```json
{
  "task_id": "c9a7d2e3-4f51-4123-88ab-9912bc05f321",
  "status": "queued",
  "detail": "Tác vụ đã được tiếp nhận và đang được xử lý...",
  "status_url": "/status/c9a7d2e3-4f51-4123-88ab-9912bc05f321",
  "download_url": "/download/c9a7d2e3-4f51-4123-88ab-9912bc05f321",
  "image_url": "/image/c9a7d2e3-4f51-4123-88ab-9912bc05f321"
}
```

### 2. Kiểm tra tiến độ tác vụ
- **Endpoint**: `GET /status/{task_id}`
- **Response**:
```json
{
  "task_id": "c9a7d2e3-4f51-4123-88ab-9912bc05f321",
  "prompt": "A majestic glowing dragon...",
  "status": "generating_video",
  "progress": 75.0,
  "detail": "Đang render khung hình video 15/30...",
  "image_filename": "c9a7d2e3-4f51-4123-88ab-9912bc05f321.png",
  "video_filename": null
}
```

### 3. Tải Video MP4 Kết Quả
- **Endpoint**: `GET /download/{task_id}`
- Trả về trực tiếp file `video_c9a7d2e3.mp4` để tải về hoặc xem trên trình duyệt.

### 4. Xem/Tải Ảnh Trung Gian
- **Endpoint**: `GET /image/{task_id}`
- Trả về file ảnh `.png` vừa được tạo từ Text Prompt.

### 5. Danh Sách Lịch Sử Tác Vụ
- **Endpoint**: `GET /tasks`

### 6. Kiểm Tra Trạng Thái GPU / Server
- **Endpoint**: `GET /health` hoặc `GET /runtime`

---

## 🎨 Danh sách kiểu chuyển động Camera 3D (`motion_type`)

| Tên `motion_type` | Mô tả hiệu ứng |
|---|---|
| `zoom_in` | Mắt camera tự động phóng to dần vào tâm ảnh |
| `zoom_out` | Mắt camera thu nhỏ dần ra xa |
| `pan_left` | Camera lia mượt sang bên trái |
| `pan_right` | Camera lia mượt sang bên phải |
| `3d_parallax` | Tạo hiệu ứng chiều sâu 3D lắc góc camera theo bản đồ độ sâu |
| `circle_orbit` | Camera lượn vòng tròn 3D nghệ thuật |
