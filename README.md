# Text to Image & Base64 Image to Video Generator (Text/Image-to-Video API)

Hệ thống API và Google Colab Notebook cho phép chuyển đổi từ **Văn bản (Text Prompt)** hoặc **Ảnh Base64 (PNG/JPG/WEBP)** sang **Video chuyển động 3D / Parallax (MP4)**.

Mô hình kiến trúc được thiết kế theo cấu trúc dự án `voicecolab`, tích hợp các thuật toán sinh ảnh từ `Stable Diffusion / Diffusers` và công nghệ dựng hiệu ứng camera chuyển động 3D từ `3d-photo-inpainting` / `OpenCV`.

---

## 🚀 Tính năng nổi bật

- 📝 **Text to Video**: Tự động chuyển văn bản (Text Prompt) thành ảnh AI và render thành video chuyển động 3D.
- 🖼️ **Base64 Image to Video**: Cho phép gửi trực tiếp dữ liệu ảnh dạng chuỗi **Base64** để tạo video chuyển động 3D lập tức.
- 🎬 **Hiệu ứng Camera 3D Đa dạng**: Hỗ trợ 6 kiểu chuyển động camera (`zoom_in`, `zoom_out`, `pan_left`, `pan_right`, `3d_parallax`, `circle_orbit`).
- ⏱️ **Độ phân giải & Thời lượng cao**: Hỗ trợ kích thước khung hình lên tới **2048px** và số lượng khung hình lên tới **600 frames** (tối đa ~40 giây video).
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
cd backend
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
  "width": 1024,
  "height": 720,
  "num_inference_steps": 20,
  "motion_type": "zoom_in",
  "num_frames": 225,
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

---

### 2. Gửi yêu cầu CHỈ sinh Ảnh AI (Prompt to Image only)
- **Endpoint**: `POST /generate-image`
- **Content-Type**: `application/json`
- **Body Request**:
```json
{
  "prompt": "A hyperrealistic cybernetic tiger in a futuristic forest",
  "negative_prompt": "blurry, low quality",
  "width": 1376,
  "height": 768,
  "num_inference_steps": 25
}
```
- **Response**:
```json
{
  "task_id": "b1a2c3d4-5e6f-7890-abcd-ef1234567890",
  "status": "queued",
  "detail": "Tác vụ sinh ảnh đã được tiếp nhận và đang được xử lý...",
  "status_url": "/status/b1a2c3d4-5e6f-7890-abcd-ef1234567890",
  "image_url": "/image/b1a2c3d4-5e6f-7890-abcd-ef1234567890"
}
```

---

### 3. Gửi yêu cầu sinh Video từ Ảnh Base64
- **Endpoint**: `POST /generate-from-image`
- **Content-Type**: `application/json`
- **Body Request**:
```json
{
  "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
  "motion_type": "3d_parallax",
  "num_frames": 225,
  "fps": 15,
  "prompt": "Tác vụ video từ ảnh Base64"
}
```
- **Response**:
```json
{
  "task_id": "8f3b41d2-9c12-4011-a889-112233445566",
  "status": "queued",
  "detail": "Đã nhận ảnh Base64 và khởi chạy tiến trình dựng video 3D...",
  "status_url": "/status/8f3b41d2-9c12-4011-a889-112233445566",
  "download_url": "/download/8f3b41d2-9c12-4011-a889-112233445566",
  "image_url": "/image/8f3b41d2-9c12-4011-a889-112233445566"
}
```

---

### 3. Kiểm tra tiến độ tác vụ
- **Endpoint**: `GET /status/{task_id}`
- **Response**:
```json
{
  "task_id": "c9a7d2e3-4f51-4123-88ab-9912bc05f321",
  "prompt": "A majestic glowing dragon...",
  "status": "generating_video",
  "progress": 75.0,
  "detail": "Đang render khung hình video 168/225...",
  "image_filename": "c9a7d2e3-4f51-4123-88ab-9912bc05f321.png",
  "video_filename": null
}
```

---

### 4. Tải Video MP4 Kết Quả
- **Endpoint**: `GET /download/{task_id}`
- Trả về trực tiếp file `video_c9a7d2e3.mp4` để tải về hoặc xem trên trình duyệt.

---

### 5. Xem/Tải Ảnh Trung Gian
- **Endpoint**: `GET /image/{task_id}`
- Trả về file ảnh `.png` được tạo từ Text Prompt hoặc ảnh tải lên.

---

### 6. Danh Sách Lịch Sử Tác Vụ
- **Endpoint**: `GET /tasks`

---

### 7. Kiểm Tra Trạng Thái GPU / Server
- **Endpoint**: `GET /health` hoặc `GET /runtime`

---

## 🎨 Danh sách kiểu chuyển động Camera 3D (`motion_type`)

| Tên `motion_type` | Mô tả hiệu ứng |
|---|---|
| `zoom_in` | Mắt camera tự động phóng to dần vào tâm ảnh |
| `zoom_out` | Mắt camera thu nhỏ dần ra xa |
| `pan_left` | Camera lia mượt sang bên trái |
| `pan_right` | Camera lia mượt sang bên phải |
| `3d_parallax` | Tạo hiệu ứng chiều sâu 3D lắc góc camera theo bản đồ độ sâu (Depth Map) |
| `circle_orbit` | Camera lượn vòng tròn 3D nghệ thuật xung quanh tâm |

---

## 📐 Công thức tính thời lượng Video

$$\text{Thời lượng video (giây)} = \frac{\text{num\_frames}}{\text{fps}}$$

- **Video 15 giây (FPS 15)**: Set `"fps": 15` và `"num_frames": 225` ($15 \times 15 = 225$).
- **Video 15 giây (FPS 24 - Chuẩn điện ảnh)**: Set `"fps": 24` và `"num_frames": 360` ($24 \times 15 = 360$).
- **Video 30 giây (FPS 15)**: Set `"fps": 15` và `"num_frames": 450` ($15 \times 30 = 450$).

---

## 🖥️ Ứng dụng Desktop GUI Client (`client_app.py`)

Dự án cung cấp sẵn một phần mềm giao diện đồ họa Python (Desktop GUI Client) hỗ trợ tự động hóa render hàng loạt:

### Cách khởi chạy ứng dụng GUI:
```bash
python client_app.py
```

### Các tính năng trên giao diện:
1. **Ô điền URL Server**: Nhập địa chỉ Cloudflare Tunnel (vd: `https://tobacco-went-harper-que.trycloudflare.com`) kèm nút **Kiểm Tra Kết Nối**.
2. **Tính Năng 1 - Render Danh Sách Text Prompts**: Nhập danh sách văn bản (mỗi prompt 1 dòng). Chương trình sẽ tự động ngẫu nhiên hóa các hiệu ứng camera 3D (`zoom_in`, `zoom_out`, `pan_left`, `pan_right`, `3d_parallax`, `circle_orbit`), theo dõi tiến độ và tự động tải video MP4 về máy.
3. **Tính Năng 2 - Render Thư Mục Ảnh (Base64 Mode)**: Chọn một thư mục chứa các file ảnh (`.png`, `.jpg`, `.jpeg`, `.webp`), chuyển đổi ảnh thành chuỗi Base64 và gửi lên API render video hiệu ứng ngẫu nhiên.

