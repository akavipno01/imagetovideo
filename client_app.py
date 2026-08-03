#!/usr/bin/env python3
"""
Ứng dụng Client GUI (Tkinter) kết nối với API Image-to-Video Server.
Tính năng:
1. Nhập địa chỉ API Server (vd: https://xxx.trycloudflare.com).
2. Render danh sách Text Prompts (mỗi prompt 1 dòng) với hiệu ứng ngẫu nhiên.
3. Chọn thư mục chứa Ảnh -> Chuyển thành Base64 -> Render thành Video với hiệu ứng ngẫu nhiên.
4. Tải file Video MP4 về thư mục xuất (Output Directory).
"""

import base64
import os
import random
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

try:
    import requests
except ImportError:
    raise ImportError("Vui lòng cài đặt thư viện 'requests': pip install requests")

# Danh sách hiệu ứng chuyển động camera 3D
MOTION_EFFECTS = [
    "zoom_in",
    "zoom_out",
    "pan_left",
    "pan_right",
    "3d_parallax",
    "circle_orbit",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class ImageToVideoClientApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Image & Text To Video Batch Client v1.0")
        self.geometry("960 ...")
        self.geometry("980x720")
        self.minsize(850, 600)

        # Cấu hình Theme / Styles
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.configure_styles()

        # Biến trạng thái
        self.api_url_var = tk.StringVar(value="https://tobacco-went-harper-que.trycloudflare.com")
        self.output_dir_var = tk.StringVar(value=str(Path.home() / "Downloads" / "AI_Videos"))
        self.is_processing = False
        self.stop_requested = False

        self.create_widgets()

    def configure_styles(self):
        bg_dark = "#1e1e2e"
        fg_light = "#cdd6f4"
        card_bg = "#313244"
        accent_color = "#89b4fa"

        self.configure(bg=bg_dark)
        self.style.configure(".", background=bg_dark, foreground=fg_light, font=("Segoe UI", 10))
        self.style.configure("TFrame", background=bg_dark)
        self.style.configure("Card.TFrame", background=card_bg, relief="flat")
        self.style.configure("TLabel", background=bg_dark, foreground=fg_light)
        self.style.configure("Card.TLabel", background=card_bg, foreground=fg_light)
        self.style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground=accent_color)
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        self.style.configure("Accent.TButton", background=accent_color, foreground="#11111b")
        self.style.map("Accent.TButton", background=[("active", "#b4befe")])
        self.style.configure("TNotebook", background=bg_dark, borderwidth=0)
        self.style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[12, 6])
        self.style.map("TNotebook.Tab", background=[("selected", card_bg), ("!selected", bg_dark)],
                                        foreground=[("selected", accent_color), ("!selected", fg_light)])

    def create_widgets(self):
        # 1. TOP BAR: Cấu hình địa chỉ Server & Thư mục Lưu
        top_frame = ttk.Frame(self, style="Card.TFrame", padding=12)
        top_frame.pack(fill="x", padx=12, pady=10)

        # Hàng 1: API Server URL
        lbl_api = ttk.Label(top_frame, text="🌐 URL Server API:", style="Card.TLabel", font=("Segoe UI", 10, "bold"))
        lbl_api.grid(row=0, column=0, sticky="w", padx=5, pady=4)

        entry_api = ttk.Entry(top_frame, textvariable=self.api_url_var, font=("Segoe UI", 10), width=50)
        entry_api.grid(row=0, column=1, sticky="ew", padx=5, pady=4)

        btn_test = ttk.Button(top_frame, text="🔌 Kiểm Tra Kết Nối", command=self.test_connection)
        btn_test.grid(row=0, column=2, padx=5, pady=4)

        # Hàng 2: Thư mục kết quả (Output Dir)
        lbl_out = ttk.Label(top_frame, text="📁 Thư Mục Lưu Video:", style="Card.TLabel", font=("Segoe UI", 10, "bold"))
        lbl_out.grid(row=1, column=0, sticky="w", padx=5, pady=4)

        entry_out = ttk.Entry(top_frame, textvariable=self.output_dir_var, font=("Segoe UI", 10), width=50)
        entry_out.grid(row=1, column=1, sticky="ew", padx=5, pady=4)

        btn_browse_out = ttk.Button(top_frame, text="Chọn Thư Mục...", command=self.browse_output_dir)
        btn_browse_out.grid(row=1, column=2, padx=5, pady=4)

        top_frame.columnconfigure(1, weight=1)

        # 2. TAB CONTROL: Tính năng 1 (Text List) & Tính năng 2 (Image Folder)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=5)

        self.tab_text = ttk.Frame(self.notebook, padding=10)
        self.tab_image = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.tab_text, text="📝 Tính Năng 1: Text Prompt List -> Video")
        self.notebook.add(self.tab_image, text="🖼️ Tính Năng 2: Folder Ảnh Base64 -> Video")

        self.setup_tab_text()
        self.setup_tab_image()

        # 3. BOTTOM PANEL: Tiến độ & Log Hệ thống
        bottom_frame = ttk.Frame(self, style="Card.TFrame", padding=10)
        bottom_frame.pack(fill="x", padx=12, pady=10)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(bottom_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=4)

        self.lbl_status = ttk.Label(bottom_frame, text="Trạng thái: Sẵn sàng", style="Card.TLabel")
        self.lbl_status.pack(anchor="w", pady=2)

        # Log Text Box
        log_frame = ttk.Frame(bottom_frame)
        log_frame.pack(fill="both", expand=True, pady=4)

        self.log_text = tk.Text(
            log_frame, height=6, bg="#181825", fg="#a6adc8", font=("Consolas", 9), relief="flat", wrap="word"
        )
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    # ================= TAB 1: TEXT PROMPTS =================
    def setup_tab_text(self):
        lbl = ttk.Label(self.tab_text, text="Nhập danh sách Text Prompt (mỗi prompt nằm trên 1 dòng):", style="Header.TLabel")
        lbl.pack(anchor="w", pady=(0, 5))

        # Khung chứa Textbox & Tùy chọn
        mid_frame = ttk.Frame(self.tab_text)
        mid_frame.pack(fill="both", expand=True)

        self.txt_prompts = tk.Text(mid_frame, height=10, bg="#181825", fg="#cdd6f4", font=("Segoe UI", 10), insertbackground="white")
        txt_scroll = ttk.Scrollbar(mid_frame, command=self.txt_prompts.yview)
        self.txt_prompts.configure(yscrollcommand=txt_scroll.set)

        self.txt_prompts.pack(side="left", fill="both", expand=True)
        txt_scroll.pack(side="right", fill="y")

        # Ví dụ gợi ý
        sample_prompts = (
            "A serene Japanese garden in spring with cherry blossoms falling, 4k cinematic\n"
            "A futuristic cyberpunk street at midnight with neon lights reflections\n"
            "A majestic waterfall inside a tropical rainforest with rainbow light"
        )
        self.txt_prompts.insert("1.0", sample_prompts)

        # Các tùy chọn tham số
        opts_frame = ttk.Frame(self.tab_text, padding=5)
        opts_frame.pack(fill="x", pady=8)

        ttk.Label(opts_frame, text="Chiều rộng (width):").grid(row=0, column=0, padx=4)
        self.spn_txt_width = ttk.Spinbox(opts_frame, from_=256, to=2048, increment=64, width=6)
        self.spn_txt_width.set(1024)
        self.spn_txt_width.grid(row=0, column=1, padx=4)

        ttk.Label(opts_frame, text="Chiều cao (height):").grid(row=0, column=2, padx=4)
        self.spn_txt_height = ttk.Spinbox(opts_frame, from_=256, to=2048, increment=64, width=6)
        self.spn_txt_height.set(720)
        self.spn_txt_height.grid(row=0, column=3, padx=4)

        ttk.Label(opts_frame, text="Frames:").grid(row=0, column=4, padx=4)
        self.spn_txt_frames = ttk.Spinbox(opts_frame, from_=10, to=600, increment=15, width=6)
        self.spn_txt_frames.set(225)
        self.spn_txt_frames.grid(row=0, column=5, padx=4)

        ttk.Label(opts_frame, text="FPS:").grid(row=0, column=6, padx=4)
        self.spn_txt_fps = ttk.Spinbox(opts_frame, from_=5, to=60, increment=1, width=5)
        self.spn_txt_fps.set(15)
        self.spn_txt_fps.grid(row=0, column=7, padx=4)

        self.chk_txt_random = tk.BooleanVar(value=True)
        chk_rnd = ttk.Checkbutton(opts_frame, text="🎲 Random Hiệu Ứng Camera 3D", variable=self.chk_txt_random)
        chk_rnd.grid(row=0, column=8, padx=10)

        # Nút điều khiển
        btn_frame = ttk.Frame(self.tab_text)
        btn_frame.pack(fill="x", pady=5)

        self.btn_run_text = ttk.Button(btn_frame, text="🚀 ĐÀO THẢI / RENDER DANH SÁCH TEXT", style="Accent.TButton", command=self.start_text_batch)
        self.btn_run_text.pack(side="left", padx=5)

        self.btn_stop_text = ttk.Button(btn_frame, text="⏹️ Dừng", command=self.request_stop, state="disabled")
        self.btn_stop_text.pack(side="left", padx=5)

    # ================= TAB 2: IMAGE FOLDER =================
    def setup_tab_image(self):
        lbl = ttk.Label(self.tab_image, text="Chọn Thư Mục Chứa Ảnh để Render Video (Base64 Mode):", style="Header.TLabel")
        lbl.pack(anchor="w", pady=(0, 5))

        # Khung chọn folder
        folder_frame = ttk.Frame(self.tab_image)
        folder_frame.pack(fill="x", pady=5)

        self.img_folder_var = tk.StringVar()
        entry_img_dir = ttk.Entry(folder_frame, textvariable=self.img_folder_var, font=("Segoe UI", 10))
        entry_img_dir.pack(side="left", fill="x", expand=True, padx=(0, 5))

        btn_browse_img = ttk.Button(folder_frame, text="📁 Chọn Thư Mục Ảnh...", command=self.browse_image_folder)
        btn_browse_img.pack(side="right")

        # Danh sách ảnh tìm thấy
        lbl_files = ttk.Label(self.tab_image, text="Danh sách các file ảnh sẽ được xử lý:")
        lbl_files.pack(anchor="w", pady=(8, 2))

        list_frame = ttk.Frame(self.tab_image)
        list_frame.pack(fill="both", expand=True)

        self.lst_images = tk.Listbox(list_frame, bg="#181825", fg="#cdd6f4", font=("Consolas", 9), selectbackground="#45475a")
        lst_scroll = ttk.Scrollbar(list_frame, command=self.lst_images.yview)
        self.lst_images.configure(yscrollcommand=lst_scroll.set)

        self.lst_images.pack(side="left", fill="both", expand=True)
        lst_scroll.pack(side="right", fill="y")

        # Các tùy chọn tham số
        opts_frame = ttk.Frame(self.tab_image, padding=5)
        opts_frame.pack(fill="x", pady=8)

        ttk.Label(opts_frame, text="Frames:").grid(row=0, column=0, padx=4)
        self.spn_img_frames = ttk.Spinbox(opts_frame, from_=10, to=600, increment=15, width=6)
        self.spn_img_frames.set(225)
        self.spn_img_frames.grid(row=0, column=1, padx=4)

        ttk.Label(opts_frame, text="FPS:").grid(row=0, column=2, padx=4)
        self.spn_img_fps = ttk.Spinbox(opts_frame, from_=5, to=60, increment=1, width=5)
        self.spn_img_fps.set(15)
        self.spn_img_fps.grid(row=0, column=3, padx=4)

        self.chk_img_random = tk.BooleanVar(value=True)
        chk_rnd = ttk.Checkbutton(opts_frame, text="🎲 Random Hiệu Ứng Camera 3D", variable=self.chk_img_random)
        chk_rnd.grid(row=0, column=4, padx=10)

        # Nút điều khiển
        btn_frame = ttk.Frame(self.tab_image)
        btn_frame.pack(fill="x", pady=5)

        self.btn_run_img = ttk.Button(btn_frame, text="🚀 RENDER TẤT CẢ ÁNH TRONG FOLDER", style="Accent.TButton", command=self.start_image_batch)
        self.btn_run_img.pack(side="left", padx=5)

        self.btn_stop_img = ttk.Button(btn_frame, text="⏹️ Dừng", command=self.request_stop, state="disabled")
        self.btn_stop_img.pack(side="left", padx=5)

    # ================= LOG & HELPER UTILS =================
    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}\n"
        self.log_text.insert("end", formatted)
        self.log_text.see("end")

    def update_status(self, text: str):
        self.lbl_status.config(text=f"Trạng thái: {text}")

    def get_api_base(self) -> str:
        url = self.api_url_var.get().strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        return url

    def browse_output_dir(self):
        path = filedialog.askdirectory(title="Chọn Thư Mục Lưu Video Kết Quả")
        if path:
            self.output_dir_var.set(path)

    def browse_image_folder(self):
        path = filedialog.askdirectory(title="Chọn Thư Mục Chứa Ảnh")
        if path:
            self.img_folder_var.set(path)
            self.load_images_from_folder(path)

    def load_images_from_folder(self, folder_path: str):
        self.lst_images.delete(0, "end")
        p = Path(folder_path)
        if not p.is_dir():
            return

        files = [f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        files.sort()
        for f in files:
            self.lst_images.insert("end", f.name)

        self.log(f"Tìm thấy {len(files)} file ảnh hợp lệ trong thư mục: {folder_path}")

    def test_connection(self):
        base_url = self.get_api_base()
        self.log(f"Đang kiểm tra kết nối tới Server: {base_url} ...")

        def run_test():
            try:
                res = requests.get(f"{base_url}/health", timeout=8)
                if res.status_code == 200:
                    data = res.json()
                    self.after(0, lambda: messagebox.showinfo("Thành công", f"Kết nối Server thành công!\nTrạng thái: {data.get('status')}"))
                    self.after(0, lambda: self.log(f"✅ Kết nối Server thành công: {data}"))
                else:
                    self.after(0, lambda: messagebox.showwarning("Lỗi", f"Server trả về mã HTTP {res.status_code}"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Lỗi Kết Nối", f"Không thể kết nối Server:\n{str(e)}"))
                self.after(0, lambda: self.log(f"❌ Lỗi kết nối: {str(e)}"))

        threading.Thread(target=run_test, daemon=True).start()

    def set_buttons_state(self, running: bool):
        state_run = "disabled" if running else "normal"
        state_stop = "normal" if running else "disabled"
        self.btn_run_text.config(state=state_run)
        self.btn_run_img.config(state=state_run)
        self.btn_stop_text.config(state=state_stop)
        self.btn_stop_img.config(state=state_stop)

    def request_stop(self):
        if self.is_processing:
            self.stop_requested = True
            self.log("⚠️ Đã nhận yêu cầu DỪNG tiến trình sau khi tác vụ hiện tại hoàn tất...")

    # ================= MAIN RENDER PIPELINE =================
    def wait_and_download_task(self, base_url: str, task_id: str, save_filename: str) -> bool:
        """Kiểm tra tiến độ tác vụ từ Server và tải file video khi completed."""
        status_url = f"{base_url}/status/{task_id}"
        download_url = f"{base_url}/download/{task_id}"
        out_dir = Path(self.output_dir_var.get().strip())
        out_dir.mkdir(parents=True, exist_ok=True)
        dest_path = out_dir / save_filename

        start_time = time.time()

        while not self.stop_requested:
            try:
                res = requests.get(status_url, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    status = data.get("status")
                    progress = data.get("progress", 0.0)
                    detail = data.get("detail", "")

                    self.after(0, lambda p=progress: self.progress_var.set(p))
                    self.after(0, lambda d=detail: self.update_status(d))

                    if status == "completed":
                        self.log(f"✅ Tác vụ {task_id[:8]} hoàn tất! Bắt đầu tải video...")
                        dl_res = requests.get(download_url, timeout=60, stream=True)
                        if dl_res.status_code == 200:
                            with open(dest_path, "wb") as f:
                                for chunk in dl_res.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            self.log(f"💾 Đã lưu Video: {dest_path}")
                            return True
                        else:
                            self.log(f"❌ Không thể tải video: HTTP {dl_res.status_code}")
                            return False
                    elif status == "failed":
                        self.log(f"❌ Tác vụ {task_id[:8]} thất bại: {data.get('error')}")
                        return False

                time.sleep(2.0)
            except Exception as e:
                self.log(f"⚠️ Thử lại kiểm tra trạng thái ({task_id[:8]}): {e}")
                time.sleep(3.0)

        self.log(f"⏹️ Tác vụ {task_id[:8]} bị dừng bởi người dùng.")
        return False

    # --- TÍNH NĂNG 1: RENDER TEXT LIST ---
    def start_text_batch(self):
        raw_text = self.txt_prompts.get("1.0", "end").strip()
        prompts = [line.strip() for line in raw_text.splitlines() if line.strip()]

        if not prompts:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập ít nhất 1 text prompt!")
            return

        self.is_processing = True
        self.stop_requested = False
        self.set_buttons_state(True)
        self.log(f"🚀 Khởi chạy Render Batch danh sách {len(prompts)} Text Prompts...")

        base_url = self.get_api_base()
        width = int(self.spn_txt_width.get())
        height = int(self.spn_txt_height.get())
        num_frames = int(self.spn_txt_frames.get())
        fps = int(self.spn_txt_fps.get())
        use_random = self.chk_txt_random.get()

        def worker():
            total = len(prompts)
            for idx, prompt in enumerate(prompts, start=1):
                if self.stop_requested:
                    break

                effect = random.choice(MOTION_EFFECTS) if use_random else "zoom_in"
                self.log(f"\n--- [{idx}/{total}] Prompt: '{prompt[:40]}...' (Hiệu ứng: {effect}) ---")

                payload = {
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "num_frames": num_frames,
                    "fps": fps,
                    "motion_type": effect,
                }

                try:
                    res = requests.post(f"{base_url}/generate", json=payload, timeout=15)
                    if res.status_code == 202:
                        task_id = res.json().get("task_id")
                        self.log(f"📩 Tác vụ đã được gửi, ID: {task_id}")
                        safe_title = "".join(c if c.isalnum() else "_" for c in prompt[:25]).strip("_")
                        save_filename = f"text_{idx:03d}_{safe_title}_{effect}.mp4"

                        self.wait_and_download_task(base_url, task_id, save_filename)
                    else:
                        self.log(f"❌ Lỗi gửi request ({res.status_code}): {res.text}")
                except Exception as e:
                    self.log(f"❌ Lỗi kết nối gửi task: {e}")

            self.after(0, lambda: self.log("\n🎉 HOÀN TẤT RENDERING TẤT CẢ PROMPTS!"))
            self.after(0, lambda: self.update_status("Hoàn tất!"))
            self.after(0, lambda: self.progress_var.set(100.0))
            self.after(0, lambda: self.set_buttons_state(False))
            self.is_processing = False

        threading.Thread(target=worker, daemon=True).start()

    # --- TÍNH NĂNG 2: RENDER FOLDER ÁNH BASE64 ---
    def start_image_batch(self):
        folder_path = self.img_folder_var.get().strip()
        p = Path(folder_path)
        if not p.is_dir():
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn thư mục chứa ảnh hợp lệ!")
            return

        files = [f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        files.sort()

        if not files:
            messagebox.showwarning("Cảnh báo", "Không tìm thấy file ảnh hợp lệ nào trong thư mục này!")
            return

        self.is_processing = True
        self.stop_requested = False
        self.set_buttons_state(True)
        self.log(f"🚀 Khởi chạy Render Batch {len(files)} file ảnh từ thư mục...")

        base_url = self.get_api_base()
        num_frames = int(self.spn_img_frames.get())
        fps = int(self.spn_img_fps.get())
        use_random = self.chk_img_random.get()

        def worker():
            total = len(files)
            for idx, img_file in enumerate(files, start=1):
                if self.stop_requested:
                    break

                effect = random.choice(MOTION_EFFECTS) if use_random else "3d_parallax"
                self.log(f"\n--- [{idx}/{total}] Đang xử lý file ảnh: '{img_file.name}' (Hiệu ứng: {effect}) ---")

                try:
                    # Chuyển ảnh thành chuỗi Base64
                    with open(img_file, "rb") as f:
                        img_bytes = f.read()
                    b64_str = base64.b64encode(img_bytes).decode("utf-8")

                    payload = {
                        "image_base64": b64_str,
                        "motion_type": effect,
                        "num_frames": num_frames,
                        "fps": fps,
                        "prompt": f"Render Image: {img_file.stem}",
                    }

                    res = requests.post(f"{base_url}/generate-from-image", json=payload, timeout=30)
                    if res.status_code == 202:
                        task_id = res.json().get("task_id")
                        self.log(f"📩 Tác vụ đã được gửi thành công, ID: {task_id}")
                        save_filename = f"img_{idx:03d}_{img_file.stem}_{effect}.mp4"

                        self.wait_and_download_task(base_url, task_id, save_filename)
                    else:
                        self.log(f"❌ Lỗi gửi API ({res.status_code}): {res.text}")
                except Exception as e:
                    self.log(f"❌ Lỗi đọc ảnh hoặc gửi request: {e}")

            self.after(0, lambda: self.log("\n🎉 HOÀN TẤT RENDERING TẤT CẢ ÁNH TRONG FOLDER!"))
            self.after(0, lambda: self.update_status("Hoàn tất!"))
            self.after(0, lambda: self.progress_var.set(100.0))
            self.after(0, lambda: self.set_buttons_state(False))
            self.is_processing = False

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    app = ImageToVideoClientApp()
    app.mainloop()
