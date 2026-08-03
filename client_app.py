#!/usr/bin/env python3
"""
===============================================================================
🎬 AI IMAGE & TEXT TO VIDEO BATCH STUDIO v4.1
===============================================================================
Ứng dụng Client GUI chuyên nghiệp kết nối API Image-to-Video Server.

Tính năng mới v4.1:
- Tab 1 ("Text to video"): Loại bỏ tùy chọn luồng (chạy tuần tự chuẩn xác), giữ nguyên Checkbox "🖼️ Tải Kèm Ảnh AI Gốc (.png)".
- Tab 2 ("Image to video"): Đặt MẶC ĐỊNH 4 LUỒNG song song (Tối đa 4 luồng).
- Mặc định: Frame = 360, FPS = 20 (Độ phân giải 1080 x 720 HD, Random Hiệu Ứng 3D).
- Bảng Monitor bên phải (Right Panel) hiển thị Real-time Log các lượt gọi GET /status/{task_id}.
- Mặc định mở ở chế độ Full Screen maximized.
===============================================================================
"""

import base64
from concurrent.futures import ThreadPoolExecutor
import os
import random
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

# Tối ưu hóa độ phân giải cao High-DPI trên Windows
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

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

MOTION_LABELS = {
    "random": "🎲 Random Hiệu Ứng (Ngẫu Nhiên)",
    "zoom_in": "🔍 Zoom In (Phóng To Camera)",
    "zoom_out": "🔎 Zoom Out (Thu Nhỏ Camera)",
    "pan_left": "⬅️ Pan Left (Lia Trái)",
    "pan_right": "➡️ Pan Right (Lia Phải)",
    "3d_parallax": "✨ 3D Parallax (Nổi Khối Chiều Sâu)",
    "circle_orbit": "🔄 Circle Orbit (Xoay Vòng Tròn 3D)",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Danh sách độ phân giải sẵn (Resolution Presets)
RESOLUTION_PRESETS = {
    "1080 x 720 (HD - Mặc định)": (1080, 720),
    "1920 x 1080 (Full HD 1080p)": (1920, 1080),
    "1080 x 1080 (Square 1:1)": (1080, 1080),
    "720 x 1280 (Vertical TikTok/Reels)": (720, 1280),
    "512 x 512 (Standard SD)": (512, 512),
    "Tùy chỉnh (Custom)": (1080, 720),
}


class ProfessionalVideoStudioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Image & Text To Video Batch Studio v4.1 (Full Screen)")
        
        # Đặt kích thước cơ sở & Phóng to Toàn màn hình (Full Screen Maximized)
        self.geometry("1360x860")
        self.minsize(1100, 700)
        try:
            self.state("zoomed")  # Mặc định Full Screen trên Windows
        except Exception:
            pass

        # Cấu hình Bảng màu Dark Studio (Catppuccin / Modern Slate Palette)
        self.colors = {
            "bg_dark": "#11111b",       # Nền chính tối
            "card_bg": "#1e1e2e",       # Khung Card
            "card_border": "#313244",   # Viền khung
            "input_bg": "#181825",     # Nền ô nhập dữ liệu
            "text_main": "#ffffff",    # Chữ trắng sáng (Tương phản cao)
            "text_sub": "#a6adc8",     # Chữ phụ
            "accent": "#89b4fa",       # Xanh lam Accent
            "accent_hover": "#b4befe", # Hover Xanh lam
            "success": "#a6e3a1",      # Xanh lá thành công
            "danger": "#f38ba8",       # Đỏ dừng/lỗi
            "warning": "#fab387",      # Cam cảnh báo
            "status_log_bg": "#11111b",# Nền bảng status log riêng
            "status_log_fg": "#a6e3a1",# Chữ status log màu xanh neon
        }

        self.configure_styles()

        # Biến khóa luồng an toàn (Thread Lock)
        self.counter_lock = threading.Lock()

        # Các biến quản lý dữ liệu giao diện
        self.api_url_var = tk.StringVar(value="https://tobacco-went-harper-que.trycloudflare.com")
        self.output_dir_var = tk.StringVar(value=str(Path.home() / "Downloads" / "AI_Videos"))
        self.connection_status_var = tk.StringVar(value="⚪ Chưa kết nối")
        self.chk_txt_save_img_var = tk.BooleanVar(value=True)  # Mặc định TẢI KÈM ÁNH AI GỐC (.png)

        # Thống kê tiến trình & Task active
        self.active_task_id_var = tk.StringVar(value="Chưa có tác vụ")
        self.active_task_status_var = tk.StringVar(value="IDLE")
        self.active_task_progress_var = tk.StringVar(value="0.0 %")

        self.total_tasks = 0
        self.success_count = 0
        self.failed_count = 0

        self.is_processing = False
        self.stop_requested = False

        self.create_widgets()

    def configure_styles(self):
        self.configure(bg=self.colors["bg_dark"])
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        # Cấu hình danh sách popup Combobox rõ chữ, tương phản cao
        self.option_add("*TCombobox*Listbox.font", ("Segoe UI", 10, "bold"))
        self.option_add("*TCombobox*Listbox.background", "#181825")
        self.option_add("*TCombobox*Listbox.foreground", "#ffffff")
        self.option_add("*TCombobox*Listbox.selectBackground", "#89b4fa")
        self.option_add("*TCombobox*Listbox.selectForeground", "#11111b")

        self.style.configure(".", background=self.colors["bg_dark"], foreground=self.colors["text_main"], font=("Segoe UI", 10))
        self.style.configure("TFrame", background=self.colors["bg_dark"])
        self.style.configure("Card.TFrame", background=self.colors["card_bg"], relief="flat", borderwidth=1)
        self.style.configure("Header.TFrame", background=self.colors["card_bg"])

        self.style.configure("TLabel", background=self.colors["bg_dark"], foreground=self.colors["text_main"])
        self.style.configure("Card.TLabel", background=self.colors["card_bg"], foreground=self.colors["text_main"])
        self.style.configure("Sub.TLabel", background=self.colors["card_bg"], foreground=self.colors["text_sub"], font=("Segoe UI", 9))
        self.style.configure("Title.TLabel", background=self.colors["card_bg"], font=("Segoe UI", 14, "bold"), foreground=self.colors["accent"])
        self.style.configure("Section.TLabel", background=self.colors["card_bg"], font=("Segoe UI", 11, "bold"), foreground=self.colors["accent"])

        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6, relief="flat", background=self.colors["card_border"], foreground=self.colors["text_main"])
        self.style.map("TButton", background=[("active", self.colors["accent"]), ("disabled", "#2a2b3d")], foreground=[("active", "#11111b")])

        self.style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=9, background=self.colors["accent"], foreground="#11111b")
        self.style.map("Accent.TButton", background=[("active", self.colors["accent_hover"]), ("disabled", "#313244")], foreground=[("disabled", "#6c7086")])

        self.style.configure("Danger.TButton", font=("Segoe UI", 10, "bold"), padding=9, background=self.colors["danger"], foreground="#11111b")
        self.style.map("Danger.TButton", background=[("active", "#f5e0dc")])

        self.style.configure("TNotebook", background=self.colors["bg_dark"], borderwidth=0)
        self.style.configure("TNotebook.Tab", font=("Segoe UI", 11, "bold"), padding=[20, 8], background=self.colors["bg_dark"], foreground=self.colors["text_sub"])
        self.style.map("TNotebook.Tab", background=[("selected", self.colors["card_bg"])], foreground=[("selected", self.colors["accent"])])

        self.style.configure("TCheckbutton", background=self.colors["card_bg"], foreground=self.colors["text_main"], font=("Segoe UI", 10, "bold"))
        self.style.map("TCheckbutton", background=[("active", self.colors["card_bg"])], foreground=[("active", self.colors["accent"])])

        self.style.configure("TEntry", fieldbackground=self.colors["input_bg"], foreground="#ffffff", font=("Segoe UI", 10, "bold"), borderwidth=1)

        # Style Combobox TƯƠNG PHẢN CAO, BẬT RÕ CHỮ
        self.style.configure(
            "TCombobox",
            fieldbackground="#181825",
            background="#313244",
            foreground="#ffffff",
            selectbackground="#181825",
            selectforeground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            arrowcolor="#89b4fa",
            padding=4,
        )
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#181825"), ("active", "#313244"), ("disabled", "#2a2b3d")],
            foreground=[("readonly", "#ffffff"), ("active", "#ffffff"), ("disabled", "#6c7086")],
            selectbackground=[("readonly", "#181825"), ("focus", "#181825")],
            selectforeground=[("readonly", "#ffffff"), ("focus", "#ffffff")],
        )

        # Style Spinbox RÕ CHỮ
        self.style.configure(
            "TSpinbox",
            fieldbackground="#181825",
            background="#313244",
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            arrowcolor="#89b4fa",
            padding=4,
        )
        self.style.map(
            "TSpinbox",
            fieldbackground=[("active", "#313244")],
            foreground=[("active", "#ffffff")],
        )

    def create_widgets(self):
        # Header Logo & Subtitle
        header_panel = ttk.Frame(self, style="Card.TFrame", padding=(16, 12))
        header_panel.pack(fill="x", padx=14, pady=(12, 6))

        lbl_logo = ttk.Label(header_panel, text="🎬 AI IMAGE & TEXT TO VIDEO STUDIO v4.1", style="Title.TLabel")
        lbl_logo.pack(anchor="w")

        lbl_desc = ttk.Label(
            header_panel,
            text="Hệ thống Render Batch Video 3D | Tab Image to Video Mặc Định 4 Luồng Song Song & Tải Kèm Ảnh AI Gốc",
            style="Sub.TLabel",
        )
        lbl_desc.pack(anchor="w", pady=(2, 0))

        # 1. TOP SERVER CONTROL CARD
        server_card = ttk.Frame(self, style="Card.TFrame", padding=14)
        server_card.pack(fill="x", padx=14, pady=6)

        # Hàng 1: URL Server & Status
        lbl_api = ttk.Label(server_card, text="🌐 URL Server API:", style="Card.TLabel", font=("Segoe UI", 10, "bold"))
        lbl_api.grid(row=0, column=0, sticky="w", padx=4, pady=4)

        entry_api = ttk.Entry(server_card, textvariable=self.api_url_var, font=("Segoe UI", 10, "bold"), width=54)
        entry_api.grid(row=0, column=1, sticky="ew", padx=6, pady=4)

        btn_test = ttk.Button(server_card, text="🔌 Kiểm Tra Kết Nối", command=self.test_connection)
        btn_test.grid(row=0, column=2, padx=4, pady=4)

        self.lbl_connection_badge = ttk.Label(
            server_card, textvariable=self.connection_status_var, style="Card.TLabel", font=("Segoe UI", 10, "bold")
        )
        self.lbl_connection_badge.grid(row=0, column=3, padx=12, pady=4, sticky="e")

        # Hàng 2: Thư mục đầu ra (Output Directory)
        lbl_out = ttk.Label(server_card, text="📁 Thư Mục Lưu Video:", style="Card.TLabel", font=("Segoe UI", 10, "bold"))
        lbl_out.grid(row=1, column=0, sticky="w", padx=4, pady=4)

        entry_out = ttk.Entry(server_card, textvariable=self.output_dir_var, font=("Segoe UI", 10, "bold"), width=54)
        entry_out.grid(row=1, column=1, sticky="ew", padx=6, pady=4)

        btn_browse_out = ttk.Button(server_card, text="Chọn Thư Mục...", command=self.browse_output_dir)
        btn_browse_out.grid(row=1, column=2, padx=4, pady=4)

        server_card.columnconfigure(1, weight=1)

        # 2. KHU VỰC CHÍNH SPLIT: BÊN TRÁI (TABS & TIẾN TRÌNH) + BÊN PHẢI (LOG STATUS MONITOR REALTIME)
        main_split_frame = ttk.Frame(self)
        main_split_frame.pack(fill="both", expand=True, padx=14, pady=6)

        # --- BÊN TRÁI (LEFT PANEL - 65% WIDTH) ---
        left_panel = ttk.Frame(main_split_frame)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self.notebook = ttk.Notebook(left_panel)
        self.notebook.pack(fill="both", expand=True, pady=(0, 6))

        self.tab_text = ttk.Frame(self.notebook, padding=12)
        self.tab_image = ttk.Frame(self.notebook, padding=12)

        # Đặt tên 2 Tab theo đúng yêu cầu người dùng
        self.notebook.add(self.tab_text, text="📝 Text to video")
        self.notebook.add(self.tab_image, text="🖼️ Image to video")

        self.setup_tab_text()
        self.setup_tab_image()

        # Khung Thanh Tiến Trình & Trạng Thái ở phía dưới bên trái
        bottom_left_card = ttk.Frame(left_panel, style="Card.TFrame", padding=14)
        bottom_left_card.pack(fill="x", pady=(6, 0))

        prog_top_frame = ttk.Frame(bottom_left_card, style="Card.TFrame")
        prog_top_frame.pack(fill="x", pady=(0, 4))

        self.lbl_status = ttk.Label(prog_top_frame, text="Trạng thái: Sẵn sàng làm việc", style="Card.TLabel", font=("Segoe UI", 10, "bold"))
        self.lbl_status.pack(side="left")

        self.lbl_stats = ttk.Label(prog_top_frame, text="Hoàn tất: 0/0", style="Sub.TLabel", font=("Segoe UI", 9, "bold"))
        self.lbl_stats.pack(side="right")

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(bottom_left_card, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=6)

        # --- BÊN PHẢI (RIGHT PANEL - DEDICATED STATUS MONITOR LOG 35% WIDTH) ---
        right_panel = ttk.Frame(main_split_frame, style="Card.TFrame", padding=14, width=440)
        right_panel.pack(side="right", fill="both", expand=False, padx=(6, 0))

        # Tiêu đề Bảng Status Monitor
        lbl_status_title = ttk.Label(right_panel, text="📡 BẢNG MONITOR STATUS REAL-TIME", style="Section.TLabel")
        lbl_status_title.pack(anchor="w", pady=(0, 4))

        lbl_status_sub = ttk.Label(right_panel, text="Theo dõi liên tục lượt gọi GET /status/{task_id} từng giây", style="Sub.TLabel")
        lbl_status_sub.pack(anchor="w", pady=(0, 8))

        # Card hiển thị thông tin Task Active hiện tại
        active_card = ttk.Frame(right_panel, style="Card.TFrame", padding=10, relief="solid")
        active_card.pack(fill="x", pady=(0, 10))

        ttk.Label(active_card, text="Task ID Active:", style="Card.TLabel", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")
        lbl_act_id = ttk.Label(active_card, textvariable=self.active_task_id_var, style="Card.TLabel", font=("Segoe UI", 9, "bold"), foreground=self.colors["accent"])
        lbl_act_id.grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(active_card, text="Trạng thái API:", style="Card.TLabel", font=("Segoe UI", 9, "bold")).grid(row=1, column=0, sticky="w")
        lbl_act_st = ttk.Label(active_card, textvariable=self.active_task_status_var, style="Card.TLabel", font=("Segoe UI", 9, "bold"), foreground=self.colors["warning"])
        lbl_act_st.grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(active_card, text="Tiến độ:", style="Card.TLabel", font=("Segoe UI", 9, "bold")).grid(row=2, column=0, sticky="w")
        lbl_act_pr = ttk.Label(active_card, textvariable=self.active_task_progress_var, style="Card.TLabel", font=("Segoe UI", 9, "bold"), foreground=self.colors["success"])
        lbl_act_pr.grid(row=2, column=1, sticky="w", padx=6)

        # Header Khung Log Status
        status_log_header = ttk.Frame(right_panel, style="Card.TFrame")
        status_log_header.pack(fill="x", pady=(2, 4))
        ttk.Label(status_log_header, text="🌐 Lịch Sử Polling /status Log:", style="Card.TLabel", font=("Segoe UI", 9, "bold")).pack(side="left")
        btn_clear_st = ttk.Button(status_log_header, text="Xóa Status Log", command=self.clear_status_logs)
        btn_clear_st.pack(side="right")

        # Text Box chuyên dụng hiển thị Log Status Call
        st_log_frame = ttk.Frame(right_panel)
        st_log_frame.pack(fill="both", expand=True)

        self.status_log_text = tk.Text(
            st_log_frame,
            height=26,
            width=42,
            bg=self.colors["status_log_bg"],
            fg=self.colors["status_log_fg"],
            font=("Consolas", 9, "bold"),
            relief="flat",
            wrap="word",
        )
        scrollbar_st = ttk.Scrollbar(st_log_frame, command=self.status_log_text.yview)
        self.status_log_text.configure(yscrollcommand=scrollbar_st.set)
        self.status_log_text.pack(side="left", fill="both", expand=True)
        scrollbar_st.pack(side="right", fill="y")

    # ================= TAB 1: TEXT TO VIDEO SETUP =================
    def setup_tab_text(self):
        lbl = ttk.Label(self.tab_text, text="Nhập danh sách Text Prompts (Mỗi prompt 1 dòng):", style="Section.TLabel")
        lbl.pack(anchor="w", pady=(0, 6))

        txt_frame = ttk.Frame(self.tab_text)
        txt_frame.pack(fill="both", expand=True, pady=(0, 8))

        self.txt_prompts = tk.Text(
            txt_frame, height=7, bg=self.colors["input_bg"], fg=self.colors["text_main"], font=("Segoe UI", 10), insertbackground="white", relief="flat"
        )
        txt_scroll = ttk.Scrollbar(txt_frame, command=self.txt_prompts.yview)
        self.txt_prompts.configure(yscrollcommand=txt_scroll.set)

        self.txt_prompts.pack(side="left", fill="both", expand=True)
        txt_scroll.pack(side="right", fill="y")

        sample_prompts = (
            "A serene cyberpunk neon city at sunset with rain reflections, 4k hyperrealistic\n"
            "A majestic glowing phoenix bird flying above mist covered mountains at dawn\n"
            "A cozy wooden cabin inside a snowy pine forest under aurora borealis"
        )
        self.txt_prompts.insert("1.0", sample_prompts)

        # CẤU HÌNH THAM SỐ TAB 1 (FRAME=360, FPS=20, ĐÃ BỎ PHẦN SỐ LUỒNG, CÓ CHECKBOX TẢI ÁNH)
        opts_card = ttk.Frame(self.tab_text, style="Card.TFrame", padding=10)
        opts_card.pack(fill="x", pady=6)

        # Hàng 1: Khung Ảnh, Width, Height
        ttk.Label(opts_card, text="📐 Mẫu Khung Ảnh:", style="Card.TLabel", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=4, pady=4)

        self.preset_var = tk.StringVar(value="1080 x 720 (HD - Mặc định)")
        cmb_preset = ttk.Combobox(opts_card, textvariable=self.preset_var, values=list(RESOLUTION_PRESETS.keys()), state="readonly", width=26)
        cmb_preset.grid(row=0, column=1, padx=4, pady=4)
        cmb_preset.bind("<<ComboboxSelected>>", self.on_resolution_preset_changed)

        ttk.Label(opts_card, text="Width:", style="Card.TLabel", font=("Segoe UI", 10, "bold")).grid(row=0, column=2, padx=4, pady=4)
        self.spn_txt_width = ttk.Spinbox(opts_card, from_=256, to=2048, increment=64, width=6)
        self.spn_txt_width.set(1080)
        self.spn_txt_width.grid(row=0, column=3, padx=4, pady=4)

        ttk.Label(opts_card, text="Height:", style="Card.TLabel", font=("Segoe UI", 10, "bold")).grid(row=0, column=4, padx=4, pady=4)
        self.spn_txt_height = ttk.Spinbox(opts_card, from_=256, to=2048, increment=64, width=6)
        self.spn_txt_height.set(720)
        self.spn_txt_height.grid(row=0, column=5, padx=4, pady=4)

        # Hàng 2: Hiệu ứng 3D, Frames, FPS
        ttk.Label(opts_card, text="🎬 Hiệu Ứng 3D:", style="Card.TLabel", font=("Segoe UI", 10, "bold")).grid(row=1, column=0, sticky="w", padx=4, pady=4)

        self.txt_motion_var = tk.StringVar(value="random")
        cmb_txt_motion = ttk.Combobox(
            opts_card,
            textvariable=self.txt_motion_var,
            values=[v for k, v in MOTION_LABELS.items()],
            state="readonly",
            width=26,
        )
        cmb_txt_motion.grid(row=1, column=1, padx=4, pady=4)
        cmb_txt_motion.set(MOTION_LABELS["random"])

        ttk.Label(opts_card, text="Frames:", style="Card.TLabel", font=("Segoe UI", 10, "bold")).grid(row=1, column=2, padx=4, pady=4)
        self.spn_txt_frames = ttk.Spinbox(opts_card, from_=10, to=600, increment=15, width=5)
        self.spn_txt_frames.set(360)  # MẶC ĐỊNH 360 FRAMES
        self.spn_txt_frames.grid(row=1, column=3, padx=4, pady=4)

        ttk.Label(opts_card, text="FPS:", style="Card.TLabel", font=("Segoe UI", 10, "bold")).grid(row=1, column=4, padx=4, pady=4)
        self.spn_txt_fps = ttk.Spinbox(opts_card, from_=5, to=60, increment=1, width=5)
        self.spn_txt_fps.set(20)  # MẶC ĐỊNH 20 FPS
        self.spn_txt_fps.grid(row=1, column=5, padx=4, pady=4)

        # Hàng 3: CHECKBOX TẢI KÈM ÁNH AI GỐC (.PNG) (Đã bỏ tùy chọn số luồng ở Tab 1)
        chk_save_img = ttk.Checkbutton(opts_card, text="🖼️ Tải Kèm Ảnh AI Gốc (.png)", variable=self.chk_txt_save_img_var, style="TCheckbutton")
        chk_save_img.grid(row=2, column=0, columnspan=6, sticky="w", padx=4, pady=4)

        btn_action_frame = ttk.Frame(self.tab_text)
        btn_action_frame.pack(fill="x", pady=8)

        self.btn_run_text = ttk.Button(
            btn_action_frame, text="🚀 KHỞI CHẠY RENDER DANH SÁCH TEXT PROMPTS", style="Accent.TButton", command=self.start_text_batch
        )
        self.btn_run_text.pack(side="left", padx=(0, 8))

        self.btn_stop_text = ttk.Button(btn_action_frame, text="⏹️ DỪNG TIẾN TRÌNH", style="Danger.TButton", command=self.request_stop, state="disabled")
        self.btn_stop_text.pack(side="left")

    def on_resolution_preset_changed(self, event):
        preset_name = self.preset_var.get()
        if preset_name in RESOLUTION_PRESETS:
            w, h = RESOLUTION_PRESETS[preset_name]
            self.spn_txt_width.set(w)
            self.spn_txt_height.set(h)

    # ================= TAB 2: IMAGE TO VIDEO SETUP =================
    def setup_tab_image(self):
        lbl = ttk.Label(self.tab_image, text="Chọn Thư Mục Chứa Ảnh để Render Video (Base64 Mode):", style="Section.TLabel")
        lbl.pack(anchor="w", pady=(0, 6))

        folder_card = ttk.Frame(self.tab_image, style="Card.TFrame", padding=10)
        folder_card.pack(fill="x", pady=(0, 8))

        self.img_folder_var = tk.StringVar()
        entry_img_dir = ttk.Entry(folder_card, textvariable=self.img_folder_var, font=("Segoe UI", 10, "bold"))
        entry_img_dir.pack(side="left", fill="x", expand=True, padx=(0, 8))

        btn_browse_img = ttk.Button(folder_card, text="📁 Chọn Thư Mục Ảnh...", command=self.browse_image_folder)
        btn_browse_img.pack(side="right")

        lbl_files = ttk.Label(self.tab_image, text="Danh sách các file ảnh hợp lệ:")
        lbl_files.pack(anchor="w", pady=(4, 2))

        list_frame = ttk.Frame(self.tab_image)
        list_frame.pack(fill="both", expand=True, pady=(0, 8))

        self.lst_images = tk.Listbox(
            list_frame, bg=self.colors["input_bg"], fg=self.colors["text_main"], font=("Consolas", 9), relief="flat", selectbackground="#45475a"
        )
        lst_scroll = ttk.Scrollbar(list_frame, command=self.lst_images.yview)
        self.lst_images.configure(yscrollcommand=lst_scroll.set)

        self.lst_images.pack(side="left", fill="both", expand=True)
        lst_scroll.pack(side="right", fill="y")

        opts_card = ttk.Frame(self.tab_image, style="Card.TFrame", padding=10)
        opts_card.pack(fill="x", pady=6)

        # Hàng 1: Hiệu ứng 3D, Frames, FPS
        ttk.Label(opts_card, text="🎬 Hiệu Ứng 3D:", style="Card.TLabel", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=4, pady=4)

        self.img_motion_var = tk.StringVar(value="random")
        cmb_img_motion = ttk.Combobox(
            opts_card,
            textvariable=self.img_motion_var,
            values=[v for k, v in MOTION_LABELS.items()],
            state="readonly",
            width=26,
        )
        cmb_img_motion.grid(row=0, column=1, padx=4, pady=4)
        cmb_img_motion.set(MOTION_LABELS["random"])

        ttk.Label(opts_card, text="Frames:", style="Card.TLabel", font=("Segoe UI", 10, "bold")).grid(row=0, column=2, padx=4, pady=4)
        self.spn_img_frames = ttk.Spinbox(opts_card, from_=10, to=600, increment=15, width=5)
        self.spn_img_frames.set(360)  # MẶC ĐỊNH 360 FRAMES
        self.spn_img_frames.grid(row=0, column=3, padx=4, pady=4)

        ttk.Label(opts_card, text="FPS:", style="Card.TLabel", font=("Segoe UI", 10, "bold")).grid(row=0, column=4, padx=4, pady=4)
        self.spn_img_fps = ttk.Spinbox(opts_card, from_=5, to=60, increment=1, width=5)
        self.spn_img_fps.set(20)  # MẶC ĐỊNH 20 FPS
        self.spn_img_fps.grid(row=0, column=5, padx=4, pady=4)

        # Hàng 2: SỐ LUỒNG (WORKERS) MẶC ĐỊNH 4 LUỒNG CHO TAB IMAGE TO VIDEO
        ttk.Label(opts_card, text="⚡ Số Luồng (Workers):", style="Card.TLabel", font=("Segoe UI", 10, "bold")).grid(row=1, column=0, sticky="w", padx=4, pady=4)
        self.spn_img_threads = ttk.Spinbox(opts_card, from_=1, to=4, increment=1, width=5)
        self.spn_img_threads.set(4)  # MẶC ĐỊNH 4 LUỒNG CHO IMAGE TO VIDEO
        self.spn_img_threads.grid(row=1, column=1, sticky="w", padx=4, pady=4)

        btn_action_frame = ttk.Frame(self.tab_image)
        btn_action_frame.pack(fill="x", pady=8)

        self.btn_run_img = ttk.Button(
            btn_action_frame, text="🚀 KHỞI CHẠY RENDER TẤT CẢ ÁNH TRONG THƯ MỤC", style="Accent.TButton", command=self.start_image_batch
        )
        self.btn_run_img.pack(side="left", padx=(0, 8))

        self.btn_stop_img = ttk.Button(btn_action_frame, text="⏹️ DỪNG TIẾN TRÌNH", style="Danger.TButton", command=self.request_stop, state="disabled")
        self.btn_stop_img.pack(side="left")

    # ================= LOG & STATUS MONITOR UTILS =================
    def status_log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}\n"
        self.status_log_text.insert("end", formatted)
        self.status_log_text.see("end")

    def log(self, message: str):
        """Tập trung log vào bảng Monitor Status bên phải."""
        self.status_log(message)

    def clear_status_logs(self):
        self.status_log_text.delete("1.0", "end")

    def update_status(self, text: str):
        self.lbl_status.config(text=f"Trạng thái: {text}")

    def update_stats_label(self):
        self.lbl_stats.config(text=f"Hoàn tất: {self.success_count}/{self.total_tasks} | Lỗi: {self.failed_count}")

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

        self.status_log(f"🔍 Đã tìm thấy {len(files)} file ảnh hợp lệ trong thư mục: {folder_path}")

    def test_connection(self):
        base_url = self.get_api_base()
        self.status_log(f"🔌 Đang kiểm tra kết nối tới Server: {base_url} ...")
        self.status_log(f"🌐 CALL: GET {base_url}/health")
        self.connection_status_var.set("🟡 Đang kết nối...")

        def run_test():
            try:
                res = requests.get(f"{base_url}/health", timeout=8)
                if res.status_code == 200:
                    data = res.json()
                    self.after(0, lambda: self.connection_status_var.set("🟢 Đã kết nối (Online)"))
                    self.after(0, lambda: self.status_log(f"✅ RESP: 200 OK | {data}"))
                    self.after(0, lambda: messagebox.showinfo("Kết Nối Thành Công", f"Kết nối Server thành công!\nTrạng thái: {data.get('status')}"))
                else:
                    self.after(0, lambda: self.connection_status_var.set("🔴 Lỗi HTTP"))
                    self.after(0, lambda: self.status_log(f"❌ RESP: HTTP {res.status_code}"))
                    self.after(0, lambda: messagebox.showwarning("Lỗi Server", f"Server trả về mã HTTP {res.status_code}"))
            except Exception as e:
                self.after(0, lambda: self.connection_status_var.set("🔴 Lỗi kết nối"))
                self.after(0, lambda: self.status_log(f"❌ FAIL: {str(e)}"))
                self.after(0, lambda: messagebox.showerror("Lỗi Kết Nối", f"Không thể kết nối Server:\n{str(e)}"))

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
            self.status_log("⚠️ Yêu cầu DỪNG tiến trình được kích hoạt.")

    def get_selected_motion_code(self, label_value: str) -> str:
        for code, label in MOTION_LABELS.items():
            if label == label_value:
                if code == "random":
                    return random.choice(MOTION_EFFECTS)
                return code
        return random.choice(MOTION_EFFECTS)

    def download_image_task(self, base_url: str, task_id: str, save_filename: str) -> bool:
        """Tải file ảnh AI gốc .png từ API /image/{task_id}."""
        image_url = f"{base_url}/image/{task_id}"
        out_dir = Path(self.output_dir_var.get().strip())
        out_dir.mkdir(parents=True, exist_ok=True)
        dest_path = out_dir / save_filename

        try:
            res = requests.get(image_url, timeout=30)
            if res.status_code == 200:
                with open(dest_path, "wb") as f:
                    f.write(res.content)
                self.status_log(f"🖼️ ĐÃ TẢI ÁNH AI GỐC (.png): {save_filename}")
                return True
            else:
                self.status_log(f"⚠️ Không thể tải ảnh AI gốc: HTTP {res.status_code}")
        except Exception as e:
            self.status_log(f"⚠️ Lỗi tải ảnh AI gốc ({task_id[:8]}): {e}")
        return False

    # ================= PIPELINE XỬ LÝ TÁC VỤ & REAL-TIME STATUS LOGGING =================
    def wait_and_download_task(self, base_url: str, task_id: str, save_filename: str) -> bool:
        status_url = f"{base_url}/status/{task_id}"
        download_url = f"{base_url}/download/{task_id}"
        out_dir = Path(self.output_dir_var.get().strip())
        out_dir.mkdir(parents=True, exist_ok=True)
        dest_path = out_dir / save_filename

        self.after(0, lambda: self.active_task_id_var.set(task_id[:12] + "..."))
        self.status_log(f"▶️ BẮT ĐẦU MONITOR TASK: {task_id}")

        while not self.stop_requested:
            try:
                res = requests.get(status_url, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    status = data.get("status")
                    progress = data.get("progress", 0.0)
                    detail = data.get("detail", "")

                    log_msg = f"🌐 GET /status/{task_id[:8]} -> 200 OK | Status: {status} ({progress:.1f}%) | {detail}"
                    self.after(0, lambda m=log_msg: self.status_log(m))

                    self.after(0, lambda s=status: self.active_task_status_var.set(s))
                    self.after(0, lambda p=progress: self.active_task_progress_var.set(f"{p:.1f} %"))

                    self.after(0, lambda p=progress: self.progress_var.set(p))
                    self.after(0, lambda d=detail: self.update_status(d))

                    if status == "completed":
                        self.status_log(f"🎉 TASK COMPLETED: {task_id[:8]} | Bắt đầu tải video...")

                        dl_res = requests.get(download_url, timeout=60, stream=True)
                        if dl_res.status_code == 200:
                            with open(dest_path, "wb") as f:
                                for chunk in dl_res.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            self.status_log(f"💾 ĐÃ TẢI THÀNH CÔNG VIDEO: {save_filename}")
                            with self.counter_lock:
                                self.success_count += 1
                            self.after(0, self.update_stats_label)
                            return True
                        else:
                            self.status_log(f"❌ ERROR DOWNLOAD: HTTP {dl_res.status_code}")
                            with self.counter_lock:
                                self.failed_count += 1
                            self.after(0, self.update_stats_label)
                            return False
                    elif status == "failed":
                        err_msg = data.get('error')
                        self.status_log(f"❌ TASK FAILED: {task_id[:8]} | {err_msg}")
                        with self.counter_lock:
                            self.failed_count += 1
                        self.after(0, self.update_stats_label)
                        return False

                time.sleep(2.0)
            except Exception as e:
                self.status_log(f"⚠️ RETRY POLLING ({task_id[:8]}): {e}")
                time.sleep(3.0)

        self.status_log(f"⏹️ TASK STOPPED BY USER: {task_id[:8]}")
        return False

    # --- TÍNH NĂNG 1: TEXT TO VIDEO BATCH (SEQUENTIAL RENDERING & SAVE AI IMAGE) ---
    def start_text_batch(self):
        raw_text = self.txt_prompts.get("1.0", "end").strip()
        prompts = [line.strip() for line in raw_text.splitlines() if line.strip()]

        if not prompts:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập ít nhất 1 text prompt!")
            return

        self.is_processing = True
        self.stop_requested = False
        self.total_tasks = len(prompts)
        self.success_count = 0
        self.failed_count = 0
        self.update_stats_label()

        self.set_buttons_state(True)

        base_url = self.get_api_base()
        width = int(self.spn_txt_width.get())
        height = int(self.spn_txt_height.get())
        num_frames = int(self.spn_txt_frames.get())
        fps = int(self.spn_txt_fps.get())
        save_image = self.chk_txt_save_img_var.get()
        selected_motion_label = self.txt_motion_var.get()

        self.status_log(f"🚀 Bắt đầu Batch {len(prompts)} Prompts (Text to Video Mode)...")

        def worker():
            total = len(prompts)
            for idx, prompt in enumerate(prompts, start=1):
                if self.stop_requested:
                    break

                effect = self.get_selected_motion_code(selected_motion_label)
                self.status_log(f"\n--- [{idx}/{total}] Prompt: '{prompt[:35]}...' | {width}x{height} | Hiệu ứng: {effect} ---")

                payload = {
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "num_frames": num_frames,
                    "fps": fps,
                    "motion_type": effect,
                }

                try:
                    self.status_log(f"📤 POST /generate | Prompt: '{prompt[:20]}...'")
                    res = requests.post(f"{base_url}/generate", json=payload, timeout=15)
                    if res.status_code == 202:
                        task_id = res.json().get("task_id")
                        self.status_log(f"✅ POST 202 Accepted | Task ID: {task_id}")
                        safe_title = "".join(c if c.isalnum() else "_" for c in prompt[:20]).strip("_")
                        save_filename = f"text_{idx:03d}_{safe_title}_{effect}.mp4"

                        success = self.wait_and_download_task(base_url, task_id, save_filename)

                        # Tải kèm ảnh AI gốc nếu chọn Checkbox
                        if success and save_image:
                            img_filename = f"text_{idx:03d}_{safe_title}.png"
                            self.download_image_task(base_url, task_id, img_filename)
                    else:
                        self.status_log(f"❌ POST FAILED: HTTP {res.status_code}")
                        with self.counter_lock:
                            self.failed_count += 1
                        self.after(0, self.update_stats_label)
                except Exception as e:
                    self.status_log(f"❌ REQ EXCEPTION: {e}")
                    with self.counter_lock:
                        self.failed_count += 1
                    self.after(0, self.update_stats_label)

            self.after(0, lambda: self.status_log("🎉 BATCH FINISHED: Hoàn tất tất cả Text Prompts."))
            self.after(0, lambda: self.update_status("Hoàn tất!"))
            self.after(0, lambda: self.progress_var.set(100.0))
            self.after(0, lambda: self.set_buttons_state(False))
            self.is_processing = False

        threading.Thread(target=worker, daemon=True).start()

    # --- TÍNH NĂNG 2: IMAGE TO VIDEO BATCH (MULTI-THREADING 4 LUỒNG MẶC ĐỊNH) ---
    def start_image_batch(self):
        folder_path = self.img_folder_var.get().strip()
        p = Path(folder_path)
        if not p.is_dir():
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn thư mục chứa ảnh hợp lệ!")
            return

        files = [f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        files.sort()

        if not files:
            messagebox.showwarning("Cảnh báo", "Không tìm thấy file ảnh hợp lệ nào trong thư mục!")
            return

        self.is_processing = True
        self.stop_requested = False
        self.total_tasks = len(files)
        self.success_count = 0
        self.failed_count = 0
        self.update_stats_label()

        self.set_buttons_state(True)

        base_url = self.get_api_base()
        num_frames = int(self.spn_img_frames.get())
        fps = int(self.spn_img_fps.get())
        max_workers = max(1, min(4, int(self.spn_img_threads.get())))
        selected_motion_label = self.img_motion_var.get()

        self.status_log(f"🚀 Bắt đầu Batch {len(files)} Ảnh (Base64 Mode) - Chạy {max_workers} Luồng Song Song...")

        def process_single_image(idx_and_file):
            idx, img_file = idx_and_file
            if self.stop_requested:
                return

            effect = self.get_selected_motion_code(selected_motion_label)
            self.status_log(f"\n--- [Luồng {threading.current_thread().name}] [{idx}/{len(files)}] File ảnh: '{img_file.name}' | Hiệu ứng: {effect} ---")

            try:
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

                self.status_log(f"📤 POST /generate-from-image | Image: {img_file.name}")
                res = requests.post(f"{base_url}/generate-from-image", json=payload, timeout=30)
                if res.status_code == 202:
                    task_id = res.json().get("task_id")
                    self.status_log(f"✅ POST 202 Accepted | Task ID: {task_id}")
                    save_filename = f"img_{idx:03d}_{img_file.stem}_{effect}.mp4"

                    self.wait_and_download_task(base_url, task_id, save_filename)
                else:
                    self.status_log(f"❌ POST BASE64 FAILED: HTTP {res.status_code}")
                    with self.counter_lock:
                        self.failed_count += 1
                    self.after(0, self.update_stats_label)
            except Exception as e:
                self.status_log(f"❌ EXCEPTION BASE64: {e}")
                with self.counter_lock:
                    self.failed_count += 1
                self.after(0, self.update_stats_label)

        def master_worker():
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="Worker") as executor:
                items = list(enumerate(files, start=1))
                futures = [executor.submit(process_single_image, item) for item in items]
                for future in futures:
                    future.result()

            self.after(0, lambda: self.status_log("🎉 BATCH FINISHED: Hoàn tất tất cả Ảnh trong thư mục."))
            self.after(0, lambda: self.update_status("Hoàn tất!"))
            self.after(0, lambda: self.progress_var.set(100.0))
            self.after(0, lambda: self.set_buttons_state(False))
            self.is_processing = False

        threading.Thread(target=master_worker, daemon=True).start()


if __name__ == "__main__":
    app = ProfessionalVideoStudioApp()
    app.mainloop()
