# 🎵 AI Music Detector Agent (Deep Forensic 4-Tier)

Hệ thống AI Agent thông minh giúp phân loại chính xác các bài hát do **AI sáng tác (Suno, Udio, Stable Audio...)** hay do **Con người làm**, sử dụng phân tích vật lý âm học sâu kết hợp phát hiện vết vân deconvolution **Fourier Fakeprint (Deezer ISMIR 2025)** và mô hình đa phương thức **Gemini 3.1 / 3.5 Flash Lite** từ Google AI Studio với chi phí **$0** (Free Tier).

---

## 🌟 Tính năng nổi bật & Đột phá công nghệ

* **Hệ thống phân tầng 4 nhãn (4-Tier Verdict System):**
  * 🟢 **CONFIRMED HUMAN:** Nhạc do con người sáng tác, thu âm mộc / phòng thu tự nhiên.
  * 🟡 **PROCESSED HUMAN:** Nhạc người làm có can thiệp công nghệ phòng thu (Auto-Tune / Melodyne, beat Trap/EDM/MIDI số, nén YouTube) - **Chấm dứt hoàn toàn tình trạng bắt oan nghệ sĩ thật!**
  * 🟠 **HYBRID / AI-ASSISTED:** Tác phẩm lai ghép (Người phối khí dùng giọng AI, hoặc người hát trên beat AI).
  * 🔴 **AI-GENERATED:** Bài hát sinh hoàn toàn bằng mô hình AI (Suno, Udio, Stable Audio...).

* **Phát hiện vết vân Deconvolution Fourier (Deezer ISMIR 2025):**
  * Tích hợp thuật toán phân tích gai chu kỳ deconvolution trong miền Fourier (`FakeprintDetector`).
  * Phát hiện chính xác các mô hình AI từ Suno v3.5, Suno v4, Udio 1.5 mà không bị lừa bởi âm sắc hay EQ.

* **Bù trừ nén Codec Web (YouTube Opus / AAC):**
  * Tự động nhận diện dải cao bị cắt giảm tự nhiên do codec nén truyền tải của YouTube để không phán nhầm cho ca sĩ.
  * Thay thế ngưỡng cứng 16kHz bằng phép đo độ mở dải tần tự nhiên **Spectral Rolloff (85% & 95%)** và tỷ lệ sóng hài **Harmonic-to-Percussive (HPSS)**.

* **Biểu đồ giám định âm học 2 tầng:**
  * Tầng 1: Mel-Spectrogram (0 - 22.05kHz) trực quan hóa năng lượng dải tần.
  * Tầng 2: Phổ vân Fourier Deconvolution trực quan hóa phần dư đỉnh (Peak Residues).

* **Đầu vào linh hoạt & Xuất báo cáo:**
  * Quét qua Link (YouTube, SoundCloud, TikTok...) hoặc tên bài hát.
  * Quét hàng loạt qua Text hoặc file Excel (`.xlsx`, `.csv`).
  * Xuất file báo cáo giám định chi tiết ra Excel / CSV.

---

## 🚀 Hướng dẫn cài đặt & Khởi chạy

### 1. Chuẩn bị API Key Google AI Studio (Miễn phí)

1. Truy cập [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Tạo một API Key miễn phí.
3. Tạo file `.env` từ mẫu `.env.example` và dán key của bạn vào:

   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### 2. Khởi chạy Giao diện Web (Streamlit)

* **Cách 1 (Nhanh nhất):** Nhấp đúp chuột vào file `run_app.bat` trên Windows (script tự động dùng `uv run`).
* **Cách 2 (Bằng uv CLI):**

  ```bash
  uv run streamlit run frontend/app.py
  ```

### 3. Sử dụng qua dòng lệnh (CLI)

```bash
# Kiểm tra 1 bài hát qua link hoặc tên bài:
uv run main.py --query "Sơn Tùng M-TP - Đừng Làm Trái Tim Anh Đau"

# Quét hàng loạt từ file Excel:
uv run main.py --file "sample_songs.xlsx"
```

---

## 📂 Kiến Trúc Hệ Thống Chuẩn Quốc Tế (FAANG Enterprise Architecture)

Dự án được tái cấu trúc triệt để theo mô hình phân tách **Frontend ra Frontend, Backend ra Backend**, tuân thủ nguyên lý Clean Architecture & Dependency Inversion Principle (DIP):

```text
d:/test/musicidentify/
├── backend/                       # 🟢 BACKEND LAYER HOÀN CHỈNH
│   ├── api/                       # [LỚP 1] CONTROLLER & REST API
│   │   ├── app.py                 # FastAPI Application (Swagger /docs)
│   │   ├── controllers/           # Controllers (TrackController, BatchController, ReportController)
│   │   ├── routers/               # REST API Endpoints (/tracks, /batch, /reports, /health)
│   │   └── schemas/               # Request & Response DTOs chuẩn hóa
│   ├── core/                      # DOMAIN CORE, CONFIG & INTERFACES
│   │   ├── config.py              # Cấu hình môi trường & đường dẫn
│   │   ├── container.py           # Dependency Injection Container (IoC)
│   │   ├── exceptions.py          # Domain & Application Exceptions
│   │   ├── models/                # Domain Entities (TrackItem, DetectionResult, VerdictType...)
│   │   └── interfaces/            # KHÔNG GỌI TRỰC TIẾP SERVICE/REPO - THÔNG QUA INTERFACES:
│   │       ├── services/          # IAudioEngine, ISignalAnalyzer, IFakeprintDetector, IAIClassifier...
│   │       └── repositories/      # ITrackRepository, IReportRepository, IStorageRepository
│   ├── services/                  # [LỚP 2] SERVICE LAYER (Hiện thực các Service Interfaces)
│   │   ├── audio_engine_service.py       # Tải & cắt phân đoạn âm thanh đa tầng
│   │   ├── signal_analyzer_service.py    # Phân tích âm học (inject IFakeprintDetector)
│   │   ├── fakeprint_detector_service.py # Thuật toán Fourier Deconvolution ISMIR 2025
│   │   ├── gemini_classifier_service.py  # Giám định AI kèm Failover 1 > 2 > 3
│   │   ├── resolver_service.py           # Phân giải dữ liệu đầu vào (URL, file, YouTube)
│   │   ├── detection_service.py          # Điều phối nhận diện đơn bài (Orchestrator)
│   │   ├── batch_service.py              # Xử lý hàng loạt có Rate Limiting & bảo toàn 100% bài
│   │   └── report_service.py             # Xuất báo cáo (inject IReportRepository)
│   └── repositories/              # [LỚP 3] REPOSITORY LAYER (Data Persistence & Storage)
│       ├── in_memory_track_repository.py # Quản lý dữ liệu bài hát và kết quả
│       ├── file_report_repository.py     # Xuất & lưu trữ báo cáo Excel (.xlsx) / CSV
│       └── temp_storage_repository.py    # Quản lý và dọn dẹp file tạm an toàn
│
├── frontend/                      # 🔵 FRONTEND LAYER HOÀN CHỈNH
│   ├── app.py                     # Entrypoint Web Streamlit hiện đại
│   ├── assets/
│   │   └── styles.css             # Hệ thống thiết kế Dark Glassmorphism cao cấp
│   ├── components/                # Modular UI Components (Header, Sidebar, VerdictCard, Metrics...)
│   ├── services/
│   │   └── backend_client.py      # Frontend Backend Adapter kết nối Controller/API
│   └── state/
│       └── session_state.py       # Quản lý trạng thái phiên làm việc (Session State)
│
├── tests/                         # 🧪 Test Suite toàn diện (Kiểm thử 3 tầng, Failover, Pipeline)
├── main.py                        # CLI & Unified App Launcher (--ui, --api, --query, --file)
├── run_app.bat                    # Script khởi chạy nhanh giao diện Web (Streamlit)
└── pyproject.toml                 # Cấu hình dự án (FastAPI, Uvicorn, Streamlit, Librosa...)
```

### 💡 Khởi chạy hệ thống bằng `uv`

* **Khởi chạy Web UI:** `uv run streamlit run frontend/app.py` hoặc click đúp file `run_app.bat`
* **Khởi chạy REST API Server:** `uv run main.py --api` (Tài liệu Swagger: `http://localhost:8000/docs`)
* **Quét 1 bài hát qua CLI:** `uv run main.py --query "Tên bài hoặc URL"`
* **Quét hàng loạt từ file:** `uv run main.py --file "sample_songs.xlsx"`
* **Chạy kiểm thử tự động:** `uv run python -m unittest discover tests`
* **Đánh giá độ chính xác & Vẽ Confusion Matrix:** `uv run python tests/test_confusion_matrix.py --sample 20` (hoặc `--from-report` để vẽ từ file đối soát trước đó)
