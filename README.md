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

*(Lưu ý: Bạn cũng có thể nhập API Key trực tiếp trên giao diện Web mà không cần sửa file).*

### 2. Khởi chạy Giao diện Web (Streamlit)

* **Cách 1 (Nhanh nhất):** Nhấp đúp chuột vào file `run_app.bat` trên Windows.
* **Cách 2 (Bằng uv):**

  ```bash
  uv run streamlit run src/ui/app.py
  ```

* **Cách 3 (Bằng môi trường ảo):**

  ```bash
  .venv\Scripts\python.exe -m streamlit run src/ui/app.py
  ```

### 3. Sử dụng qua dòng lệnh (CLI)

```bash
# Kiểm tra 1 bài hát qua link hoặc tên bài:
uv run main.py --query "Sơn Tùng M-TP - Đừng Làm Trái Tim Anh Đau"

# Quét hàng loạt từ file Excel:
uv run main.py --file "sample_songs.xlsx"
```

---

## 📂 Cấu trúc thư mục dự án

```text
d:/test/musicidentify/
├── .env.example              # Mẫu biến môi trường
├── pyproject.toml            # Cấu hình dự án & dependencies chuẩn uv
├── run_app.bat               # File chạy nhanh Web UI (hỗ trợ cả uv và .venv)
├── main.py                   # Điểm khởi chạy CLI & Web launcher
├── README.md                 # Tài liệu hướng dẫn
├── data/
│   ├── temp/                 # File audio 30s và ảnh phổ giám định 2 tầng
│   └── outputs/              # File báo cáo kết quả Excel / CSV
├── tests/
│   └── test_pipeline.py      # Bộ kiểm thử tự động toàn diện
└── src/
    ├── config.py             # Cấu hình hệ thống & đường dẫn FFmpeg
    ├── models/
    │   └── schemas.py        # Pydantic Schemas (VerdictType 4 nhãn, AcousticMetrics...)
    ├── services/
    │   ├── resolver.py       # Phân giải link và tìm kiếm bài hát
    │   ├── audio_engine.py   # Tải & cắt 30s audio bằng yt-dlp + FFmpeg
    │   ├── fakeprint_detector.py # Bắt gai chu kỳ Deconvolution ISMIR 2025
    │   ├── signal_analyzer.py# Đo Spectral Rolloff 95% & xuất ảnh 2 tầng
    │   ├── gemini_classifier.py # Biện luận ngữ cảnh 4 nhãn bằng Gemini Flash Lite
    │   └── batch_manager.py  # Xử lý hàng loạt có Rate Limiter & xuất Excel
    └── ui/
        └── app.py            # Giao diện Web Streamlit chuẩn giám định số
```
