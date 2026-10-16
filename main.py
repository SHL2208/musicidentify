import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Đảm bảo in tiếng Việt mượt mà trên console Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from backend.api.schemas.dtos import TrackDetectRequest
from backend.core.config import API_HOST, API_PORT, DEFAULT_GEMINI_MODEL
from backend.core.container import get_container


def run_streamlit():
    """Khởi động giao diện người dùng Streamlit Web từ thư mục frontend/."""
    app_path = root_dir / "frontend" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    subprocess.run(cmd)

def run_api_server():
    """Khởi động FastAPI REST Server bằng uvicorn."""
    import uvicorn
    print(f"🚀 Đang khởi chạy API Server tại http://{API_HOST}:{API_PORT} (Tài liệu Swagger: http://{API_HOST}:{API_PORT}/docs)...")
    uvicorn.run("backend.api.app:app", host=API_HOST, port=API_PORT, reload=True)

def main():
    parser = argparse.ArgumentParser(description="AI Music Detector Agent - Enterprise Clean Architecture (FAANG Standard)")
    parser.add_argument("--ui", action="store_true", help="Mở giao diện Web trực quan (Streamlit)")
    parser.add_argument("--api", action="store_true", help="Khởi chạy REST API Server (FastAPI)")
    parser.add_argument("--query", type=str, help="Kiểm tra 1 bài hát (Link URL hoặc Tên bài hát)")
    parser.add_argument("--file", type=str, help="Kiểm tra hàng loạt từ file (.txt, .csv, .xlsx)")
    parser.add_argument("--model", type=str, default=DEFAULT_GEMINI_MODEL, help="Model khởi đầu: gemini-3.8-flash (mặc định), gemini-3.7-flash, gemini-3.6-flash...")

    args = parser.parse_args()

    if args.api:
        run_api_server()
        return

    if args.ui or (not args.query and not args.file):
        print("🚀 Đang khởi chạy giao diện Web Streamlit (frontend/app.py)...")
        run_streamlit()
        return

    # Khởi tạo Dependency Injection Container
    container = get_container(default_model=args.model)

    if args.query:
        print(f"🔍 Đang phân tích: '{args.query}' bằng model {args.model} qua Controller Layer...")
        track_controller = container.track_controller
        request = TrackDetectRequest(query=args.query, model_override=args.model)
        response = track_controller.detect_track(request)

        if not response.success or not response.data:
            print(f"❌ Thất bại: {response.error or response.message}")
            sys.exit(1)

        result = response.data
        print("\n" + "="*60)
        print(f"🎵 BÀI HÁT: {result.track_title}")
        print(f"🎯 KẾT LUẬN: {result.verdict_label} (Tự tin: {result.confidence_score*100:.1f}%)")
        print(f"🏷️ PHÂN LOẠI 4 TẦNG: {result.verdict_type.value}")
        if result.acoustic_metrics:
            m = result.acoustic_metrics
            fp_tag = "⚠️ NGHI VẤN AI" if m.fakeprint_detected else "✅ TỰ NHIÊN"
            print(f"🔬 ĐIỂM FAKEPRINT FOURIER (ISMIR 2025): {m.fakeprint_score * 100:.1f}% ({fp_tag})")
            print(f"📡 ĐỘ MỞ DẢI TẦN (ROLLOFF 95%): {m.spectral_rolloff_95_hz:.0f} Hz")
            print(f"📦 ĐÁNH GIÁ NÉN TRUYỀN TẢI: {m.compression_source}")
            print(f"🎼 TỶ LỆ SÓNG HÀI (HARMONIC): {m.harmonic_ratio * 100:.1f}%")
        print(f"🤖 NỀN TẢNG NGHI VẤN: {result.suspected_engine}")
        print("📌 LUẬN CỨ CHÍNH:")
        for r in result.reasons:
            print(f"   - {r}")
        print(f"📝 GIẢI THÍCH CHI TIẾT:\n{result.detailed_explanation}")
        print("="*60)

    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ Không tìm thấy file: {args.file}")
            sys.exit(1)
        print(f"📁 Đang đọc file: {file_path} qua Batch Controller...")

        batch_controller = container.batch_controller

        def progress(idx, total, tr, res, err):
            if res:
                print(f"[{idx}/{total}] ✅ {res.track_title} -> {res.verdict_label} ({res.confidence_score*100:.0f}%)")
            else:
                print(f"[{idx}/{total}] ❌ Lỗi: {tr.raw_query} ({err})")

        response = batch_controller.process_file(file_path, model_override=args.model, on_progress=progress)

        if not response.success or not response.data:
            print(f"❌ Lỗi xử lý lô: {response.error or response.message}")
            sys.exit(1)

        report = response.data
        print("\n🎉 HOÀN THÀNH!")
        print(f"Tổng: {report.total_tracks} | AI: {report.ai_count} | Người: {report.human_count} | Lỗi: {report.failed_count}")
        if report.report_file_path:
            print(f"📥 File báo cáo Excel đã lưu tại: {report.report_file_path}")

if __name__ == "__main__":
    main()
