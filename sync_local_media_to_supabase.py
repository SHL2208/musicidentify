"""
Công cụ Đồng bộ Tệp tin Đa phương tiện Cục bộ (Local Media) lên Supabase Storage
Quét toàn bộ các file mẫu âm thanh (.wav) và ảnh quang phổ (.png) trong data/temp,
tải lên các bucket trên Supabase Cloud và dọn dẹp ổ đĩa cục bộ.
"""

import os
import sys
from pathlib import Path

# Thêm root vào sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dotenv import load_dotenv

load_dotenv()

from backend.core.config import TEMP_DIR
from backend.core.database import get_db_session, is_database_available
from backend.core.models.db_models import DetectionResultModel, MediaAssetModel
from backend.services.supabase_storage_service import SupabaseStorageService


def sync_media_to_supabase(cleanup_after_upload: bool = True):
    print("=" * 70)
    print("☁️  BẮT ĐẦU ĐỒNG BỘ MEDIA ASSETS LÊN SUPABASE CLOUD STORAGE")
    print("=" * 70)

    if not is_database_available():
        print("❌ LỖI: CSDL Supabase không khả dụng. Vui lòng kiểm tra DATABASE_URL trong .env.")
        return

    storage_service = SupabaseStorageService()
    if not storage_service.client:
        print("❌ LỖI: Không thể khởi tạo Supabase Storage Client (kiểm tra SUPABASE_URL & SUPABASE_KEY).")
        return

    wav_files = list(TEMP_DIR.glob("*_multisegment.wav"))
    png_files = list(TEMP_DIR.glob("*_forensic.png"))

    total_files = len(wav_files) + len(png_files)
    print(f"📁 Tìm thấy {len(wav_files)} file âm thanh (.wav) và {len(png_files)} file ảnh quang phổ (.png) trong {TEMP_DIR}")
    if total_files == 0:
        print("🎉 Thư mục data/temp hiện đang trống, không có file nào cần đồng bộ.")
        return

    synced_wav = 0
    synced_png = 0

    with get_db_session() as session:
        # 1. Đồng bộ file ảnh quang phổ (.png)
        print("\n🖼️  [1/2] Đang đồng bộ ảnh quang phổ lên bucket 'spectrograms'...")
        for p in png_files:
            file_name = p.name
            track_id = file_name.replace("_multisegment_forensic.png", "").replace("_forensic.png", "")
            
            public_url = storage_service.upload_file(p, bucket="spectrograms")
            if public_url.startswith("http"):
                synced_png += 1
                # Cập nhật trong bảng detection_results
                res_id = f"res_{track_id}"
                dr = session.get(DetectionResultModel, res_id)
                if dr:
                    # update model if needed
                    pass

                # Cập nhật trong bảng media_assets
                asset_id = f"asset_spec_{track_id}"
                ma = session.get(MediaAssetModel, asset_id)
                if ma:
                    ma.public_url = public_url
                else:
                    new_ma = MediaAssetModel(
                        id=asset_id,
                        track_id=track_id,
                        asset_type="SPECTROGRAM_FORENSIC",
                        file_path=str(p),
                        public_url=public_url,
                        file_size=p.stat().st_size if p.exists() else 0,
                    )
                    session.add(new_ma)

                if cleanup_after_upload:
                    p.unlink(missing_ok=True)

        # 2. Đồng bộ file âm thanh (.wav)
        print("\n🎵 [2/2] Đang đồng bộ file âm thanh lên bucket 'audio-samples'...")
        for w in wav_files:
            file_name = w.name
            track_id = file_name.replace("_multisegment.wav", "")
            
            public_url = storage_service.upload_file(w, bucket="audio-samples")
            if public_url.startswith("http"):
                synced_wav += 1
                asset_id = f"asset_audio_{track_id}"
                ma = session.get(MediaAssetModel, asset_id)
                if ma:
                    ma.public_url = public_url
                else:
                    new_ma = MediaAssetModel(
                        id=asset_id,
                        track_id=track_id,
                        asset_type="AUDIO_SAMPLE",
                        file_path=str(w),
                        public_url=public_url,
                        file_size=w.stat().st_size if w.exists() else 0,
                    )
                    session.add(new_ma)

                if cleanup_after_upload:
                    w.unlink(missing_ok=True)

    print("\n" + "=" * 70)
    print(f"✅ HOÀN TẤT ĐỒNG BỘ:")
    print(f"   • Đã tải lên và gắn link Cloud cho: {synced_png} ảnh biểu đồ & {synced_wav} file âm thanh")
    if cleanup_after_upload:
        print("   • Đã tự động dọn dẹp giải phóng dung lượng ổ đĩa cục bộ trên máy của bạn!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Đồng bộ media local lên Supabase Storage")
    parser.add_argument("--no-cleanup", action="store_true", help="Không xóa file local sau khi upload")
    args = parser.parse_args()
    
    sync_media_to_supabase(cleanup_after_upload=not args.no_cleanup)
