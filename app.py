import runpy
import sys
from pathlib import Path

# Thêm thư mục gốc vào sys.path để các module backend/frontend import đồng nhất
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Chạy giao diện Streamlit từ frontend/app.py (Hỗ trợ Hugging Face Spaces & Streamlit Cloud)
frontend_entrypoint = root_dir / "frontend" / "app.py"
runpy.run_path(str(frontend_entrypoint), run_name="__main__")
