import sys
from pathlib import Path

# Thêm thư mục gốc vào sys.path để các module backend/frontend import đồng nhất
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from frontend.app import main

if __name__ == "__main__":
    main()
else:
    main()
