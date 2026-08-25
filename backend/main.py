"""
Clinical QA Studio - FastAPI Backend Entrypoint
Run from the backend/ directory via:
    uv run main.py
or
    python main.py
"""

import sys
import io
from pathlib import Path
import uvicorn

# Ensure UTF-8 stdout encoding on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ensure backend directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.config import settings

if __name__ == "__main__":
    print("=" * 65)
    print(f"[SERVER] Starting {settings.PROJECT_NAME}")
    print(f"[SERVER] Local URL          : http://127.0.0.1:{settings.PORT}")
    print(f"[SERVER] API Documentation  : http://127.0.0.1:{settings.PORT}/docs")
    print(f"[SERVER] Health Check        : http://127.0.0.1:{settings.PORT}/api/v1/health")
    print("=" * 65)
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=settings.PORT,
        reload=False,
    )
