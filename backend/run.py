import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("TEXT_TO_VIDEO_PORT", "3930"))
    host = os.environ.get("TEXT_TO_VIDEO_HOST", "127.0.0.1")
    print(f"Starting Text-to-Image-to-Video Backend on http://{host}:{port}")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
    )
