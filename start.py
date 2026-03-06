import os
import subprocess
import sys
from pathlib import Path
from loguru import logger
import time
import requests

ROOT = Path(__file__).parent

PIPELINE_SCRIPT = ROOT / "scripts" / "run_pipeline.py"
REQUIREMENTS = ROOT / "requirements.txt"

FRONTEND_APP = ROOT / "frontend" / "app.py"

DATA_DIR = ROOT / "data"
EMBEDDINGS = DATA_DIR / "embeddings.npy"
PCA_MODEL = DATA_DIR / "pca_model.joblib"
FCM_CENTERS = DATA_DIR / "fcm_centers.npy"


def install_requirements():
    """Install dependencies from requirements.txt"""
    if REQUIREMENTS.exists():
        logger.info("Installing dependencies...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)]
        )
    else:
        logger.warning("requirements.txt not found — skipping install")


def pipeline_needed():
    """Check if pipeline artifacts exist"""
    return not (
        EMBEDDINGS.exists()
        and PCA_MODEL.exists()
        and FCM_CENTERS.exists()
    )


def run_pipeline():
    """Run data preprocessing pipeline"""
    logger.info("Running setup pipeline...")
    subprocess.check_call(
        [
            sys.executable,
            str(PIPELINE_SCRIPT),
            "--dataset-path",
            "data/20_newsgroups",
            "--skip-k-selection"
        ]
    )
def wait_for_backend(timeout=60):
    """Wait until backend API is ready"""
    logger.info("Waiting for backend to be ready...")

    start_time = time.time()

    while True:
        try:
            r = requests.get("http://127.0.0.1:8000/health")

            if r.status_code == 200:
                logger.success("Backend is ready.")
                return

        except Exception:
            pass

        if time.time() - start_time > timeout:
            logger.error("Backend failed to start within timeout.")
            sys.exit(1)

        time.sleep(2)

def start_backend():
    """Start FastAPI backend"""
    logger.info("Starting backend API...")

    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
    )


def start_frontend():
    """Start Streamlit frontend"""
    logger.info("Starting frontend UI...")

    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(FRONTEND_APP),
        ]
    )


def main():
    logger.info("=" * 60)
    logger.info("Semantic Search System Launcher")
    logger.info("=" * 60)

    install_requirements()

    if pipeline_needed():
        logger.info("Pipeline artifacts missing — running pipeline.")
        run_pipeline()
    else:
        logger.info("Pipeline artifacts already exist — skipping pipeline.")

    backend = start_backend()


    wait_for_backend()

    frontend = start_frontend()

    logger.info("")
    logger.info("System started successfully!")
    logger.info("")
    logger.info("Backend API:  http://localhost:8000")
    logger.info("API Docs:     http://localhost:8000/docs")
    logger.info("Frontend UI:  http://localhost:8501")
    logger.info("")

    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        logger.info("Shutting down services...")
        backend.terminate()
        frontend.terminate()


if __name__ == "__main__":
    main()