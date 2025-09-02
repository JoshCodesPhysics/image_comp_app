"""
FastAPI Image Comparison Service

Features:
- POST /comparison: Accept two images, compute histogram-based difference scores (Correlation, Chi-Square, Bhattacharyya),
  generate a pixel-wise heatmap overlay (semi-transparent) over the first image, persist results to Postgres, return JSON.
- GET /comparison/{id}: Retrieve a previous comparison by UUID.
- Static files served at /static for diff image retrieval.

Implementation Notes:
- Histograms are computed per color channel (B, G, R) using OpenCV, normalized (L1), and
  compared with cv2.compareHist for each method. Per-channel scores are mapped to a 0–100% difference
  and then averaged across channels to yield a single % per method.
- Heatmap overlay is built from per-pixel absolute difference, Otsu-thresholded to emphasize significant regions,
  colored with COLORMAP_JET, and alpha-blended over the first image.
"""

import os
import uuid
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sqlalchemy import text, String, TIMESTAMP, Float
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from dotenv import load_dotenv

# ---------------------------
# Config
# ---------------------------

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/image_diff")
STATIC_ROOT = Path(os.getenv("STATIC_ROOT", "./static")).resolve()
DIFF_SUBDIR = os.getenv("DIFF_SUBDIR", "diffs")
DIFF_DIR = STATIC_ROOT / DIFF_SUBDIR
DIFF_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Image Comparison Service", version="1.0.0")

# Enable CORS as needed (allow all by default for demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")

# Run (if __main__)
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
