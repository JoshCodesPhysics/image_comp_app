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

# ---------------------------
# Database setup (SQLAlchemy 2.x Async)
# ---------------------------
# So that all classes inheriting from Base have a common metadata registry (Base.metadata.create_all() creates all tables in the database at once)
# Tracks table metadata, provides python object conversion into dbs, can create tables from models, provides query methods
class Base(DeclarativeBase):
    pass

class Comparison(Base):
    __tablename__ = "comparisons"

    # Object relational mapping so that rows/columns in the database are mapped to python objects and vice versa
    # 36 char hex UUID primary key, e.g., 0123456789abcdef0123456789abcdef
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    correlation_score: Mapped[float] = mapped_column(Float, nullable=False)
    chi_square_score: Mapped[float] = mapped_column(Float, nullable=False)
    bhattacharyya_score: Mapped[float] = mapped_column(Float, nullable=False)
    diff_image_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)

# Create async database engine - handles connection pool to PostgreSQL
# echo=False: don't print SQL queries to console 
# future=True: enables SQLAlchemy 2.x features
# Concurrent database operations running at the same time as web server operations
# Brides the python code to the database
engine = create_async_engine(DATABASE_URL, echo=False, future=True)

# Create session factory - generates database sessions for each request
# expire_on_commit=False: keeps objects accessible after commit
# class_=AsyncSession: uses async session for non-blocking database operations
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Pydantic scheme for the API response (copy of Comparison but with url instead of path)
class ComparisonResponse(BaseModel):
    id: str
    correlation: float = Field(..., ge=0, le=100)
    chi_square: float = Field(..., ge=0, le=100)
    bhattacharyya: float = Field(..., ge=0, le=100)
    diff_image_url: str
    created_at: datetime

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)