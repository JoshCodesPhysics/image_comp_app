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
import io
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2
import uvicorn
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

# UploadFile contains file object, name, content type (image/jpeg, image/png etc.), file size in bytes
def _read_image_to_bgr(upload: UploadFile) -> np.ndarray:
    """Read an UploadFile into an OpenCV BGR image with comprehensive validation."""
    # Check file size (prevent extremely large files)
    if hasattr(upload, 'size') and upload.size and upload.size > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")
    
    # Check content type for basic validation
    if upload.content_type and not upload.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {upload.content_type}. Expected image file.")
    
    try:
        # Returns raw bytes
        data = upload.file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed reading file: {e}")
    
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload")
    
    # Check minimum file size (prevent empty or very small files)
    if len(data) < 100:  # Minimum 100 bytes for a valid image
        raise HTTPException(status_code=400, detail="File too small to be a valid image")
    
    try:
        # Convert raw bytes to numpy array
        nparr = np.frombuffer(data, np.uint8)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to convert file data to image array: {e}")
    
    # 3D matrix of pixels (height, width, colour channels)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format or corrupted image data")
    
    # Check if image has valid dimensions
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Image has invalid dimensions")
    
    # Check if image has the expected 3 channels (BGR)
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise HTTPException(status_code=400, detail="Image must be a color image with 3 channels (BGR)")
    
    return img

def _ensure_same_size(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize img2 to img1's size if different, preserving aspect via direct resize."""
    try:
        # Validate input arrays
        if img1 is None or img2 is None:
            raise ValueError("One or both images are None")
        
        if len(img1.shape) != 3 or len(img2.shape) != 3:
            raise ValueError("Images must be 3D arrays (height, width, channels)")
        
        # Get dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Validate dimensions
        if h1 is None or w1 is None or h2 is None or w2 is None:
            raise ValueError("Image dimensions are None")
        
        if h1 <= 0 or w1 <= 0 or h2 <= 0 or w2 <= 0:
            raise ValueError("Image dimensions must be positive")
        
        # If same size, return as-is
        if (h1, w1) == (h2, w2):
            return img1, img2
        
        # Resize img2 to match img1's dimensions
        try:
          resized = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_AREA)
        except Exception as e:
            raise RuntimeError(f"Error in cv2.resize: {e}")
        
        # Validate resize result
        if resized is None:
            raise RuntimeError("cv2.resize returned None")
        
        return img1, resized
        
    except Exception as e:
        raise RuntimeError(f"Error in _ensure_same_size: {e}")

def test_image_reading():
    """Test the _read_image_to_bgr function with a real image file."""
    
    try:
        try:
            with open("image1.png", "rb") as f:
                image_data = f.read()
            print(f"✅ File reading successful: {len(image_data)} bytes")
        except Exception as e:
            print(f"❌ File reading failed: {e}")
            return None
        
        try:
            upload_file = UploadFile(
                file=io.BytesIO(image_data),
                filename="image1.png"
            )
            print(f"✅ UploadFile creation successful: {upload_file.filename}")
        except Exception as e:
            print(f"❌ UploadFile creation failed: {e}")
            return None
        
        try:
            img = _read_image_to_bgr(upload_file)
            print(f"✅ Image processing successful!")
            print(f"   Image shape: {img.shape}")
            print(f"   Image dtype: {img.dtype}")
            print(f"   Image size: {len(image_data)} bytes")
            print(f"   Filename: {upload_file.filename}")
            return img
        except Exception as e:
            print(f"❌ Image processing failed: {e}")
            return None
        
    except Exception as e:
        print(f"❌ Overall test failed: {e}")
        return None

# Test the _ensure_same_size function with both images
def test_image_size_normalization():
    """Test the _ensure_same_size function with image1.png and image2.png."""
    
    try:
        # Block 1: Read image1.png
        try:
            with open("image1.png", "rb") as f:
                image1_data = f.read()
            print(f"✅ Image1 reading successful: {len(image1_data)} bytes")
        except Exception as e:
            print(f"❌ Image1 reading failed: {e}")
            return None
        
        # Block 2: Read image2.png
        try:
            with open("image2.png", "rb") as f:
                image2_data = f.read()
            print(f"✅ Image2 reading successful: {len(image2_data)} bytes")
        except Exception as e:
            print(f"❌ Image2 reading failed: {e}")
            return None
        
        # Block 3: Convert both to numpy arrays using _read_image_to_bgr logic
        try:
            # Convert image1
            upload_file1 = UploadFile(file=io.BytesIO(image1_data), filename="image1.png")
            img1 = _read_image_to_bgr(upload_file1)
            print(f"✅ Image1 processing successful: shape {img1.shape}")
            
            # Convert image2
            upload_file2 = UploadFile(file=io.BytesIO(image2_data), filename="image2.png")
            img2 = _read_image_to_bgr(upload_file2)
            print(f"✅ Image2 processing successful: shape {img2.shape}")
            
        except Exception as e:
            print(f"❌ Image processing failed: {e}")
            return None
        
        # Block 4: Test _ensure_same_size function
        try:
            img1_resized, img2_resized = _ensure_same_size(img1, img2)
            print(f"✅ Size normalization successful!")
            print(f"   Original img1 shape: {img1.shape}")
            print(f"   Original img2 shape: {img2.shape}")
            print(f"   Resized img1 shape: {img1_resized.shape}")
            print(f"   Resized img2 shape: {img2_resized.shape}")
            print(f"   Images now have same dimensions: {img1_resized.shape == img2_resized.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Size normalization failed: {e}")
            return None
        
    except Exception as e:
        print(f"❌ Overall test failed: {e}")
        return None

test_image_reading()
test_image_size_normalization()

# API Endpoints
# Test at http://localhost:8000/health
@app.get("/health")
async def health():
    """Simple connectivity test to Postgres"""
    try:
        async with engine.connect() as conn:
            # Execute a simple query to verify the connection
            await conn.execute(text("SELECT 1"))
        db = "ok"
    except Exception as e:
        db = f"error: {e}"
    return {"status": "ok", "db": db}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)