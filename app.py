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

async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

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
        
        # Calculate total pixels for each image to determine which is larger
        pixels1 = h1 * w1
        pixels2 = h2 * w2

        result: tuple[np.ndarray, np.ndarray] | None = None
        
        try:
            if pixels1 > pixels2:
                # img1 is larger, resize it to match img2 (downsample for better quality)
                resized_img1 = cv2.resize(img1, (w2, h2), interpolation=cv2.INTER_AREA)
                # Validate resize result
                if resized_img1 is None:
                    raise RuntimeError("cv2.resize returned None")
                result = (resized_img1, img2)
            else:
                # img2 is larger, resize it to match img1 (downsample for better quality)
                resized_img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_AREA)
                # Validate resize result
                if resized_img2 is None:
                    raise RuntimeError("cv2.resize returned None")
                result = (img1, resized_img2)
        except Exception as e:
            raise RuntimeError(f"Error in cv2.resize: {e}")
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Error in _ensure_same_size: {e}")

def _compute_channel_hist(bgr_img: np.ndarray, channel_index: int, bins: int = 256) -> np.ndarray:
    """Compute histogram for a specific channel of a BGR image."""
    try:
      hist = cv2.calcHist([bgr_img], [channel_index], None, [bins], [0, 256])
    except Exception as e:
        raise RuntimeError(f"Error in cv2.calcHist: {e}")
    # L1 normalize to sum to 1
    try:
      hist = cv2.normalize(hist, None, alpha=1.0, norm_type=cv2.NORM_L1).flatten()
    except Exception as e:
      raise RuntimeError(f"Error in cv2.normalize: {e}")
    return hist

def _compare_histograms_per_method(h1: np.ndarray, h2: np.ndarray) -> dict:
    """Return raw OpenCV scores for each method on two 1D hist arrays."""
    try:
        # Validate input histograms
        if h1 is None or h2 is None:
            raise ValueError("One or both histograms are None")
        
        if len(h1.shape) != 1 or len(h2.shape) != 1:
            raise ValueError("Histograms must be 1D arrays")
        
        if h1.shape[0] != h2.shape[0]:
            raise ValueError(f"Histograms must have same length: {h1.shape[0]} vs {h2.shape[0]}")
        
        # Convert to float32 for OpenCV compatibility
        h1_float = h1.astype("float32")
        h2_float = h2.astype("float32")
        
        try:
            corr = float(cv2.compareHist(h1_float, h2_float, cv2.HISTCMP_CORREL))
        except Exception as e:
            raise RuntimeError(f"Error computing correlation: {e}")
        
        try:
            chi = float(cv2.compareHist(h1_float, h2_float, cv2.HISTCMP_CHISQR))
        except Exception as e:
            raise RuntimeError(f"Error computing chi-square: {e}")
        
        # Bhattacharyya distance: 0 identical, 1 very different (when L1 normalized)
        # Represents a notion of similarity between two probability distributions.
        try:
            bha = float(cv2.compareHist(h1_float, h2_float, cv2.HISTCMP_BHATTACHARYYA))
        except Exception as e:
            raise RuntimeError(f"Error computing Bhattacharyya: {e}")
        
        return {"corr": corr, "chi": chi, "bha": bha}
        
    except Exception as e:
        raise RuntimeError(f"Error in _compare_histograms_per_method: {e}")

def _find_difference_regions(img1: np.ndarray, img2: np.ndarray, threshold: float = 60.0, min_region_size: int = 50) -> dict:
    """
    Find spatial regions where images differ significantly.
    Returns information about connected regions of difference.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same shape: {img1.shape} vs {img2.shape}")
    
    # Calculate absolute difference for each pixel
    if img1 is None or img2 is None:
        raise ValueError("Images are None")
    try:
      diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    except Exception as e:
        raise RuntimeError(f"Error in calculating the absolute pixel difference: {e}")
    
    # Calculate total difference per pixel (sum across all channels)
    try:
      pixel_diffs = np.sum(diff, axis=2)
    except Exception as e:
        raise RuntimeError(f"Error in calculating the total pixel difference: {e}")
    
    # Create binary mask for significant differences
    try:
      diff_mask = pixel_diffs > threshold
    except Exception as e:
        raise RuntimeError(f"Error in creating the binary mask for significant differences: {e}")
    
    # Find connected components (regions) for pixels P with binary value 1 in mask, connect diagonal and adjacent for better fragmentation.
    # Think minesweeper!
    # X X X
    # X P X
    # X X X
    # num_labels is the number of connected components, labels is a matrix of pixels with 0 for not above threshold, and then 1, 2, 3, etc. for each connected component.
    # 0 0 0 0 0 0 0
    # 0 1 1 0 0 0 0
    # 0 1 1 0 0 2 0
    # 0 0 0 0 0 2 0
    # 0 0 0 2 2 2 0
    # 0 0 0 2 2 0 0
    # 0 0 0 0 0 0 0
    # Stats contains the left-top corner coordinate of the connected region, width, height, and area.
    # Centroids contains the centroid of the connected region.
    try:
      num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
          diff_mask.astype(np.uint8), connectivity=8
      )
    except Exception as e:
        raise RuntimeError(f"Error in finding connected components: {e}")
    
    significant_regions : list[dict[str, Any]] = []
    
    if num_labels is None or num_labels <= 0:
        raise ValueError("Number of labels is None or less than 0")
    # Start from 1 to ignore below threshold pixels
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] is None or stats[i, cv2.CC_STAT_AREA] <= 0:
            raise ValueError(f"Region size is None or less than 0 for label {i}")
        region_size = stats[i, cv2.CC_STAT_AREA]
        if region_size >= min_region_size:
            if stats[i, cv2.CC_STAT_LEFT] is None or stats[i, cv2.CC_STAT_TOP] is None or stats[i, cv2.CC_STAT_WIDTH] is None or stats[i, cv2.CC_STAT_HEIGHT] is None:
                raise ValueError(f"Region coordinates are None for label {i}")
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            try:
              cx, cy = centroids[i]
            except Exception as e:
                raise RuntimeError(f"Error in getting centroid for label {i}: {e}")
            
            # Calculate average difference between pixels in this region
            region_mask = (labels == i)
            try:
              avg_diff = np.mean(pixel_diffs[region_mask])
            except Exception as e:
              raise RuntimeError(f"Error in calculating average difference for label {i}: {e}")
            try:
              max_diff = np.max(pixel_diffs[region_mask])
            except Exception as e:
              raise RuntimeError(f"Error in calculating max difference for label {i}: {e}")
            
            significant_regions.append({
                'id': i,
                'size': region_size,
                'bbox': (x, y, w, h),
                'centroid': (cx, cy),
                'avg_difference': avg_diff,
                'max_difference': max_diff,
                'mask': region_mask
            })
    
    # Sort by region size (largest first)
    significant_regions.sort(key=lambda x: x.get('size', 0), reverse=True)
    
    return {
        'total_regions': len(significant_regions),
        'regions': significant_regions,
        'threshold': threshold,
        'total_different_pixels': np.sum(diff_mask),
        'total_pixels': diff_mask.size,
        'difference_percentage': (np.sum(diff_mask) / diff_mask.size) * 100
    }

def _create_difference_visualization(img1: np.ndarray, img2: np.ndarray, regions_info: dict) -> np.ndarray:
    """
    Create a visualization showing the difference regions overlaid on the original image.
    """

    # Start with a copy of the first image
    vis_img = img1.copy()
    
    # Create a colored overlay for difference regions
    overlay = np.zeros_like(img1)
    
    # Color each region differently
    colors = [
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta  
        (0, 255, 0),      # Green
        (255, 255, 0),    # Cyan
        (255, 0, 0),      # Blue
        (0, 0, 255),      # Red
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (0, 128, 255),    # Light Blue
        (255, 0, 128),    # Pink
        (128, 255, 0),    # Lime
        (255, 255, 255),  # White
    ]
    
    regions = regions_info.get('regions')
    if regions is None:
        raise ValueError("Missing 'regions' key in regions_info")
    
    max_regions = min(12, len(regions))
    
    for i, region in enumerate(regions[:max_regions]):
        color = colors[i % len(colors)]
        mask = region.get('mask')
        if mask is None:
            raise ValueError(f"Missing 'mask' key in region {i}")
        overlay[mask] = color
    
    # Blend the overlay with the original image
    alpha = 0.3  # Transparency
    # result = src1 * 1-alpha + src2 * alpha + gamma
    # No brightness adjustment (0)
    vis_img = cv2.addWeighted(vis_img, 1-alpha, overlay, alpha, 0)
    
    # Draw bounding boxes and labels
    for i, region in enumerate(regions[:max_regions]):
        bbox = region.get('bbox')
        if bbox is None:
            raise ValueError(f"Missing 'bbox' key in region {i}")
        x, y, w, h = bbox
        
        centroid = region.get('centroid')
        if centroid is None:
            raise ValueError(f"Missing 'centroid' key in region {i}")
        cx, cy = centroid
        
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), colors[i % len(colors)], 2)
        
        size = region.get('size')
        if size is None:
            raise ValueError(f"Missing 'size' key in region {i}")
        label = f"R{i+1}: {size}px"
        # Label region by ranking
        cv2.putText(vis_img, label, (int(cx-30), int(cy)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i % len(colors)], 1)
    
    return vis_img


def build_diff_url(diff_path: Path) -> str:
    """
    Build a URL for a difference image file relative to the static mount.
    
    Args:
        diff_path: Path to the difference image file
        
    Returns:
        URL string relative to /static mount (e.g., "/static/diffs/image.png")
        
    Raises:
        ValueError: If diff_path is not a valid Path object
    """
    try:
        # Ensure we have a Path object
        if not isinstance(diff_path, Path):
            diff_path = Path(diff_path)
        
        # Try to get relative path from STATIC_ROOT
        try:
            rel = diff_path.relative_to(STATIC_ROOT)
            # Convert to posix-style path and build URL
            return f"/static/{rel.as_posix()}"
        except ValueError:
            # If outside static root, just return filename (not expected in normal operation)
            return f"/static/{diff_path.name}"
        
    except Exception as e:
        raise ValueError(f"Failed to build diff URL for path {diff_path}: {e}")


# API Endpoints
@app.on_event("startup")
async def on_startup():
    """
    Initialize database on server startup.
    This ensures the database tables are created before the server starts accepting requests.
    """
    try:
        await init_db()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        print("⚠️  Server will start but database operations may fail")


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

# Test with curl -X POST "http://localhost:8000/comparison" -F "file1=@image1.png" -F "file2=@image2.png"
@app.post("/comparison", response_model=ComparisonResponse)
async def create_comparison(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    Compare two images and return similarity scores with a difference visualization.
    
    Args:
        file1: First image file (UploadFile)
        file2: Second image file (UploadFile)
        
    Returns:
        ComparisonResponse with similarity scores and difference image URL
        
    Raises:
        HTTPException: If image processing or database operations fail
    """
    try:
        comp_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        
        print(f"🔄 Processing comparison {comp_id}...")
        img1 = _read_image_to_bgr(file1)
        img2 = _read_image_to_bgr(file2)
        
        img1, img2 = _ensure_same_size(img1, img2)
        print(f"✅ Images processed: {img1.shape}")
        
        # Blue channel comparison
        hist1 = _compute_channel_hist(img1, 0) 
        hist2 = _compute_channel_hist(img2, 0)  
        scores = _compare_histograms_per_method(hist1, hist2)
        
        correlation_score = scores.get('corr')
        chi_square_score = scores.get('chi')
        bhattacharyya_score = scores.get('bha')
        
        if correlation_score is None or chi_square_score is None or bhattacharyya_score is None:
            raise HTTPException(status_code=500, detail="Failed to compute histogram similarity scores")
        
        print(f"✅ Histogram scores computed: corr={correlation_score:.6f}, chi={chi_square_score:.6f}, bha={bhattacharyya_score:.6f}")
        
        # Convert scores to percentages for storage and response (except chi-square which is a distance metric)
        correlation_percent = correlation_score * 100
        bhattacharyya_percent = bhattacharyya_score * 100
        
        print(f"✅ Scores converted: corr={correlation_percent:.2f}%, chi={chi_square_score:.6f} (distance), bha={bhattacharyya_percent:.2f}%")
        
        # Find difference regions and create visualization
        regions_info = _find_difference_regions(img1, img2, threshold=50.0, min_region_size=50)
        diff_img = _create_difference_visualization(img1, img2, regions_info)
        
        diff_filename = f"comparison_{comp_id}.png"
        diff_path = DIFF_DIR / diff_filename
        
        DIFF_DIR.mkdir(parents=True, exist_ok=True)
        
        success = cv2.imwrite(str(diff_path), diff_img)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to save difference image to {diff_path}")
        
        print(f"✅ Difference image saved: {diff_path}")
        
        async with SessionLocal() as session:
            entity = Comparison(
                id=comp_id,
                correlation_score=correlation_percent,
                chi_square_score=chi_square_score,
                bhattacharyya_score=bhattacharyya_percent,
                diff_image_path=str(diff_path),
                created_at=created_at,
            )
            session.add(entity)
            await session.commit()
        
        print(f"✅ Comparison saved to database: {comp_id}")
        
        # Return response
        return ComparisonResponse(
            id=comp_id,
            correlation=correlation_percent,
            chi_square=chi_square_score,
            bhattacharyya=bhattacharyya_percent,
            diff_image_url=build_diff_url(diff_path),
            created_at=created_at,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image comparison failed: {str(e)}")

# Test with curl "http://localhost:8000/comparison/{comp_id}" after posting a comparison
@app.get("/comparison/{comp_id}", response_model=ComparisonResponse)
async def get_comparison(comp_id: str):
    """
    Retrieve a comparison by its ID.
    
    Args:
        comp_id: The unique identifier of the comparison
        
    Returns:
        ComparisonResponse with the comparison data
        
    Raises:
        HTTPException: If comparison is not found
    """
    try:
        async with SessionLocal() as session:
            result = await session.get(Comparison, comp_id)
            if result is None:
                raise HTTPException(status_code=404, detail=f"Comparison with ID {comp_id} not found")
            
            return ComparisonResponse(
                id=result.id,
                correlation=result.correlation_score,  # Stored as percentage
                chi_square=result.chi_square_score,    # Stored as decimal (distance metric)
                bhattacharyya=result.bhattacharyya_score,  # Stored as percentage
                diff_image_url=build_diff_url(Path(result.diff_image_path)),
                created_at=result.created_at,
            )
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Failed to retrieve comparison {comp_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve comparison: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)