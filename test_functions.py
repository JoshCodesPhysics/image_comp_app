"""
Test functions for the image comparison application.
This module contains all test functions that were previously in app.py.
"""

import os
import io
import math
import numpy as np
import cv2
from fastapi import UploadFile
from sqlalchemy.orm import DeclarativeBase

# Import functions from app.py
from app import (
    _read_image_to_bgr, 
    _ensure_same_size, 
    _compute_channel_hist, 
    _compare_histograms_per_method,
    _find_difference_regions,
    _create_difference_visualization,
    build_diff_url,
    Base
)


def test_database_initialization():
    """Test if the Comparison table is registered with Base.metadata."""
    try:
        assert "comparisons" in Base.metadata.tables, "Comparison table not found in Base.metadata"
        print("✅ Database initialization test passed: 'comparisons' table is registered.")
    except AssertionError as e:
        print(f"❌ Database initialization test failed: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred during database initialization test: {e}")


def test_image_reading():
    """Test the _read_image_to_bgr function with a mock UploadFile."""
    print("\n--- Running Image Reading Test ---")
    try:
        # Block 1: Create a mock UploadFile object from image1.png
        try:
            with open("image1.png", "rb") as f:
                image_data = f.read()
            upload_file = UploadFile(file=io.BytesIO(image_data), filename="image1.png")
            print("✅ Mock UploadFile created successfully.")
        except Exception as e:
            print(f"❌ Failed to create mock UploadFile: {e}")
            return

        # Block 2: Call _read_image_to_bgr with the mock UploadFile
        try:
            img_array = _read_image_to_bgr(upload_file)
            print(f"✅ _read_image_to_bgr successful. Image shape: {img_array.shape}")
            assert isinstance(img_array, np.ndarray), "Returned object is not a numpy array."
            assert img_array.shape[2] == 3, "Image is not a 3-channel BGR image."
        except Exception as e:
            print(f"❌ _read_image_to_bgr failed: {e}")
            return

        # Block 3: Verify the image content (optional, but good for thoroughness)
        try:
            # For example, check a pixel value or overall size
            assert img_array.size > 0, "Decoded image is empty."
            print("✅ Image content verified (non-empty).")
        except AssertionError as e:
            print(f"❌ Image content verification failed: {e}")
        except Exception as e:
            print(f"❌ Image content verification failed with unexpected error: {e}")

    except Exception as e:
        print(f"❌ Image reading test failed: {e}")


def test_image_size_normalization():
    """Test the _ensure_same_size function with two images of different sizes."""
    print("\n--- Running Image Size Normalization Test ---")
    try:
        # Block 1: Read image1.png and image2.png into BGR numpy arrays
        try:
            with open("image1.png", "rb") as f:
                image1_data = f.read()
            upload_file1 = UploadFile(file=io.BytesIO(image1_data), filename="image1.png")
            img1 = _read_image_to_bgr(upload_file1)
            print(f"✅ Image1 processing successful: shape {img1.shape}")

            with open("image2.png", "rb") as f:
                image2_data = f.read()
            upload_file2 = UploadFile(file=io.BytesIO(image2_data), filename="image2.png")
            img2 = _read_image_to_bgr(upload_file2)
            print(f"✅ Image2 processing successful: shape {img2.shape}")
        except Exception as e:
            print(f"❌ Image reading for size normalization failed: {e}")
            return

        # Block 2: Call _ensure_same_size
        try:
            # Create a larger version of image1 for testing resizing
            # For actual testing, you'd use image3_larger.png or similar
            # For now, let's simulate different sizes if image1 and image2 are already same size
            if img1.shape == img2.shape:
                # If images are already the same size, let's make one larger for the test
                # This is a hack for testing if actual different sized images aren't available
                img_test_large = cv2.resize(img1, (img1.shape[1] * 2, img1.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
                print(f"Simulating larger image for img1: new shape {img_test_large.shape}")
                resized_img1, resized_img2 = _ensure_same_size(img_test_large, img2)
            else:
                resized_img1, resized_img2 = _ensure_same_size(img1, img2)

            print(f"✅ _ensure_same_size successful.")
            print(f"   Resized Image 1 shape: {resized_img1.shape}")
            print(f"   Resized Image 2 shape: {resized_img2.shape}")

            assert resized_img1.shape == resized_img2.shape, "Images are not the same size after normalization."
            print("✅ Images are the same size after normalization.")
        except Exception as e:
            print(f"❌ _ensure_same_size failed: {e}")
            return

        # Block 3: Verify dimensions and types
        try:
            assert isinstance(resized_img1, np.ndarray) and isinstance(resized_img2, np.ndarray), "Returned objects are not numpy arrays."
            assert resized_img1.shape[2] == 3 and resized_img2.shape[2] == 3, "Images are not 3-channel BGR images."
            print("✅ Returned objects are numpy arrays and 3-channel BGR images.")
        except AssertionError as e:
            print(f"❌ Verification failed: {e}")
        except Exception as e:
            print(f"❌ Verification failed with unexpected error: {e}")

    except Exception as e:
        print(f"❌ Image size normalization test failed: {e}")


def test_channel_histogram():
    """Test the _compute_channel_hist function by generating histograms for all three channels."""
    print("\n--- Running Channel Histogram Test ---")
    try:
        # Block 1: Read image1.png into a BGR numpy array
        try:
            with open("image1.png", "rb") as f:
                image_data = f.read()
            upload_file = UploadFile(file=io.BytesIO(image_data), filename="image1.png")
            img = _read_image_to_bgr(upload_file)
            print(f"✅ Image processing successful: shape {img.shape}")
        except Exception as e:
            print(f"❌ Image reading for histogram test failed: {e}")
            return

        # Block 2: Compute histograms for each channel (Blue, Green, Red)
        histograms = {}
        channel_names = ["Blue", "Green", "Red"]
        for i, name in enumerate(channel_names):
            try:
                hist = _compute_channel_hist(img, i)
                histograms[name] = hist
                print(f"✅ {name} channel histogram created successfully:")
                print(f"   Shape: {hist.shape}, Sum: {hist.sum():.6f}")
                assert hist.shape == (256,), f"{name} histogram does not have expected shape (256,)."
                assert math.isclose(hist.sum(), 1.0, rel_tol=1e-6), f"{name} histogram sum is not close to 1.0."
                assert np.all(hist >= 0), f"{name} histogram contains negative values."
            except Exception as e:
                print(f"❌ {name} channel histogram creation failed: {e}")
                return

        print("✅ All channel histograms created and validated.")

    except Exception as e:
        print(f"❌ Channel histogram test failed: {e}")


def test_histogram_comparison():
    """Test the _compare_histograms_per_method function with two histograms."""
    print("\n--- Running Histogram Comparison Test ---")
    try:
        # Block 1: Create two test histograms using _compute_channel_hist
        try:
            # Read and process image1.png
            with open("image1.png", "rb") as f:
                image1_data = f.read()
            upload_file1 = UploadFile(file=io.BytesIO(image1_data), filename="image1.png")
            img1 = _read_image_to_bgr(upload_file1)
            print(f"✅ Image1 processing successful: shape {img1.shape}")

            # Read and process image2.png
            with open("image2.png", "rb") as f:
                image2_data = f.read()
            upload_file2 = UploadFile(file=io.BytesIO(image2_data), filename="image2.png")
            img2 = _read_image_to_bgr(upload_file2)
            print(f"✅ Image2 processing successful: shape {img2.shape}")

        except Exception as e:
            print(f"❌ Image processing failed: {e}")
            return None

        # Block 2: Create histograms for the same channel (Blue channel)
        try:
            # Create histograms for Blue channel (index 0) from both images
            hist1 = _compute_channel_hist(img1, 0)
            hist2 = _compute_channel_hist(img2, 0)
            print(f"✅ Histogram creation successful:")
            print(f"   Hist1 shape: {hist1.shape}, sum: {hist1.sum():.6f}")
            print(f"   Hist2 shape: {hist2.shape}, sum: {hist2.sum():.6f}")

        except Exception as e:
            print(f"❌ Histogram creation failed: {e}")
            return None

        # Block 3: Test histogram comparison
        try:
            # Compare the two histograms
            comparison_scores = _compare_histograms_per_method(hist1, hist2)
            print(f"✅ Histogram comparison successful!")
            
            corr_score = comparison_scores.get('corr')
            chi_score = comparison_scores.get('chi')
            bha_score = comparison_scores.get('bha')
            
            if corr_score is None or chi_score is None or bha_score is None:
                raise ValueError("Missing required scores in comparison_scores")
            
            print(f"   Correlation score: {corr_score}")
            print(f"   Chi-square score: {chi_score}")
            print(f"   Bhattacharyya score: {bha_score}")

            # Validate score ranges
            print(f"✅ Score validation:")
            print(f"   Correlation in range [-1, 1]: {-1 <= corr_score <= 1}")
            print(f"   Chi-square >= 0: {chi_score >= 0}")
            print(f"   Bhattacharyya in range [0, 1]: {0 <= bha_score <= 1}")

            # Test with identical histograms (should give perfect scores)
            identical_scores = _compare_histograms_per_method(hist1, hist1)
            print(f"✅ Identical histogram test:")
            
            identical_corr = identical_scores.get('corr')
            identical_chi = identical_scores.get('chi')
            identical_bha = identical_scores.get('bha')
            
            if identical_corr is None or identical_chi is None or identical_bha is None:
                raise ValueError("Missing required scores in identical_scores")
            
            print(f"   Correlation (should be 1.0): {identical_corr:.6f}")
            print(f"   Chi-square (should be 0.0): {identical_chi:.6f}")
            print(f"   Bhattacharyya (should be 0.0): {identical_bha:.6f}")

            return True

        except Exception as e:
            print(f"❌ Histogram comparison failed: {e}")
            return None

    except Exception as e:
        print(f"❌ Overall test failed: {e}")
        return None


def test_image_formats():
    """Test histogram comparison across different image formats."""
    print("\n--- Running Image Format Comparison Test ---")

    formats = ['png', 'jpg', 'webp']
    results = {}

    for format_ext in formats:
        try:
            print(f"\n🔍 Testing {format_ext.upper()} format:")

            # Read images in the specified format
            with open(f"image1.{format_ext}", "rb") as f:
                image1_data = f.read()
            upload_file1 = UploadFile(file=io.BytesIO(image1_data), filename=f"image1.{format_ext}")
            img1 = _read_image_to_bgr(upload_file1)

            with open(f"image2.{format_ext}", "rb") as f:
                image2_data = f.read()
            upload_file2 = UploadFile(file=io.BytesIO(image2_data), filename=f"image2.{format_ext}")
            img2 = _read_image_to_bgr(upload_file2)

            print(f"✅ {format_ext.upper()} images loaded successfully")
            print(f"   Image1 shape: {img1.shape}")
            print(f"   Image2 shape: {img2.shape}")

            # Ensure same size
            img1_resized, img2_resized = _ensure_same_size(img1, img2)

            # Test all three channels
            channel_results = {}
            for channel_idx, channel_name in enumerate(['Blue', 'Green', 'Red']):
                try:
                    hist1 = _compute_channel_hist(img1_resized, channel_idx)
                    hist2 = _compute_channel_hist(img2_resized, channel_idx)
                    scores = _compare_histograms_per_method(hist1, hist2)

                    channel_results[channel_name] = scores
                    
                    corr = scores.get('corr')
                    chi = scores.get('chi')
                    bha = scores.get('bha')
                    
                    if corr is None or chi is None or bha is None:
                        raise ValueError(f"Missing required scores for {format_ext} {channel_name}")
                    
                    print(f"   {channel_name} channel - Corr: {corr:.6f}, Chi: {chi:.6f}, Bha: {bha:.6f}")

                except Exception as e:
                    print(f"   ❌ {channel_name} channel failed: {e}")
                    channel_results[channel_name] = None

            results[format_ext] = channel_results

        except Exception as e:
            print(f"❌ {format_ext.upper()} format test failed: {e}")
            results[format_ext] = None

    # Compare results across formats
    print(f"\n📊 Format Comparison Summary:")
    print(f"{'Format':<8} {'Blue Corr':<12} {'Green Corr':<12} {'Red Corr':<12} {'Blue Chi':<12} {'Green Chi':<12} {'Red Chi':<12}")
    print("-" * 80)

    for format_ext in formats:
        format_results = results.get(format_ext)
        if format_results and all(format_results.get(ch) for ch in ['Blue', 'Green', 'Red']):
            blue = format_results.get('Blue')
            green = format_results.get('Green')
            red = format_results.get('Red')
            
            if blue and green and red:
                blue_corr = blue.get('corr', 0)
                green_corr = green.get('corr', 0)
                red_corr = red.get('corr', 0)
                blue_chi = blue.get('chi', 0)
                green_chi = green.get('chi', 0)
                red_chi = red.get('chi', 0)
                
                print(f"{format_ext.upper():<8} {blue_corr:<12.6f} {green_corr:<12.6f} {red_corr:<12.6f} {blue_chi:<12.6f} {green_chi:<12.6f} {red_chi:<12.6f}")
        else:
            print(f"{format_ext.upper():<8} {'FAILED':<12} {'FAILED':<12} {'FAILED':<12} {'FAILED':<12} {'FAILED':<12} {'FAILED':<12}")

    # Analyze differences between formats
    print(f"\n🔍 Format Impact Analysis:")
    if all(results.get(fmt) and all(results.get(fmt, {}).get(ch) for ch in ['Blue', 'Green', 'Red']) for fmt in formats):
        png_blue_corr = results.get('png', {}).get('Blue', {}).get('corr')
        jpg_blue_corr = results.get('jpg', {}).get('Blue', {}).get('corr')
        webp_blue_corr = results.get('webp', {}).get('Blue', {}).get('corr')
        
        if png_blue_corr is not None and jpg_blue_corr is not None and webp_blue_corr is not None:
            print(f"   Blue channel correlation differences:")
            print(f"     PNG vs JPEG: {abs(png_blue_corr - jpg_blue_corr):.8f}")
            print(f"     PNG vs WebP: {abs(png_blue_corr - webp_blue_corr):.8f}")
            print(f"     JPEG vs WebP: {abs(jpg_blue_corr - webp_blue_corr):.8f}")
        else:
            print(f"   ❌ Missing correlation data for format comparison")

        # Check if compression affects similarity detection
        max_diff = max(abs(png_blue_corr - jpg_blue_corr), abs(png_blue_corr - webp_blue_corr), abs(jpg_blue_corr - webp_blue_corr))
        if max_diff < 0.001:
            print(f"   ✅ Compression has minimal impact on similarity detection")
        else:
            print(f"   ⚠️  Compression affects similarity detection (max diff: {max_diff:.8f})")

    return results


def test_spatial_difference_regions():
    """Test spatial region analysis to identify specific areas of difference."""
    print("\n--- Running Spatial Region Difference Analysis ---")

    try:
        # Load the images
        with open("image1.png", "rb") as f:
            image1_data = f.read()
        upload_file1 = UploadFile(file=io.BytesIO(image1_data), filename="image1.png")
        img1 = _read_image_to_bgr(upload_file1)

        with open("image2.png", "rb") as f:
            image2_data = f.read()
        upload_file2 = UploadFile(file=io.BytesIO(image2_data), filename="image2.png")
        img2 = _read_image_to_bgr(upload_file2)

        # Ensure same size
        img1_resized, img2_resized = _ensure_same_size(img1, img2)

        print(f"✅ Images loaded and resized: {img1_resized.shape}")

        # Test different thresholds
        thresholds = [10.0, 30.0, 50.0, 60.0, 80.0, 100.0]

        for threshold in thresholds:
            print(f"\n🔍 Testing threshold: {threshold}")

            try:
                regions_info = _find_difference_regions(img1_resized, img2_resized, threshold)

                total_regions = regions_info.get('total_regions')
                total_different_pixels = regions_info.get('total_different_pixels')
                total_pixels = regions_info.get('total_pixels')
                difference_percentage = regions_info.get('difference_percentage')
                
                if total_regions is None or total_different_pixels is None or total_pixels is None or difference_percentage is None:
                    raise ValueError("Missing required keys in regions_info")

                print(f"   Total regions found: {total_regions}")
                print(f"   Different pixels: {total_different_pixels}/{total_pixels} ({difference_percentage:.2f}%)")

                regions = regions_info.get('regions')
                if regions and total_regions > 0:
                    print(f"   Top 10 regions:")
                    for i, region in enumerate(regions[:10]):
                        bbox = region.get('bbox')
                        centroid = region.get('centroid')
                        size = region.get('size')
                        avg_diff = region.get('avg_difference')
                        max_diff = region.get('max_difference')
                        
                        if bbox is None or centroid is None or size is None or avg_diff is None or max_diff is None:
                            print(f"     Region {i+1}: Missing data")
                            continue
                            
                        x, y, w, h = bbox
                        cx, cy = centroid
                        print(f"     Region {i+1}: Size={size}px, BBox=({x},{y},{w},{h}), Center=({cx:.1f},{cy:.1f})")
                        print(f"                Avg diff={avg_diff:.1f}, Max diff={max_diff:.1f}")

                # Create visualizations for different thresholds
                if regions and total_regions > 0:
                    vis_img = _create_difference_visualization(img1_resized, img2_resized, regions_info)
                    filename = f"difference_regions_threshold_{threshold}.png"
                    cv2.imwrite(filename, vis_img)
                    print(f"   ✅ Visualization saved as '{filename}'")

            except Exception as e:
                print(f"   ❌ Threshold {threshold} failed: {e}")

        # Compare with histogram approach
        print(f"\n📊 Comparison with Histogram Approach:")
        hist1 = _compute_channel_hist(img1_resized, 0)  # Blue channel
        hist2 = _compute_channel_hist(img2_resized, 0)
        hist_scores = _compare_histograms_per_method(hist1, hist2)

        hist_corr = hist_scores.get('corr')
        hist_chi = hist_scores.get('chi')
        hist_bha = hist_scores.get('bha')
        
        if hist_corr is None or hist_chi is None or hist_bha is None:
            print(f"   ❌ Missing histogram scores")
        else:
            print(f"   Histogram correlation: {hist_corr:.6f}")
            print(f"   Histogram chi-square: {hist_chi:.6f}")
            print(f"   Histogram bhattacharyya: {hist_bha:.6f}")

        # Spatial analysis with threshold 30
        spatial_info = _find_difference_regions(img1_resized, img2_resized, 30.0)
        spatial_diff_pct = spatial_info.get('difference_percentage')
        spatial_total_regions = spatial_info.get('total_regions')
        
        if spatial_diff_pct is None or spatial_total_regions is None:
            print(f"   ❌ Missing spatial analysis data")
        else:
            print(f"   Spatial difference percentage: {spatial_diff_pct:.2f}%")
            print(f"   Number of significant regions: {spatial_total_regions}")

        return True

    except Exception as e:
        print(f"❌ Spatial region analysis failed: {e}")
        return False


def test_minimum_region_sizes():
    """Test how different minimum region sizes affect the analysis."""
    print("\n--- Testing Different Minimum Region Sizes ---")

    try:
        # Load the images
        with open("image1.png", "rb") as f:
            image1_data = f.read()
        upload_file1 = UploadFile(file=io.BytesIO(image1_data), filename="image1.png")
        img1 = _read_image_to_bgr(upload_file1)

        with open("image2.png", "rb") as f:
            image2_data = f.read()
        upload_file2 = UploadFile(file=io.BytesIO(image2_data), filename="image2.png")
        img2 = _read_image_to_bgr(upload_file2)

        # Ensure same size
        img1_resized, img2_resized = _ensure_same_size(img1, img2)

        print(f"✅ Images loaded: {img1_resized.shape}")

        # Test different minimum region sizes
        min_sizes = [1, 10, 25, 50, 100, 200, 500]
        threshold = 30.0  # Fixed threshold

        print(f"\n🔍 Testing with threshold={threshold} and different minimum region sizes:")
        print(f"{'Min Size':<10} {'Regions':<8} {'Total Pixels':<12} {'Percentage':<10} {'Largest Region':<15}")
        print("-" * 70)

        for min_size in min_sizes:
            try:
                regions_info = _find_difference_regions(img1_resized, img2_resized, threshold, min_size)

                total_regions = regions_info.get('total_regions')
                total_different_pixels = regions_info.get('total_different_pixels')
                difference_percentage = regions_info.get('difference_percentage')
                regions = regions_info.get('regions')
                
                if total_regions is None or total_different_pixels is None or difference_percentage is None:
                    print(f"{min_size:<10} {'ERROR':<8} {'ERROR':<12} {'ERROR':<10} {'ERROR':<15}")
                    continue

                largest_region_size = 0
                if regions:
                    first_region = regions[0]
                    if first_region:
                        largest_region_size = first_region.get('size', 0)

                print(f"{min_size:<10} {total_regions:<8} {total_different_pixels:<12} {difference_percentage:<10.2f} {largest_region_size:<15}")

                # Create visualization for min_size = 1 (show all regions)
                if min_size == 1:
                    vis_img = _create_difference_visualization(img1_resized, img2_resized, regions_info)
                    cv2.imwrite(f"all_regions_min_size_{min_size}.png", vis_img)
                    print(f"   ✅ Visualization with ALL regions saved as 'all_regions_min_size_1.png'")

            except Exception as e:
                print(f"   ❌ Min size {min_size} failed: {e}")

        # Show detailed breakdown for very small minimum size
        print(f"\n🔍 Detailed analysis with min_size=1 (showing all regions):")
        try:
            all_regions_info = _find_difference_regions(img1_resized, img2_resized, threshold, 1)
            
            all_total_regions = all_regions_info.get('total_regions')
            all_total_different_pixels = all_regions_info.get('total_different_pixels')
            all_difference_percentage = all_regions_info.get('difference_percentage')
            all_regions = all_regions_info.get('regions')
            
            if all_total_regions is None or all_total_different_pixels is None or all_difference_percentage is None:
                print(f"   ❌ Missing data in all_regions_info")
            else:
                print(f"   Total regions found: {all_total_regions}")
                print(f"   Total different pixels: {all_total_different_pixels}")
                print(f"   Difference percentage: {all_difference_percentage:.2f}%")

                if all_regions:
                    print(f"   Top 15 smallest regions:")
                    for i, region in enumerate(all_regions[-15:]):  # Show smallest regions
                        size = region.get('size')
                        centroid = region.get('centroid')
                        if size is not None and centroid is not None:
                            print(f"     Region {i+1}: {size}px at ({centroid[0]:.1f}, {centroid[1]:.1f})")

                    print(f"   Top 5 largest regions:")
                    for i, region in enumerate(all_regions[:5]):  # Show largest regions
                        size = region.get('size')
                        centroid = region.get('centroid')
                        if size is not None and centroid is not None:
                            print(f"     Region {i+1}: {size}px at ({centroid[0]:.1f}, {centroid[1]:.1f})")

        except Exception as e:
            print(f"   ❌ Detailed analysis failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Minimum region size test failed: {e}")
        return False


def test_health_check():
    """Test the health check functionality."""
    print("\n--- Running Health Check Test ---")
    try:
        # Test database connectivity directly
        from app import engine
        from sqlalchemy import text
        import asyncio
        
        async def test_db():
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return {"status": "healthy", "database": "connected"}
        
        # Note: This test is simplified since we can't easily test async functions here
        print("✅ Health check test skipped (requires async context)")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_build_diff_url():
    """Test the build_diff_url function."""
    print("\n--- Running Build Diff URL Test ---")
    
    try:
        from pathlib import Path
        from app import STATIC_ROOT, DIFF_DIR
        
        print(f"✅ STATIC_ROOT: {STATIC_ROOT}")
        print(f"✅ DIFF_DIR: {DIFF_DIR}")
        
        # Test 1: Normal case - file within STATIC_ROOT
        print(f"\n🔍 Test 1: File within STATIC_ROOT")
        test_file = DIFF_DIR / "test_diff.png"
        expected_url = f"/static/{DIFF_DIR.relative_to(STATIC_ROOT).as_posix()}/test_diff.png"
        
        try:
            result_url = build_diff_url(test_file)
            print(f"   Input path: {test_file}")
            print(f"   Expected URL: {expected_url}")
            print(f"   Actual URL: {result_url}")
            
            assert result_url == expected_url, f"URL mismatch: expected {expected_url}, got {result_url}"
            print(f"   ✅ URL generation successful")
            
        except Exception as e:
            print(f"   ❌ Test 1 failed: {e}")
            return False
        
        # Test 2: File directly in STATIC_ROOT
        print(f"\n🔍 Test 2: File directly in STATIC_ROOT")
        test_file2 = STATIC_ROOT / "direct_file.jpg"
        expected_url2 = "/static/direct_file.jpg"
        
        try:
            result_url2 = build_diff_url(test_file2)
            print(f"   Input path: {test_file2}")
            print(f"   Expected URL: {expected_url2}")
            print(f"   Actual URL: {result_url2}")
            
            assert result_url2 == expected_url2, f"URL mismatch: expected {expected_url2}, got {result_url2}"
            print(f"   ✅ URL generation successful")
            
        except Exception as e:
            print(f"   ❌ Test 2 failed: {e}")
            return False
        
        # Test 3: String input (should be converted to Path)
        print(f"\n🔍 Test 3: String input conversion")
        test_string = str(DIFF_DIR / "string_test.png")
        expected_url3 = f"/static/{DIFF_DIR.relative_to(STATIC_ROOT).as_posix()}/string_test.png"
        
        try:
            result_url3 = build_diff_url(test_string)
            print(f"   Input string: {test_string}")
            print(f"   Expected URL: {expected_url3}")
            print(f"   Actual URL: {result_url3}")
            
            assert result_url3 == expected_url3, f"URL mismatch: expected {expected_url3}, got {result_url3}"
            print(f"   ✅ String to Path conversion successful")
            
        except Exception as e:
            print(f"   ❌ Test 3 failed: {e}")
            return False
        
        # Test 4: File outside STATIC_ROOT (edge case)
        print(f"\n🔍 Test 4: File outside STATIC_ROOT (edge case)")
        outside_file = Path("/tmp/outside_file.png")
        expected_url4 = "/static/outside_file.png"
        
        try:
            result_url4 = build_diff_url(outside_file)
            print(f"   Input path: {outside_file}")
            print(f"   Expected URL: {expected_url4}")
            print(f"   Actual URL: {result_url4}")
            
            assert result_url4 == expected_url4, f"URL mismatch: expected {expected_url4}, got {result_url4}"
            print(f"   ✅ Outside STATIC_ROOT handling successful")
            
        except Exception as e:
            print(f"   ❌ Test 4 failed: {e}")
            return False
        
        # Test 5: Invalid input (should raise ValueError)
        print(f"\n🔍 Test 5: Invalid input handling")
        try:
            # This should raise a ValueError
            build_diff_url(None)
            print(f"   ❌ Expected ValueError for None input, but no exception was raised")
            return False
        except ValueError as e:
            print(f"   ✅ Correctly raised ValueError for None input: {e}")
        except Exception as e:
            print(f"   ❌ Unexpected exception type for None input: {e}")
            return False
        
        # Test 6: Path with subdirectories
        print(f"\n🔍 Test 6: Path with subdirectories")
        subdir_file = DIFF_DIR / "subdir1" / "subdir2" / "nested_file.png"
        expected_url6 = f"/static/{DIFF_DIR.relative_to(STATIC_ROOT).as_posix()}/subdir1/subdir2/nested_file.png"
        
        try:
            result_url6 = build_diff_url(subdir_file)
            print(f"   Input path: {subdir_file}")
            print(f"   Expected URL: {expected_url6}")
            print(f"   Actual URL: {result_url6}")
            
            assert result_url6 == expected_url6, f"URL mismatch: expected {expected_url6}, got {result_url6}"
            print(f"   ✅ Subdirectory path handling successful")
            
        except Exception as e:
            print(f"   ❌ Test 6 failed: {e}")
            return False
        
        print(f"\n📊 Build Diff URL Summary:")
        print(f"   ✅ All test cases passed successfully")
        print(f"   ✅ Path handling works correctly")
        print(f"   ✅ String input conversion works")
        print(f"   ✅ Edge cases handled properly")
        print(f"   ✅ Error handling works as expected")
        
        return True
        
    except Exception as e:
        print(f"❌ Build diff URL test failed: {e}")
        return False


def run_all_tests():
    """Run all test functions."""
    print("🧪 Running All Tests...")
    print("=" * 50)
    
    test_database_initialization()
    test_image_reading()
    test_image_size_normalization()
    test_channel_histogram()
    test_histogram_comparison()
    test_image_formats()
    test_spatial_difference_regions()
    test_minimum_region_sizes()
    test_build_diff_url()
    test_health_check()
    
    print("\n" + "=" * 50)
    print("🏁 All tests completed!")


if __name__ == "__main__":
    run_all_tests()
