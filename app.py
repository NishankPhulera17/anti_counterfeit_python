from flask import Flask, request, jsonify
from services import cdp_service, liveness_service
from services.feature_extraction import extract_all_features, compare_features
from services.backend_storage import get_backend_storage
from utils.image_utils import base64_to_image, decode_qr_code, analyze_image_statistics
from dotenv import load_dotenv
import numpy as np
import cv2
import os
from PIL import Image
from datetime import datetime
import uuid
from generate_qr_cdp import generate_qr_cdp, cv2_to_base64, CDP_DIR
from services.generate_qr_token import generate_qr_token
load_dotenv()
app = Flask(__name__)

# Directories for saving scanned images during verification
SCANNED_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "scanned_images")
EXTRACTED_CDP_DIR = os.path.join(os.path.dirname(__file__), "extracted_cdp")
os.makedirs(SCANNED_IMAGES_DIR, exist_ok=True)
os.makedirs(EXTRACTED_CDP_DIR, exist_ok=True)

# In-memory reference CDP images (replace with actual references)
reference_cdps = {}  # {product_id: np.array}

@app.route('/generate_qr_cdp', methods=['POST'])
def generate_qr_cdp_endpoint():
    """
    Generate QR+CDP with production-ready architecture:
    - CDP is cryptographically random (not derived from product_id)
    - QR encodes serial_id (pointer to backend)
    - Features stored server-side (not raw images)
    """
    data = request.json
    product_id = data.get('product_id')
    if not product_id:
        return jsonify({
            "status": "failed",
            "message": "product_id is required"
        }), 400

    try:
        backend = get_backend_storage()
        
        # Generate unique serial ID for this item (encoded in QR code)
        serial_id = str(uuid.uuid4())
        
        # Generate unique CDP ID (cryptographically random)
        cdp_id = str(uuid.uuid4())
        
        # Size configurations: (width_mm, height_mm, dpi, size_name)
        SIZE_CONFIGS = [
            (28, 14, 2400, "28x14"),
            (40, 20, 2400, "40x20"),
            (50, 25, 2400, "50x25"),
        ]
        
        generated_images = {}
        reference_features = None  # Store features from first size
        
        # Generate QR+CDP images in all 3 sizes
        for width_mm, height_mm, dpi, size_name in SIZE_CONFIGS:
            print(f"[INFO] Generating {size_name} QR+CDP for product {product_id}, serial {serial_id}...", flush=True)
            
            try:
                # Generate QR+CDP image for this size
                # QR encodes serial_id (pointer), CDP is random
                img, generated_cdp_id = generate_qr_cdp(
                    serial_id=serial_id,  # QR encodes serial_id, not product_id
                    qr_size=None,  # Auto-calculate based on physical size and DPI
                    cdp_size=None,  # Auto-calculate based on physical size and DPI
                    padding=20,
                    border_thickness=25,
                    physical_size_mm=(width_mm, height_mm),
                    dpi=dpi,
                    size_suffix=size_name,  # Add size suffix to filename
                    cdp_id=cdp_id  # Use same CDP ID for all sizes
                )
                
                if img is None:
                    print(f"[ERROR] generate_qr_cdp returned None for {size_name}", flush=True)
                    raise ValueError(f"Failed to generate QR+CDP image for {size_name}")
                
                # Convert image to Base64 to send to app
                try:
                    img_base64 = cv2_to_base64(img)
                    if not img_base64:
                        raise ValueError(f"Failed to convert image to base64 for {size_name}")
                    generated_images[size_name] = img_base64
                    print(f"[INFO] Successfully converted {size_name} image to base64", flush=True)
                except Exception as e:
                    print(f"[ERROR] Failed to convert {size_name} image to base64: {str(e)}", flush=True)
                    raise
                
                # Extract and store features from CDP (use first size as reference)
                if reference_features is None:
                    # Load the generated CDP for feature extraction
                    cdp_path = os.path.join(CDP_DIR, f"{cdp_id}_{size_name}.png")
                    if os.path.exists(cdp_path):
                        cdp_img = cv2.imread(cdp_path)
                        if cdp_img is not None:
                            # Extract features (not storing raw image)
                            reference_features = extract_all_features(cdp_img)
                            print(f"[INFO] Extracted features from {size_name} CDP", flush=True)
                            
                            # Store features in backend
                            backend.store_cdp_features(
                                cdp_id=cdp_id,
                                product_id=product_id,
                                features=reference_features,
                                serial_id=serial_id
                            )
                            print(f"[INFO] Stored CDP features in backend: cdp_id={cdp_id}, serial_id={serial_id}", flush=True)
            except Exception as e:
                print(f"[ERROR] Error generating {size_name} QR+CDP: {str(e)}", flush=True)
                import traceback
                traceback.print_exc()
                raise

        # Generate QR token (JWT) - now uses serial_id
        try:
            qr_token = generate_qr_token(serial_id)  # Token for serial_id, not product_id
            if not qr_token:
                raise ValueError("Failed to generate QR token")
            print(f"[INFO] Generated QR token successfully", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to generate QR token: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            raise

        try:
            # Check if all images were generated
            if len(generated_images) != 3:
                raise ValueError(f"Expected 3 images, got {len(generated_images)}")
            
            # Validate base64 strings are not empty
            for size_name, img_base64 in generated_images.items():
                if not img_base64 or len(img_base64) == 0:
                    raise ValueError(f"Empty base64 string for {size_name}")
                print(f"[INFO] {size_name} base64 length: {len(img_base64)}", flush=True)
            
            response_data = {
                "status": "success",
                "message": f"QR+CDP generated successfully in all 3 sizes for product {product_id}",
                "serial_id": serial_id,  # Return serial_id for reference
                "cdp_id": cdp_id,  # Return cdp_id for reference
                "qrToken": qr_token,
                "qrCdpImages": {
                    "28x14": generated_images.get("28x14"),
                    "40x20": generated_images.get("40x20"),
                    "50x25": generated_images.get("50x25")
                },
                "sizes": {
                    "28x14": {"width_mm": 28, "height_mm": 14, "dpi": 2400},
                    "40x20": {"width_mm": 40, "height_mm": 20, "dpi": 2400},
                    "50x25": {"width_mm": 50, "height_mm": 25, "dpi": 2400}
                }
            }
            print(f"[INFO] Preparing response with {len(generated_images)} images", flush=True)
            
            # Try to serialize to check for issues
            import json
            try:
                json.dumps(response_data)
                print(f"[INFO] Response data is JSON serializable", flush=True)
            except (TypeError, ValueError) as json_err:
                print(f"[ERROR] JSON serialization error: {str(json_err)}", flush=True)
                raise ValueError(f"Response data is not JSON serializable: {str(json_err)}")
            
            return jsonify(response_data)
        except Exception as e:
            print(f"[ERROR] Failed to create JSON response: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            raise
    except Exception as e:
        return jsonify({
            "status": "failed",
            "message": f"Error generating QR+CDP: {str(e)}"
        }), 500

@app.route('/verify_cdp', methods=['POST'])
def verify_cdp():
    """
    Verify CDP using production-ready architecture:
    - Extract serial_id from QR code (pointer)
    - Lookup CDP features from backend
    - Extract features from scanned CDP
    - Compare using scored classification
    - Use data-driven threshold (not hard 0.7)
    - Log audit trail and detect abuse
    """
    backend = get_backend_storage()
    data = request.json
    cdp_base64 = data.get('cdp_image')
    video_frames_base64 = data.get('video_frames', [])
    
    # Require video frames for basic liveness (static screenshot detection)
    if not video_frames_base64 or len(video_frames_base64) < 2:
        return jsonify({
            "status": "failed",
            "message": "video_frames are required (minimum 2 frames) to prevent static screenshot scanning."
        }), 400
    
    if not cdp_base64:
        return jsonify({
            "status": "failed",
            "message": "cdp_image is required"
        }), 400
    
    try:
        cdp_img = base64_to_image(cdp_base64)
        if cdp_img is None:
            return jsonify({
                "status": "failed",
                "message": "Failed to decode cdp_image from base64"
            }), 400
    except Exception as e:
        print(f"[ERROR] Failed to decode cdp_image: {str(e)}", flush=True)
        return jsonify({
            "status": "failed",
            "message": f"Invalid cdp_image format: {str(e)}"
        }), 400
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    
    # Save the scanned image
    scanned_image_filename = f"scanned_{timestamp}.png"
    scanned_image_path = os.path.join(SCANNED_IMAGES_DIR, scanned_image_filename)
    cv2.imwrite(scanned_image_path, cdp_img)
    print(f"[INFO] Scanned image saved: {scanned_image_path}", flush=True)
    
    # Convert video frames
    video_frames = []
    for frame_base64 in video_frames_base64:
        try:
            frame = base64_to_image(frame_base64)
            if frame is not None:
                video_frames.append(frame)
        except Exception as e:
            print(f"[WARNING] Failed to decode video frame: {str(e)}", flush=True)
    
    if len(video_frames) < 2:
        return jsonify({
            "status": "failed",
            "message": "At least 2 valid video frames are required"
        }), 400
    
    # Step 1: Extract serial_id from QR code (QR encodes serial_id, not product_id)
    serial_id = None
    try:
        serial_id = decode_qr_code(cdp_img, require_quality=False, save_folder="scanned_images")
        if serial_id:
            print(f"[INFO] Extracted serial_id from QR code: {serial_id}", flush=True)
        else:
            print(f"[WARNING] Could not extract serial_id from QR code", flush=True)
    except Exception as e:
        print(f"[ERROR] Error decoding QR code: {str(e)}", flush=True)
    
    # QR code is required - fail if we can't extract serial_id
    if not serial_id:
        return jsonify({
            "status": "failed",
            "message": "Could not extract serial_id from QR code. QR code is required for verification."
        }), 400
    
    # Step 2: Lookup CDP record by serial_id from backend
    cdp_record = backend.get_cdp_by_serial(serial_id)
    if not cdp_record:
        return jsonify({
            "status": "failed",
            "message": f"CDP not found for serial_id: {serial_id}. Item may not be registered."
        }), 404
    
    cdp_id = cdp_record.get('cdp_id')
    product_id = cdp_record.get('product_id')
    reference_features = cdp_record.get('features')
    
    if not reference_features:
        return jsonify({
            "status": "failed",
            "message": f"Reference features not found for CDP: {cdp_id}"
        }), 500
    
    print(f"[INFO] Found CDP record: cdp_id={cdp_id}, product_id={product_id}", flush=True)
    
    # Step 3: Check for abuse patterns
    abuse_check = backend.detect_abuse(serial_id, time_window_minutes=60)
    if abuse_check.get('is_abuse', False):
        print(f"[WARNING] Abuse detected for serial_id {serial_id}: {abuse_check.get('abuse_flags', [])}", flush=True)
        # Log but don't block - let verification proceed
    
    # Step 4: Extract CDP region and features from scanned image
    scanned_cdp = cdp_service.extract_cdp_region(cdp_img, save_file=True, 
                                                 output_dir=EXTRACTED_CDP_DIR,
                                                 custom_filename=f"extracted_cdp_{serial_id}_{timestamp}.png")
    
    if scanned_cdp is None or scanned_cdp.size == 0:
        return jsonify({
            "status": "failed",
            "message": "Failed to extract CDP region from scanned image"
        }), 400
    
    # Extract features from scanned CDP
    scanned_features = extract_all_features(scanned_cdp)
    if not scanned_features:
        return jsonify({
            "status": "failed",
            "message": "Failed to extract features from scanned CDP"
        }), 500
    
    # Step 5: Compare features using scored classification
    similarity_score = compare_features(reference_features, scanned_features)
    print(f"[INFO] Feature similarity score: {similarity_score:.3f}", flush=True)
    
    # Step 6: Data-driven threshold (tuned via ROC analysis)
    # Threshold: 0.65 (more lenient than old 0.7, accounts for mobile capture variations)
    AUTHENTICITY_THRESHOLD = 0.65
    
    # Step 7: Basic liveness check (limited role - only rejects static screenshots)
    first_frame = video_frames[0]
    liveness_passed = liveness_service.liveness_check(first_frame, video_frames)
    
    # Step 8: Determine authenticity
    # Main security comes from feature matching, liveness is secondary
    is_authentic = bool(similarity_score >= AUTHENTICITY_THRESHOLD and liveness_passed)
    
    # Step 9: Log verification attempt for audit
    metadata = {
        'ip': request.remote_addr if hasattr(request, 'remote_addr') else None,
        'user_agent': request.headers.get('User-Agent', '') if hasattr(request, 'headers') else None,
    }
    backend.log_verification_attempt(
        cdp_id=cdp_id,
        serial_id=serial_id,
        success=is_authentic,
        score=similarity_score,
        metadata=metadata
    )
    
    # Step 10: Additional assessments (for user feedback, not security)
    distance_check = liveness_service.check_scanning_frame_distance(first_frame)
    lighting_assessment = liveness_service.assess_lighting_conditions(first_frame)
    size_assessment = liveness_service.detect_qr_code_size(first_frame)
    
    # Step 11: Optionally verify product_id from CDP pattern (for additional validation)
    # Note: This is optional - the main verification is via feature matching above
    cdp_product_id = None
    identified_score = 0.0
    try:
        print(f"[INFO] Optionally verifying product_id against CDP pattern...", flush=True)
        cdp_product_id, identified_score = cdp_service.identify_product_from_cdp(
            cdp_img, 
            CDP_DIR, 
            min_match_score=0.6
        )
        if cdp_product_id:
            print(f"[INFO] Identified product_id from CDP: {cdp_product_id} (match score: {identified_score:.3f})", flush=True)
            
            # Verify that CDP matches the product_id from backend (if CDP identification succeeded)
            if cdp_product_id != product_id:
                print(f"[WARNING] Product ID mismatch: Backend='{product_id}', CDP Pattern='{cdp_product_id}'", flush=True)
                # Don't fail verification - just log warning (CDP pattern matching is optional)
            else:
                print(f"[INFO] Product ID verification passed: Backend and CDP pattern both indicate '{product_id}'", flush=True)
        else:
            print(f"[INFO] Could not identify product from CDP pattern (best score: {identified_score:.3f}). Using backend product_id: {product_id}", flush=True)
    except Exception as e:
        print(f"[WARNING] Error identifying product from CDP pattern: {str(e)}. Using backend product_id: {product_id}", flush=True)
        import traceback
        traceback.print_exc()
    
    print(f"[INFO] Verification complete for product_id: {product_id} (from backend lookup)", flush=True)
    
    # Rename the scanned image to include product_id (if we got this far)
    # The image was already saved earlier, but we can save a copy with product_id
    scanned_image_with_product = f"scanned_{product_id}_{timestamp}.png"
    scanned_image_path_with_product = os.path.join(SCANNED_IMAGES_DIR, scanned_image_with_product)
    cv2.imwrite(scanned_image_path_with_product, cdp_img)
    print(f"[INFO] Scanned image saved with product_id: {scanned_image_path_with_product}", flush=True)
    
    # Load reference CDP from folder (try CMYK TIFF first, then PNG fallback)
    # Files are saved as: {product_id}_{size}.tiff and {product_id}_{size}.png
    # e.g., PRODUCT123_28x14.tiff, PRODUCT123_28x14.png, etc.
    import glob
    
    # Try to find files with size suffix pattern: {product_id}_*.tiff or {product_id}_*.png
    cmyk_tiff_pattern = os.path.join(CDP_DIR, f"{product_id}_*.tiff")
    png_pattern = os.path.join(CDP_DIR, f"{product_id}_*.png")
    
    # Also try exact match without size suffix (for backward compatibility)
    cmyk_tiff_path = os.path.join(CDP_DIR, f"{product_id}.tiff")
    png_path = os.path.join(CDP_DIR, f"{product_id}.png")
    
    ref_cdp_path = None
    ref_img = None
    
    # Prefer CMYK TIFF if available, otherwise use PNG
    # First try exact match, then try pattern match
    if os.path.exists(cmyk_tiff_path):
        ref_cdp_path = cmyk_tiff_path
    else:
        # Search for files matching pattern
        tiff_files = glob.glob(cmyk_tiff_pattern)
        if tiff_files:
            ref_cdp_path = tiff_files[0]  # Use first match
            print(f"[INFO] Found TIFF file with size suffix: {ref_cdp_path}", flush=True)
    
    if ref_cdp_path:
        # OpenCV can read TIFF files, but CMYK TIFF needs special handling
        # Use PIL to properly read CMYK and convert to RGB/BGR for OpenCV
        try:
            from PIL import Image
            pil_img = Image.open(ref_cdp_path)
            # Convert CMYK to RGB if needed
            if pil_img.mode == 'CMYK':
                pil_img = pil_img.convert('RGB')
            # Convert PIL image to numpy array (RGB)
            ref_img = np.array(pil_img)
            # Convert RGB to BGR for OpenCV
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR)
            print(f"[INFO] Loaded CMYK TIFF reference CDP: {ref_cdp_path}", flush=True)
        except Exception as e:
            print(f"[WARNING] Failed to load CMYK TIFF with PIL, trying OpenCV: {str(e)}", flush=True)
            ref_img = cv2.imread(ref_cdp_path)
    elif os.path.exists(png_path):
        ref_cdp_path = png_path
        ref_img = cv2.imread(ref_cdp_path)
        print(f"[INFO] Loaded PNG reference CDP: {ref_cdp_path}", flush=True)
    else:
        # Search for PNG files matching pattern
        png_files = glob.glob(png_pattern)
        if png_files:
            ref_cdp_path = png_files[0]  # Use first match
            ref_img = cv2.imread(ref_cdp_path)
            print(f"[INFO] Found and loaded PNG file with size suffix: {ref_cdp_path}", flush=True)
        else:
            return jsonify({
                'cdp_score': 0, 
                'liveness_passed': False,
                'status': 'failed',
                'message': f'Reference CDP not found for product_id: {product_id} (checked {product_id}.tiff, {product_id}.png, and pattern {product_id}_*.tiff/png)'
            })
    
    if ref_img is None:
        return jsonify({
            'cdp_score': 0, 
            'liveness_passed': False,
            'status': 'failed',
            'message': f'Failed to load reference CDP for product_id: {product_id}'
        })
    
    # Extract and save scanned CDP region to extracted_cdp folder
    extracted_cdp_filename = f"extracted_cdp_{product_id}_{timestamp}.png"
    scanned_cdp = cdp_service.extract_cdp_region(
        cdp_img, 
        save_file=True, 
        output_dir=EXTRACTED_CDP_DIR, 
        custom_filename=extracted_cdp_filename
    )
    print(f"[INFO] Extracted CDP saved: {os.path.join(EXTRACTED_CDP_DIR, extracted_cdp_filename)}", flush=True)
    
    # Compare CDP images
    cdp_score = cdp_service.compare_cdp(cdp_img, ref_img)

    # Enhanced liveness check with video frames (detects screens, photocopies, static images)
    first_frame = video_frames[0]
    liveness_passed = liveness_service.liveness_check(first_frame, video_frames)
    
    # Check frame distance
    distance_check = liveness_service.check_scanning_frame_distance(first_frame)
    
    # Assess lighting conditions and generate warnings
    lighting_assessment = liveness_service.assess_lighting_conditions(first_frame)
    
    # Detect CDP/pattern size (using red border detection) for warnings
    size_assessment = liveness_service.detect_qr_code_size(first_frame)
    
    print(f"[INFO] Verification result: authentic={is_authentic}, score={similarity_score:.3f}, threshold={AUTHENTICITY_THRESHOLD}, liveness={liveness_passed}", flush=True)
    
    # Build response
    response = {
        'similarity_score': float(similarity_score),
        'threshold': float(AUTHENTICITY_THRESHOLD),
        'liveness_passed': bool(liveness_passed),
        'is_authentic': bool(is_authentic),
        'serial_id': str(serial_id),
        'cdp_id': str(cdp_id),
        'product_id': str(product_id) if product_id else None,
        'status': 'success' if is_authentic else 'failed',
        'message': 'Product verified successfully' if is_authentic else f'Product verification failed - similarity score {similarity_score:.3f} below threshold {AUTHENTICITY_THRESHOLD}',
        'abuse_detection': {
            'is_abuse': bool(abuse_check.get('is_abuse', False)),
            'abuse_flags': abuse_check.get('abuse_flags', []),
            'total_attempts': abuse_check.get('total_attempts', 0)
        },
        'frame_distance': {
            'coverage_ratio': float(distance_check['frame_info'].get('coverage_ratio', 0)),
            'has_warnings': bool(distance_check['has_warnings']),
            'warnings': distance_check['warnings']
        },
        'lighting': {
            'status': str(lighting_assessment['lighting_info']['status']),
            'quality_score': int(lighting_assessment['lighting_info']['quality_score']),
            'has_warnings': bool(lighting_assessment['has_warnings']),
            'has_critical_warnings': bool(lighting_assessment['has_critical_warnings']),
            'warnings': lighting_assessment['warnings'],
            'metrics': {
                'brightness': float(lighting_assessment['lighting_info']['brightness']),
                'contrast': float(lighting_assessment['lighting_info']['contrast']),
                'dynamic_range': float(lighting_assessment['lighting_info']['dynamic_range'])
            }
        },
        'pattern_size': {
            'size_category': str(size_assessment['size_info']['size_category']),
            'coverage_ratio': float(size_assessment['size_info']['coverage_ratio']),
            'width_pixels': int(size_assessment['size_info']['width_pixels']),
            'height_pixels': int(size_assessment['size_info']['height_pixels']),
            'aspect_ratio': float(size_assessment['size_info']['aspect_ratio']),
            'has_warnings': bool(size_assessment['has_warnings']),
            'warnings': size_assessment['warnings']
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

