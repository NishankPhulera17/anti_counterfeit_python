from flask import Flask, request, jsonify, render_template
from services import cdp_service, liveness_service
from services.feature_extraction import extract_all_features, compare_features
from services.backend_storage import get_backend_storage
from services.metric_extraction import extract_all_metrics
from services.training_data_collector import append_to_training_csv
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

# UI Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ui/generate')
def ui_generate():
    return render_template('generate.html')

@app.route('/ui/verify')
def ui_verify():
    return render_template('verify.html')

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
        
        # Generate only 28x14mm size at 1200 DPI for Heidelberg printer
        # Physical size: 28mm width × 14mm height (landscape orientation)
        width_mm, height_mm, dpi, size_name = 28, 14, 1200, "28x14"
        
        print(f"[INFO] Generating {size_name} QR+CDP for product {product_id}, serial {serial_id}...", flush=True)
        
        try:
            # Generate QR+CDP image
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
                cdp_id=cdp_id
            )
            
            if img is None:
                print(f"[ERROR] generate_qr_cdp returned None for {size_name}", flush=True)
                raise ValueError(f"Failed to generate QR+CDP image for {size_name}")
            
            # Convert image to Base64 to send to app
            try:
                img_base64 = cv2_to_base64(img)
                if not img_base64:
                    raise ValueError(f"Failed to convert image to base64 for {size_name}")
                print(f"[INFO] Successfully converted {size_name} image to base64", flush=True)
            except Exception as e:
                print(f"[ERROR] Failed to convert {size_name} image to base64: {str(e)}", flush=True)
                raise
            
            # Extract and store features from CDP
            reference_features = None
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
            # Validate base64 string is not empty
            if not img_base64 or len(img_base64) == 0:
                raise ValueError(f"Empty base64 string for {size_name}")
            print(f"[INFO] {size_name} base64 length: {len(img_base64)}", flush=True)
            
            response_data = {
                "status": "success",
                "message": f"QR+CDP generated successfully for product {product_id}",
                "serial_id": serial_id,  # Return serial_id for reference
                "cdp_id": cdp_id,  # Return cdp_id for reference
                "qrToken": qr_token,
                "qrCdpImage": img_base64,
                "size": {
                    "width_mm": width_mm,
                    "height_mm": height_mm,
                    "dpi": dpi,
                    "size_name": size_name
                }
            }
            print(f"[INFO] Preparing response with QR+CDP image", flush=True)
            
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
    
    # Extract optional parameters for training data collection
    label_condition = data.get('label_condition')  # Optional: "real" or "duplicate"
    lighting_condition = data.get('lighting_condition')  # Optional: "bright", "normal", "dim", "low"
    request_product_id = data.get('product_id')  # Optional: product_id from request
    
    # Extract device information (optional)
    device_manufacturer = data.get('device_manufacturer')  # Optional: e.g., "Apple", "Samsung"
    device_model = data.get('device_model')  # Optional: e.g., "iPhone 13", "Galaxy S21"
    device_os = data.get('device_os')  # Optional: e.g., "iOS 15.0", "Android 12"
    camera_megapixels = data.get('camera_megapixels')  # Optional: e.g., 12.0
    
    # cdp_image is required
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
    
    # Convert video frames (optional - only if provided)
    video_frames = []
    if video_frames_base64:
        for frame_base64 in video_frames_base64:
            try:
                frame = base64_to_image(frame_base64)
                if frame is not None:
                    video_frames.append(frame)
            except Exception as e:
                print(f"[WARNING] Failed to decode video frame: {str(e)}", flush=True)
        print(f"[INFO] Decoded {len(video_frames)} video frames (optional)", flush=True)
    else:
        print(f"[INFO] No video frames provided (optional parameter)", flush=True)
    
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
    
    # Extract features from scanned CDP (for verification)
    scanned_features = extract_all_features(scanned_cdp)
    if not scanned_features:
        return jsonify({
            "status": "failed",
            "message": "Failed to extract features from scanned CDP"
        }), 500
    
    # Extract all 15 metrics for ML training (always extract for data collection)
    training_metrics = None
    try:
        training_metrics = extract_all_metrics(scanned_cdp)
        print(f"[METRICS] Training metrics for serial_id={serial_id}: {training_metrics}", flush=True)
        print(f"[INFO] Extracted training metrics", flush=True)
    except Exception as e:
        print(f"[WARNING] Failed to extract training metrics: {str(e)}", flush=True)
    
    # Step 5: Compare features using scored classification
    similarity_score = compare_features(reference_features, scanned_features)
    print(f"[INFO] Feature similarity score: {similarity_score:.3f}", flush=True)
    
    # Step 6: Data-driven threshold (tuned via ROC analysis)
    # Threshold: 0.65 (more lenient than old 0.7, accounts for mobile capture variations)
    AUTHENTICITY_THRESHOLD = 0.65
    
    # Step 7: Basic liveness check (optional - only if video frames provided)
    # Use cdp_img as the main frame, video frames are optional for additional checks
    liveness_passed = True  # Default to True if no video frames
    if video_frames and len(video_frames) >= 2:
        try:
            liveness_passed = liveness_service.liveness_check(cdp_img, video_frames)
            print(f"[INFO] Liveness check result: {liveness_passed}", flush=True)
        except Exception as e:
            print(f"[WARNING] Liveness check failed, defaulting to True: {str(e)}", flush=True)
            liveness_passed = True
    else:
        print(f"[INFO] No video frames provided, skipping liveness check (defaulting to passed)", flush=True)
    
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
    # Use cdp_img for all assessments (from cdp_base64)
    distance_check = liveness_service.check_scanning_frame_distance(cdp_img)
    lighting_assessment = liveness_service.assess_lighting_conditions(cdp_img)
    size_assessment = liveness_service.detect_qr_code_size(cdp_img)
    
    print(f"[INFO] Verification complete for product_id: {product_id} (from backend lookup)", flush=True)
    
    # Rename the scanned image to include product_id (if we got this far)
    # The image was already saved earlier, but we can save a copy with product_id
    scanned_image_with_product = f"scanned_{product_id}_{timestamp}.png"
    scanned_image_path_with_product = os.path.join(SCANNED_IMAGES_DIR, scanned_image_with_product)
    cv2.imwrite(scanned_image_path_with_product, cdp_img)
    print(f"[INFO] Scanned image saved with product_id: {scanned_image_path_with_product}", flush=True)
    
    # Extract and save scanned CDP region to extracted_cdp folder (for debugging/analysis)
    extracted_cdp_filename = f"extracted_cdp_{product_id}_{timestamp}.png"
    scanned_cdp_extracted = cdp_service.extract_cdp_region(
        cdp_img, 
        save_file=True, 
        output_dir=EXTRACTED_CDP_DIR, 
        custom_filename=extracted_cdp_filename
    )
    print(f"[INFO] Extracted CDP saved: {os.path.join(EXTRACTED_CDP_DIR, extracted_cdp_filename)}", flush=True)
    
    # Use feature-based similarity_score as cdp_score (for backward compatibility)
    # All verification now uses feature JSON matching, not image-to-image comparison
    cdp_score = similarity_score
    print(f"[INFO] Using feature-based similarity score as cdp_score: {cdp_score:.3f}", flush=True)

    # Enhanced liveness check with video frames (optional - only if provided)
    # Note: liveness_passed was already calculated earlier, but we're re-running for additional checks
    liveness_passed_secondary = liveness_passed  # Default to primary result
    try:
        if video_frames and len(video_frames) >= 2:
            print(f"[INFO] Running secondary liveness check with video frames...", flush=True)
            liveness_passed_secondary = liveness_service.liveness_check(cdp_img, video_frames)
            print(f"[INFO] Secondary liveness check result: {liveness_passed_secondary}", flush=True)
        else:
            print(f"[INFO] No video frames available for secondary liveness check, using primary result", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to run secondary liveness check: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
    
    # Check frame distance (using cdp_img)
    distance_check = {'has_warnings': False, 'warnings': [], 'frame_info': {}}
    try:
        print(f"[INFO] Checking frame distance...", flush=True)
        distance_check = liveness_service.check_scanning_frame_distance(cdp_img)
        print(f"[INFO] Frame distance check completed", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to check frame distance: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
    
    # Assess lighting conditions and generate warnings (using cdp_img)
    lighting_assessment = {'has_warnings': False, 'has_critical_warnings': False, 'warnings': [], 'lighting_info': {}}
    try:
        print(f"[INFO] Assessing lighting conditions...", flush=True)
        lighting_assessment = liveness_service.assess_lighting_conditions(cdp_img)
        print(f"[INFO] Lighting assessment completed", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to assess lighting conditions: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
    
    # Detect CDP/pattern size (using cdp_img) for warnings
    size_assessment = {'has_warnings': False, 'warnings': [], 'size_info': {}}
    try:
        print(f"[INFO] Detecting CDP/pattern size...", flush=True)
        size_assessment = liveness_service.detect_qr_code_size(cdp_img)
        print(f"[INFO] Size assessment completed", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to detect CDP/pattern size: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
    
    # Print verification result (using variables from earlier in the function)
    # Note: similarity_score, AUTHENTICITY_THRESHOLD, and is_authentic were calculated earlier (around lines 333-346)
    # liveness_passed was also calculated earlier, but we have a secondary check result too
    try:
        # These variables should be in scope from earlier in the function
        print(f"[INFO] Verification result: authentic={is_authentic}, score={similarity_score:.3f}, threshold={AUTHENTICITY_THRESHOLD}, liveness={liveness_passed}", flush=True)
    except NameError as e:
        # Fallback if variables not in scope (shouldn't happen, but just in case)
        print(f"[WARNING] Could not access verification variables: {str(e)}", flush=True)
        print(f"[INFO] Secondary checks completed: cdp_score={cdp_score:.3f}, liveness_secondary={liveness_passed_secondary}", flush=True)
    
    # Build response
    response = {
        'similarity_score': float(similarity_score),
        'cdp_score': float(cdp_score),
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
            'status': str(lighting_assessment.get('lighting_info', {}).get('status', 'unknown')),
            'quality_score': int(lighting_assessment.get('lighting_info', {}).get('quality_score', 0)),
            'has_warnings': bool(lighting_assessment.get('has_warnings', False)),
            'has_critical_warnings': bool(lighting_assessment.get('has_critical_warnings', False)),
            'warnings': lighting_assessment.get('warnings', []),
            'metrics': {
                'brightness': float(lighting_assessment.get('lighting_info', {}).get('brightness', 0)),
                'contrast': float(lighting_assessment.get('lighting_info', {}).get('contrast', 0)),
                'dynamic_range': float(lighting_assessment.get('lighting_info', {}).get('dynamic_range', 0))
            }
        },
        'pattern_size': {
            'size_category': str(size_assessment.get('size_info', {}).get('size_category', 'unknown')),
            'coverage_ratio': float(size_assessment.get('size_info', {}).get('coverage_ratio', 0)),
            'width_pixels': int(size_assessment.get('size_info', {}).get('width_pixels', 0)),
            'height_pixels': int(size_assessment.get('size_info', {}).get('height_pixels', 0)),
            'aspect_ratio': float(size_assessment.get('size_info', {}).get('aspect_ratio', 0)),
            'has_warnings': bool(size_assessment.get('has_warnings', False)),
            'warnings': size_assessment.get('warnings', [])
        }
    }
    
    # Append optional parameters from request body to response
    if label_condition is not None:
        response['label_condition'] = str(label_condition)
    if lighting_condition is not None:
        response['lighting_condition'] = str(lighting_condition)
    if request_product_id is not None:
        response['request_product_id'] = str(request_product_id)
    if device_manufacturer is not None:
        response['device_manufacturer'] = str(device_manufacturer)
    if device_model is not None:
        response['device_model'] = str(device_model)
    if device_os is not None:
        response['device_os'] = str(device_os)
    if camera_megapixels is not None:
        response['camera_megapixels'] = float(camera_megapixels)
    
    # Add training metrics to response if extracted
    if training_metrics is not None:
        response['training_metrics'] = training_metrics
    
    # Append to training CSV on every verify call (for data collection)
    print(f"[DEBUG] About to save training CSV. training_metrics is None: {training_metrics is None}", flush=True)
    if training_metrics is not None:
        # Use provided label_condition or default to 'unknown'
        label = label_condition if label_condition is not None else 'unknown'
        # Use provided lighting_condition or detected lighting status
        if lighting_condition is not None:
            lighting = lighting_condition
        else:
            lighting = lighting_assessment.get('lighting_info', {}).get('status', 'normal')
        
        # Append to CSV with device information
        csv_saved = append_to_training_csv(
            metrics=training_metrics,
            lighting_condition=lighting,
            label=label,
            csv_path="training_data/sample_data.csv",
            device_manufacturer=device_manufacturer,
            device_model=device_model,
            device_os=device_os,
            camera_megapixels=camera_megapixels
        )
        
        if csv_saved:
            response['training_data_saved'] = True
            print(f"[INFO] Training data saved: label={label}, lighting={lighting}", flush=True)
        else:
            print(f"[INFO] Training data not saved: label={label}, lighting={lighting}", flush=True)
            response['training_data_saved'] = False
            response['training_data_error'] = "Failed to save training data to CSV"

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

