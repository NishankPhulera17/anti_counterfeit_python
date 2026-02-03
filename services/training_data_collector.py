"""
Training Data Collector Service
Appends verification data to CSV file for ML model training.
"""
import pandas as pd
import os
from typing import Dict, Optional
from pathlib import Path


def append_to_training_csv(metrics: Dict[str, float], 
                          lighting_condition: str,
                          label: str,
                          csv_path: str = "training_data/sample_data.csv") -> bool:
    """
    Append a new training sample to the CSV file.
    
    Args:
        metrics: Dictionary with all 15 metrics
        lighting_condition: Lighting condition ("bright", "normal", "dim", "low")
        label: Label ("real" or "duplicate")
        csv_path: Path to CSV file (relative or absolute)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Resolve path relative to project root if it's a relative path
        if not os.path.isabs(csv_path):
            # Get the project root (parent of services directory)
            project_root = Path(__file__).parent.parent
            csv_path = os.path.join(project_root, csv_path)
        
        # Ensure directory exists
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        
        # Prepare row data
        row_data = {
            'Sharpness': metrics.get('Sharpness', 0.0),
            'Contrast': metrics.get('Contrast', 0.0),
            'HistogramPeak': metrics.get('HistogramPeak', 0.0),
            'EdgeDensity': metrics.get('EdgeDensity', 0.0),
            'EdgeStrength': metrics.get('EdgeStrength', 0.0),
            'NoiseLevel': metrics.get('NoiseLevel', 0.0),
            'HighFreqEnergy': metrics.get('HighFreqEnergy', 0.0),
            'ColorDiversity': metrics.get('ColorDiversity', 0.0),
            'UniqueColors': metrics.get('UniqueColors', 0),
            'Saturation': metrics.get('Saturation', 0.0),
            'TextureUniformity': metrics.get('TextureUniformity', 0.0),
            'CompressionArtifacts': metrics.get('CompressionArtifacts', 0.0),
            'HistogramEntropy': metrics.get('HistogramEntropy', 0.0),
            'DynamicRange': metrics.get('DynamicRange', 0.0),
            'Brightness': metrics.get('Brightness', 0.0),
            'LightingCondition': lighting_condition,
            'Label': label
        }
        
        # Check if file exists
        if os.path.exists(csv_path):
            # Append to existing file
            df = pd.read_csv(csv_path)
            new_row = pd.DataFrame([row_data])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(csv_path, index=False)
        else:
            # Create new file with header
            df = pd.DataFrame([row_data])
            df.to_csv(csv_path, index=False)
        
        print(f"[INFO] Training data appended to {csv_path}", flush=True)
        print(f"[INFO] CSV file exists: {os.path.exists(csv_path)}, size: {os.path.getsize(csv_path) if os.path.exists(csv_path) else 0} bytes", flush=True)
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to append training data: {str(e)}", flush=True)
        print(f"[ERROR] CSV path attempted: {csv_path}", flush=True)
        import traceback
        traceback.print_exc()
        return False

