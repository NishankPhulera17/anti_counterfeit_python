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
        csv_path: Path to CSV file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
        
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
        
        print(f"[INFO] Training data appended to {csv_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to append training data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

