"""
Backend storage service for reference CDP features.
Stores features (not raw images) for security and efficiency.
"""
import os
import json
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List, Any
from pathlib import Path


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Recursively convert NumPy types and other non-JSON-serializable types
    to native Python types.
    
    Args:
        obj: Object that may contain NumPy types
    
    Returns:
        Object with all NumPy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


class BackendStorage:
    """
    Simple file-based backend storage for CDP features.
    In production, this would be replaced with a database (PostgreSQL, MongoDB, etc.)
    """
    
    def __init__(self, storage_dir: str = "backend_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Directories for different data types
        self.features_dir = self.storage_dir / "features"
        self.audit_dir = self.storage_dir / "audit"
        self.features_dir.mkdir(exist_ok=True)
        self.audit_dir.mkdir(exist_ok=True)
    
    def store_cdp_features(self, cdp_id: str, product_id: str, features: Dict, 
                          serial_id: str = None) -> bool:
        """
        Store reference CDP features.
        
        Args:
            cdp_id: Unique CDP identifier (UUID)
            product_id: Product identifier
            features: Extracted features dictionary
            serial_id: Optional serial ID for this item
        
        Returns:
            True if successful
        """
        try:
            # Convert features to JSON-serializable format (handle NumPy types)
            features_serializable = convert_to_json_serializable(features)
            
            record = {
                'cdp_id': cdp_id,
                'product_id': product_id,
                'serial_id': serial_id,
                'features': features_serializable,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            # Store as JSON file (in production, use database)
            file_path = self.features_dir / f"{cdp_id}.json"
            with open(file_path, 'w') as f:
                json.dump(record, f, indent=2)
            
            # Also create index by product_id for lookup
            index_file = self.features_dir / f"product_{product_id}.json"
            if index_file.exists():
                with open(index_file, 'r') as f:
                    index = json.load(f)
            else:
                index = {'cdp_ids': []}
            
            if cdp_id not in index['cdp_ids']:
                index['cdp_ids'].append(cdp_id)
                index['updated_at'] = datetime.utcnow().isoformat()
                with open(index_file, 'w') as f:
                    json.dump(index, f, indent=2)
            
            return True
        except Exception as e:
            print(f"[ERROR] Failed to store CDP features: {str(e)}")
            return False
    
    def get_cdp_features(self, cdp_id: str) -> Optional[Dict]:
        """
        Retrieve reference CDP features by CDP ID.
        
        Args:
            cdp_id: Unique CDP identifier
        
        Returns:
            Features dictionary or None if not found
        """
        try:
            file_path = self.features_dir / f"{cdp_id}.json"
            if not file_path.exists():
                return None
            
            with open(file_path, 'r') as f:
                record = json.load(f)
            
            return record.get('features')
        except Exception as e:
            print(f"[ERROR] Failed to retrieve CDP features: {str(e)}")
            return None
    
    def get_cdp_by_serial(self, serial_id: str) -> Optional[Dict]:
        """
        Retrieve CDP features by serial ID (from QR code).
        
        Args:
            serial_id: Serial ID from QR code
        
        Returns:
            Full record with features or None if not found
        """
        try:
            # Search all feature files for matching serial_id
            for file_path in self.features_dir.glob("*.json"):
                if file_path.name.startswith("product_"):
                    continue
                
                with open(file_path, 'r') as f:
                    record = json.load(f)
                
                if record.get('serial_id') == serial_id:
                    return record
            
            return None
        except Exception as e:
            print(f"[ERROR] Failed to retrieve CDP by serial: {str(e)}")
            return None
    
    def log_verification_attempt(self, cdp_id: str, serial_id: str, 
                                 success: bool, score: float, 
                                 metadata: Dict = None) -> bool:
        """
        Log verification attempt for audit and abuse detection.
        
        Args:
            cdp_id: CDP identifier
            serial_id: Serial ID from QR code
            success: Whether verification succeeded
            score: Similarity score
            metadata: Additional metadata (IP, user agent, geo, etc.)
        
        Returns:
            True if successful
        """
        try:
            audit_record = {
                'cdp_id': cdp_id,
                'serial_id': serial_id,
                'success': success,
                'score': score,
                'timestamp': datetime.utcnow().isoformat(),
                'metadata': metadata or {}
            }
            
            # Store audit log (in production, use time-series database)
            timestamp = datetime.utcnow()
            date_str = timestamp.strftime('%Y%m%d')
            audit_file = self.audit_dir / f"audit_{date_str}.jsonl"
            
            with open(audit_file, 'a') as f:
                f.write(json.dumps(audit_record) + '\n')
            
            return True
        except Exception as e:
            print(f"[ERROR] Failed to log verification attempt: {str(e)}")
            return False
    
    def detect_abuse(self, serial_id: str, time_window_minutes: int = 60) -> Dict:
        """
        Detect potential abuse patterns.
        
        Args:
            serial_id: Serial ID to check
            time_window_minutes: Time window for abuse detection
        
        Returns:
            Dictionary with abuse detection results
        """
        try:
            from datetime import timedelta
            cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
            
            # Count verification attempts in time window
            attempts = []
            for audit_file in self.audit_dir.glob("audit_*.jsonl"):
                with open(audit_file, 'r') as f:
                    for line in f:
                        record = json.loads(line)
                        if record.get('serial_id') == serial_id:
                            attempt_time = datetime.fromisoformat(record['timestamp'])
                            if attempt_time >= cutoff_time:
                                attempts.append(record)
            
            # Analyze patterns
            total_attempts = len(attempts)
            successful = sum(1 for a in attempts if a.get('success', False))
            failed = total_attempts - successful
            
            # Abuse indicators
            abuse_flags = []
            if total_attempts > 10:  # Too many attempts
                abuse_flags.append('high_frequency')
            if failed > 5:  # Many failed attempts
                abuse_flags.append('repeated_failures')
            if total_attempts > 0 and successful == 0:  # All failed
                abuse_flags.append('all_failed')
            
            return {
                'is_abuse': len(abuse_flags) > 0,
                'abuse_flags': abuse_flags,
                'total_attempts': total_attempts,
                'successful': successful,
                'failed': failed,
                'time_window_minutes': time_window_minutes
            }
        except Exception as e:
            print(f"[ERROR] Failed to detect abuse: {str(e)}")
            return {
                'is_abuse': False,
                'abuse_flags': [],
                'error': str(e)
            }


# Global instance
_backend_storage = None

def get_backend_storage() -> BackendStorage:
    """Get or create global backend storage instance"""
    global _backend_storage
    if _backend_storage is None:
        _backend_storage = BackendStorage()
    return _backend_storage

