"""
Authenticity Classifier Service
Uses 15 extracted metrics to classify images as "real" (authentic print) vs "duplicate" (photocopy/screenshot).

This is Approach A: Simple, fast, interpretable classification using only your metrics.
"""
import numpy as np
import joblib
import os
from typing import Dict, Optional, List
from pathlib import Path
import warnings

# Try to import ML libraries (optional)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Install with: pip install scikit-learn")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")


class AuthenticityClassifier:
    """
    Classifier that uses 15 base metrics + 4 derived features to detect authentic prints vs duplicates.
    
    Input Features (19 features + lighting):
    Base metrics (15):
    1. Sharpness
    2. Contrast
    3. HistogramPeak
    4. EdgeDensity
    5. EdgeStrength
    6. NoiseLevel
    7. HighFreqEnergy
    8. ColorDiversity
    9. UniqueColors
    10. Saturation
    11. TextureUniformity
    12. CompressionArtifacts
    13. HistogramEntropy
    14. DynamicRange
    15. Brightness
    
    16. LightingCondition (encoded)
    """
    
    # Feature names in order
    FEATURE_NAMES = [
        'Sharpness', 'Contrast', 'HistogramPeak', 'EdgeDensity',
        'EdgeStrength', 'NoiseLevel', 'HighFreqEnergy', 
        'ColorDiversity', 'UniqueColors', 'Saturation',
        'TextureUniformity', 'CompressionArtifacts',
        'HistogramEntropy', 'DynamicRange', 'Brightness',
        'LightingEncoded'
    ]
    
    # Lighting condition encoding
    LIGHTING_MAP = {'bright': 0, 'normal': 1, 'dim': 2, 'low': 3}
    
    def __init__(self, model_type: str = 'xgboost', model_path: Optional[str] = None):
        """
        Initialize classifier.
        
        Args:
            model_type: 'random_forest' or 'xgboost'
            model_path: Path to saved model (if loading existing)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            self._create_model()
    
    def _create_model(self):
        """Create model based on type"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',  # Handle imbalanced data
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            )
        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available. Install with: pip install xgboost")
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _encode_lighting(self, lighting: str) -> int:
        """Encode lighting condition to integer"""
        return self.LIGHTING_MAP.get(lighting.lower(), 1)  # Default to 'normal'
    
    def _prepare_features(self, metrics_dict: Dict) -> np.ndarray:
        """
        Convert metrics dictionary to feature vector.
        
        Args:
            metrics_dict: Dictionary with metric names and values
        
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Extract features in order
        for feature_name in self.FEATURE_NAMES:
            if feature_name == 'LightingEncoded':
                # Special handling for lighting
                lighting = metrics_dict.get('LightingCondition', 'normal')
                features.append(self._encode_lighting(lighting))
            else:
                value = metrics_dict.get(feature_name, 0.0)
                features.append(float(value))
        
        return np.array(features, dtype=np.float32)
    
    def predict_single(self, metrics_dict: Dict) -> Dict:
        """
        Predict authenticity for a single image's metrics.
        
        Args:
            metrics_dict: Dictionary with all 15 base metrics + LightingCondition
                Example:
                {
                    'Sharpness': 107.54,
                    'Contrast': 84.59,
                    'NoiseLevel': 2.00,
                    'HighFreqEnergy': 2.36e10,
                    'LightingCondition': 'bright',
                    # ... all 15 metrics
                }
        
        Returns:
            Dictionary with prediction results:
            {
                'prediction': 'real' or 'duplicate',
                'confidence': float (0-100),
                'probabilities': {'real': 0.85, 'duplicate': 0.15},
                'is_authentic': bool,
                'raw_features': np.array
            }
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() or load() first.")
        
        # Prepare features
        features = self._prepare_features(metrics_dict)
        
        # Predict
        prediction_raw = self.model.predict(features.reshape(1, -1))[0]
        probabilities = self.model.predict_proba(features.reshape(1, -1))[0]
        
        # Handle XGBoost numeric outputs vs Random Forest string outputs
        if self.model_type == 'xgboost':
            # XGBoost outputs: 0 = duplicate, 1 = real
            reverse_label_encoder = {0: 'duplicate', 1: 'real'}
            prediction = reverse_label_encoder.get(int(prediction_raw), 'duplicate')
            # Probabilities are in order: [P(duplicate), P(real)]
            proba_dict = {
                'duplicate': float(probabilities[0]),
                'real': float(probabilities[1])
            }
            is_authentic = (prediction == 'real') or (int(prediction_raw) == 1)
        else:
            # Random Forest outputs string labels
            classes = self.model.classes_
            prediction = str(prediction_raw)
            proba_dict = {
                str(classes[0]): float(probabilities[0]),
                str(classes[1]): float(probabilities[1])
            }
            is_authentic = (prediction == 'real') or (prediction == 1)
        
        confidence = max(probabilities) * 100
        
        return {
            'prediction': str(prediction),
            'confidence': float(confidence),
            'probabilities': proba_dict,
            'is_authentic': bool(is_authentic),
            'raw_features': features
        }
    
    def train(self, training_data: List[Dict], test_size: float = 0.2, 
              verbose: bool = True) -> Dict:
        """
        Train model on labeled data.
        
        Args:
            training_data: List of dictionaries, each with:
                - All 15 base metrics
                - 'LightingCondition': str
                - 'Label': 'real' or 'duplicate'
            test_size: Fraction of data to use for testing
            verbose: Whether to print training progress
        
        Returns:
            Dictionary with training results and metrics
        """
        if not training_data:
            raise ValueError("Training data is empty")
        
        # Prepare features and labels
        X = []
        y = []
        
        for sample in training_data:
            features = self._prepare_features(sample)
            label = sample.get('Label', sample.get('label', None))
            
            if label is None:
                continue
            
            X.append(features)
            y.append(str(label).lower())
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) == 0:
            raise ValueError("No valid training samples found")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        if verbose:
            print(f"[INFO] Training on {len(X_train)} samples, testing on {len(X_test)} samples")
            print(f"[INFO] Class distribution - Train: {np.bincount([1 if l == 'real' else 0 for l in y_train])}")
            print(f"[INFO] Class distribution - Test: {np.bincount([1 if l == 'real' else 0 for l in y_test])}")
        
        # For XGBoost, encode string labels to numeric (0, 1)
        # 'duplicate' -> 0, 'real' -> 1
        if self.model_type == 'xgboost':
            label_encoder = {'duplicate': 0, 'real': 1}
            y_train_encoded = np.array([label_encoder.get(str(label).lower(), 0) for label in y_train])
            y_test_encoded = np.array([label_encoder.get(str(label).lower(), 0) for label in y_test])
        else:
            y_train_encoded = y_train
            y_test_encoded = y_test
        
        # Train
        if verbose:
            print(f"[INFO] Training {self.model_type} model...")
        
        self.model.fit(X_train, y_train_encoded)
        self.is_trained = True
        
        # Evaluate
        y_pred_encoded = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # For XGBoost, decode numeric predictions back to string labels
        if self.model_type == 'xgboost':
            reverse_label_encoder = {0: 'duplicate', 1: 'real'}
            y_pred = np.array([reverse_label_encoder.get(int(pred), 'duplicate') for pred in y_pred_encoded])
        else:
            y_pred = y_pred_encoded
        
        results = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'accuracy': float(np.mean(y_pred == y_test)),
        }
        
        if verbose:
            print("\n" + "="*50)
            print("Training Results")
            print("="*50)
            print(f"\nAccuracy: {results['accuracy']:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                importance_dict = dict(zip(self.FEATURE_NAMES, importances))
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                print("\nTop 10 Most Important Features:")
                for i, (feature, importance) in enumerate(sorted_importance[:10], 1):
                    print(f"  {i:2d}. {feature:25s}: {importance:.4f}")
                
                results['feature_importance'] = importance_dict
        
        # ROC AUC if binary classification
        if len(np.unique(y_test)) == 2:
            try:
                # Get probability of positive class (assuming 'real' is positive = 1)
                if self.model_type == 'xgboost':
                    # XGBoost: class 1 is 'real'
                    positive_class_idx = 1
                    y_test_binary = (y_test == 'real').astype(int)
                else:
                    # Random Forest: find index of 'real' class
                    positive_class_idx = list(self.model.classes_).index('real') if 'real' in self.model.classes_ else 1
                    y_test_binary = (y_test == 'real').astype(int)
                auc = roc_auc_score(y_test_binary, y_pred_proba[:, positive_class_idx])
                results['roc_auc'] = float(auc)
                if verbose:
                    print(f"\nROC AUC: {auc:.4f}")
            except Exception as e:
                if verbose:
                    print(f"\n[WARNING] Could not calculate ROC AUC: {str(e)}")
                pass
        
        return results
    
    def save(self, filepath: str):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save untrained model.")
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.FEATURE_NAMES,
            'lighting_map': self.LIGHTING_MAP
        }, filepath)
        print(f"[INFO] Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained model from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        data = joblib.load(filepath)
        self.model = data['model']
        self.model_type = data.get('model_type', 'xgboost')
        self.is_trained = True
        print(f"[INFO] Model loaded from {filepath}")


# Global instance (lazy loading)
_classifier = None

def get_classifier(model_type: str = 'xgboost', 
                  model_path: Optional[str] = None) -> AuthenticityClassifier:
    """
    Get or create global classifier instance.
    
    Args:
        model_type: 'random_forest' or 'xgboost'
        model_path: Path to saved model
    
    Returns:
        AuthenticityClassifier instance
    """
    global _classifier
    if _classifier is None:
        _classifier = AuthenticityClassifier(model_type=model_type, model_path=model_path)
    return _classifier

