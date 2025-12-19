"""
ML-Enhanced Feature Extraction for Anti-Counterfeit System
Uses pre-trained CNN models to extract deep learning features from CDP images.
This can be integrated alongside existing rule-based features.
"""
import cv2
import numpy as np
from typing import Dict, Optional
import warnings

# Try to import PyTorch, but make it optional
try:
    import torch
    import torchvision.models as models
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. ML features will be disabled. Install with: pip install torch torchvision")


class MLFeatureExtractor:
    """
    Extract deep learning features from CDP images using pre-trained CNNs.
    Can be used alongside rule-based features for enhanced verification.
    """
    
    def __init__(self, model_name: str = 'resnet50', use_gpu: bool = True):
        """
        Initialize ML feature extractor.
        
        Args:
            model_name: Model to use ('resnet50', 'resnet18', 'vgg16')
            use_gpu: Whether to use GPU if available
        """
        if not TORCH_AVAILABLE:
            self.available = False
            return
        
        self.model_name = model_name
        self.device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
        self.model = None
        self.transform = None
        self.available = True
        
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model and remove classification layer"""
        if not TORCH_AVAILABLE:
            return
        
        try:
            # Load pre-trained model
            if self.model_name == 'resnet50':
                model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
                model.fc = torch.nn.Identity()  # Remove classification layer
                self.feature_dim = 2048
            elif self.model_name == 'resnet18':
                model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
                model.fc = torch.nn.Identity()
                self.feature_dim = 512
            elif self.model_name == 'vgg16':
                model = models.vgg16(weights='VGG16_Weights.DEFAULT')
                model.classifier = torch.nn.Identity()
                self.feature_dim = 25088  # VGG16 feature size
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            self.model = model.to(self.device).eval()
            
            # ImageNet normalization
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
            ])
            
            print(f"[INFO] ML Feature Extractor loaded: {self.model_name} on {self.device}")
        except Exception as e:
            print(f"[ERROR] Failed to load ML model: {str(e)}")
            self.available = False
    
    def extract_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract deep learning features from CDP image.
        
        Args:
            image: CDP image in BGR format (OpenCV format)
        
        Returns:
            Feature vector as numpy array, or None if extraction failed
        """
        if not self.available or self.model is None:
            return None
        
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Preprocess
            img_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(img_tensor)
                features = features.cpu().numpy().flatten()
            
            return features.astype(np.float32)
        except Exception as e:
            print(f"[ERROR] ML feature extraction failed: {str(e)}")
            return None
    
    def compare_features(self, ref_features: np.ndarray, 
                        scan_features: np.ndarray) -> float:
        """
        Compare two feature vectors and return similarity score.
        
        Args:
            ref_features: Reference CDP features
            scan_features: Scanned CDP features
        
        Returns:
            Similarity score between 0 and 1 (1 = identical)
        """
        if ref_features is None or scan_features is None:
            return 0.0
        
        try:
            # Normalize features
            ref_norm = ref_features / (np.linalg.norm(ref_features) + 1e-6)
            scan_norm = scan_features / (np.linalg.norm(scan_features) + 1e-6)
            
            # Cosine similarity
            similarity = np.dot(ref_norm, scan_norm)
            
            # Convert from [-1, 1] to [0, 1]
            similarity = (similarity + 1.0) / 2.0
            
            return float(np.clip(similarity, 0.0, 1.0))
        except Exception as e:
            print(f"[ERROR] ML feature comparison failed: {str(e)}")
            return 0.0


# Global instance (lazy loading)
_ml_extractor = None

def get_ml_extractor(model_name: str = 'resnet50') -> MLFeatureExtractor:
    """
    Get or create global ML feature extractor instance.
    
    Args:
        model_name: Model to use
    
    Returns:
        MLFeatureExtractor instance
    """
    global _ml_extractor
    if _ml_extractor is None:
        _ml_extractor = MLFeatureExtractor(model_name=model_name)
    return _ml_extractor


def extract_ml_features(image: np.ndarray, 
                       model_name: str = 'resnet50') -> Optional[np.ndarray]:
    """
    Convenience function to extract ML features from an image.
    
    Args:
        image: CDP image in BGR format
        model_name: Model to use
    
    Returns:
        Feature vector or None
    """
    extractor = get_ml_extractor(model_name)
    return extractor.extract_features(image)


def compare_ml_features(ref_features: np.ndarray, 
                        scan_features: np.ndarray) -> float:
    """
    Convenience function to compare ML features.
    
    Args:
        ref_features: Reference features
        scan_features: Scanned features
    
    Returns:
        Similarity score (0-1)
    """
    extractor = get_ml_extractor()
    return extractor.compare_features(ref_features, scan_features)

