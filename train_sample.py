import pandas as pd
import os
import sys
from services.authenticity_classifier import AuthenticityClassifier

def train_model():
    print("[INFO] Starting training with sample_data.csv...")
    
    # 1. Load Data
    csv_path = 'training_data/sample_data.csv'
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # 2. Normalize Data
    # Map 'fake' -> 'duplicate'
    # Map 'avg' -> 'normal' (assuming 'avg' was intended as 'average'/'normal' lighting)
    print("[INFO] Normalizing labels and lighting conditions...")
    
    if 'Label' in df.columns:
        df['Label'] = df['Label'].replace({'fake': 'duplicate'})
    
    if 'LightingCondition' in df.columns:
        df['LightingCondition'] = df['LightingCondition'].replace({'avg': 'normal'})

    # Convert to list of dicts as expected by the classifier
    training_data = df.to_dict('records')
    print(f"[INFO] Loaded {len(training_data)} samples.")

    # 3. Initialize Classifier
    classifier = AuthenticityClassifier(model_type='random_forest')

    # 4. Train
    # We use a very small test_size or even 0 for this tiny sample, 
    # but the classifier defaults to 0.2. 
    # With only 6 samples, 0.2 might result in only 1 test item or issues with stratification if classes are unbalanced.
    # Let's try default first, but catch errors.
    
    try:
        results = classifier.train(training_data, test_size=0.2)
        print("\nTraining Successful!")
        print(f"Accuracy: {results['accuracy']}")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        # Fallback: maybe not enough data for split?
        # The service forces train_test_split. 
        # For this demo, user just wants to see it work.
        return

    # 5. Save Model
    os.makedirs('models', exist_ok=True)
    classifier.save('models/authenticity_classifier_random_forest.pkl')

if __name__ == "__main__":
    train_model()
