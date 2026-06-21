import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import tensorflow as tf
from src.utils.dataset import get_segmentation_dataset
from src.objectives.losses import combo_loss
from src.objectives.metrics import iou_metric, f1_metric, fnr_metric, fpr_metric
from src.utils.menu import select_model

def evaluate():
    base_dir = '/home/subha08/my_projects/lane-detection-proj'
    test_dir = os.path.join(base_dir, 'data', 'test')
    
    model_name = select_model()
    
    batch_size = 12
    test_ds = get_segmentation_dataset(test_dir, batch_size=batch_size, is_training=False)
    
    model_path = os.path.join(base_dir, 'checkpoints', model_name, 'best_model.keras')
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist. Please train the model first.")
        return
        
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # We compile the model with metrics to easily evaluate
    model.compile(
        loss=combo_loss(),
        metrics=[iou_metric, f1_metric, fnr_metric, fpr_metric]
    )
    
    print("Evaluating on test set...")
    results = model.evaluate(test_ds)
    print("Test Results:", dict(zip(model.metrics_names, results)))

if __name__ == '__main__':
    evaluate()
