import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.utils.dataset import get_segmentation_dataset
from src.utils.menu import select_model

def visualize():
    base_dir = '/home/subha08/my_projects/lane-detection-proj'
    test_dir = os.path.join(base_dir, 'data', 'test')
    model_name = select_model()
    
    out_dir = os.path.join(base_dir, 'output_visualizations', model_name)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading test dataset from {test_dir}...")
    test_ds = get_segmentation_dataset(test_dir, batch_size=4, is_training=False)
    
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
    
    # Constants for de-normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    print(f"Generating visualizations in {out_dir}/ ...")
    
    # Take 1 batch (4 images)
    for imgs, true_masks in test_ds.take(1):
        preds = model.predict(imgs)
        
        for i in range(len(imgs)):
            # 1. De-normalize image back to 0-255 uint8 format
            img = imgs[i].numpy()
            img = (img * std + mean) * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
            
            # 2. Extract masks
            true_mask = true_masks[i].numpy().squeeze()
            pred_mask = (preds[i].squeeze() > 0.5).astype(np.uint8)
            
            # 3. Create Color Overlays
            
            # Panel 1: True Mask Overlay
            true_overlay = img.copy()
            true_overlay[true_mask == 1] = [0, 255, 0] # Green for True Mask
            
            # Panel 2: Predicted Mask Overlay with TP, FP, FN
            pred_overlay = img.copy()
            
            tp = (true_mask == 1) & (pred_mask == 1)
            fp = (true_mask == 0) & (pred_mask == 1)
            fn = (true_mask == 1) & (pred_mask == 0)
            
            # Color scheme:
            # TP -> Green (Correct prediction)
            # FP -> Red (Predicted lane where there is none)
            # FN -> Blue (Missed lane)
            pred_overlay[tp] = [0, 255, 0]
            pred_overlay[fp] = [255, 0, 0]
            pred_overlay[fn] = [255, 0, 0] # Wait, Blue in BGR is [255, 0, 0]
            pred_overlay[fn] = [255, 0, 0] # Let's be explicitly clear. OpenCV is BGR: Blue is [255, 0, 0], Red is [0, 0, 255], Green is [0, 255, 0]
            
            # Actually, OpenCV images are loaded as BGR, but we initialized img manually. Let's fix colors:
            pred_overlay[tp] = [0, 255, 0]   # Green (BGR)
            pred_overlay[fp] = [0, 0, 255]   # Red (BGR)
            pred_overlay[fn] = [255, 0, 0]   # Blue (BGR)
            
            # Also fix true_overlay
            true_overlay[true_mask == 1] = [0, 255, 0] # Green
            
            # Alpha blending for transparency
            alpha = 0.6
            true_overlay_blend = cv2.addWeighted(true_overlay, alpha, img, 1 - alpha, 0)
            pred_overlay_blend = cv2.addWeighted(pred_overlay, alpha, img, 1 - alpha, 0)
            
            # 4. Plot side-by-side using matplotlib
            fig, ax = plt.subplots(1, 2, figsize=(16, 6))
            
            # True Mask Overlay
            ax[0].imshow(cv2.cvtColor(true_overlay_blend, cv2.COLOR_BGR2RGB))
            ax[0].set_title('Ground Truth Overlay')
            ax[0].axis('off')
            
            # Pred Mask Overlay
            ax[1].imshow(cv2.cvtColor(pred_overlay_blend, cv2.COLOR_BGR2RGB))
            ax[1].set_title('Prediction (Green=TP, Red=FP, Blue=FN)')
            ax[1].axis('off')
            
            # Save
            save_path = os.path.join(out_dir, f'pred_overlay_{i}.png')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            print(f"Saved: {save_path}")
            
    print("Done!")

if __name__ == '__main__':
    visualize()
