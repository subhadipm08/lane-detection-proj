import os
import sys
import glob
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

sys.path.append(os.path.abspath('.'))
from src.utils.menu import select_model

def run_prediction():
    base_dir = '/home/subha08/my_projects/lane-detection-proj'
    
    print("=== Lane Detection Inference Tool ===")
    
    # 1. Ask for model
    model_name = select_model()
    
    # 2. Ask for input data
    print("Input options:")
    print("  1. data/test_videos/easy_white.mp4")
    print("  2. data/test_videos/medium_yellow.mp4")
    print("  3. data/test_videos/hard_challenge.mp4")
    print("  Or enter any relative/absolute path to an image or video file.")
    
    choice = input("Select an option [1-3] or enter file path: ").strip()
    
    if choice == '1':
        input_path = os.path.join(base_dir, 'data', 'test_videos', 'easy_white.mp4')
    elif choice == '2':
        input_path = os.path.join(base_dir, 'data', 'test_videos', 'medium_yellow.mp4')
    elif choice == '3':
        input_path = os.path.join(base_dir, 'data', 'test_videos', 'hard_challenge.mp4')
    else:
        input_path = os.path.abspath(choice)
        
    if not os.path.exists(input_path):
        print(f"Error: Path '{input_path}' does not exist.")
        return
        
    # 3. Detect data type
    ext = os.path.splitext(input_path)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg']
    is_image = ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    if not is_video and not is_image:
        print(f"Error: Unsupported file extension '{ext}'")
        return
        
    print(f"Input type detected: {'Video' if is_video else 'Image'}")
    
    # 4. Load model from best_model.keras
    model_path = os.path.join(base_dir, 'checkpoints', model_name, 'best_model.keras')
    print(f"\nLoading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist. Please train the model first.")
        return
        
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # Create output dir
    out_dir = os.path.join(base_dir, 'output_predictions', model_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Normalization constants
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # 5. Process Image or Video
    if is_image:
        print(f"Processing image: {os.path.basename(input_path)}")
        img = cv2.imread(input_path)
        if img is None:
            print("Error: Failed to load image.")
            return
            
        # Resize original for display and input
        orig_resized = cv2.resize(img, (512, 288))
        
        # Normalize
        img_rgb = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_norm = (img_norm - mean) / std
        img_batch = np.expand_dims(img_norm, axis=0)
        
        # Predict
        pred = model.predict(img_batch, verbose=0)[0]
        pred_mask = (pred.squeeze() > 0.5).astype(np.uint8)
        
        # Create overlay
        overlay = orig_resized.copy()
        overlay[pred_mask == 1] = [0, 255, 0] # Green in BGR
        
        # Alpha blending
        alpha = 0.4
        overlay_blend = cv2.addWeighted(overlay, alpha, orig_resized, 1 - alpha, 0)
        
        # Stack side-by-side
        side_by_side = np.hstack([orig_resized, overlay_blend])
        
        # Save output
        out_name = f"pred_{os.path.basename(input_path)}"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, side_by_side)
        print(f"\nSuccess! Saved side-by-side image to: {out_path}")
        
    elif is_video:
        print(f"Processing video: {os.path.basename(input_path)}")
        cap = cv2.VideoCapture(input_path)
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Stacking side-by-side means output width is 2 * 512 = 1024
        out_size = (1024, 288)
        
        # Define codec and writer
        out_name = f"pred_{os.path.basename(input_path)}"
        out_path = os.path.join(out_dir, out_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, out_size)
        
        print(f"Total frames to process: {total_frames}")
        for _ in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame
            orig_resized = cv2.resize(frame, (512, 288))
            
            # Normalize
            img_rgb = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)
            img_norm = img_rgb.astype(np.float32) / 255.0
            img_norm = (img_norm - mean) / std
            img_batch = np.expand_dims(img_norm, axis=0)
            
            # Predict
            pred = model(img_batch, training=False)[0] # fast call
            pred_mask = (pred.numpy().squeeze() > 0.5).astype(np.uint8)
            
            # Create overlay
            overlay = orig_resized.copy()
            overlay[pred_mask == 1] = [0, 255, 0] # Green in BGR
            
            # Alpha blending
            alpha = 0.4
            overlay_blend = cv2.addWeighted(overlay, alpha, orig_resized, 1 - alpha, 0)
            
            # Stack side-by-side
            side_by_side = np.hstack([orig_resized, overlay_blend])
            
            # Write to output
            out_writer.write(side_by_side)
            
        cap.release()
        out_writer.release()
        print(f"\nSuccess! Saved side-by-side video to: {out_path}")

if __name__ == '__main__':
    run_prediction()
