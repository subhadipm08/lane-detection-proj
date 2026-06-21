import json
import os
import cv2
import numpy as np
from tqdm import tqdm

def generate_masks():
    base_dir = '/home/subha08/my_projects/lane-detection-proj'
    splits_dir = os.path.join(base_dir, 'data', 'splits')
    tusimple_dir = os.path.join(base_dir, 'data', 'tusimple')
    
    splits = ['train', 'val', 'test']
    
    target_w, target_h = 512, 288
    orig_w, orig_h = 1280, 720
    
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    
    for split in splits:
        print(f"Processing split: {split}")
        split_file = os.path.join(splits_dir, f"{split}.txt")
        
        out_img_dir = os.path.join(base_dir, 'data', split, 'images')
        out_mask_dir = os.path.join(base_dir, 'data', split, 'masks')
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)
        
        with open(split_file, 'r') as f:
            lines = f.readlines()
            
        for line in tqdm(lines):
            data = json.loads(line)
            raw_file = data['raw_file']
            h_samples = data['h_samples']
            lanes = data['lanes']
            
            img_path = os.path.join(tusimple_dir, raw_file)
            if not os.path.exists(img_path):
                print(f"Missing image: {img_path}")
                continue
                
            # Create a unique filename based on the clip path
            unique_name = raw_file.replace('/', '_').replace('.jpg', '')
            out_img_path = os.path.join(out_img_dir, f"{unique_name}.jpg")
            out_mask_path = os.path.join(out_mask_dir, f"{unique_name}.png")
            
            # Load and resize image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            
            # Original tuSimple images are usually 1280x720, but just in case, read dimensions
            h, w = img.shape[:2]
            actual_scale_x = target_w / w
            actual_scale_y = target_h / h
            
            img_resized = cv2.resize(img, (target_w, target_h))
            cv2.imwrite(out_img_path, img_resized)
            
            # Draw mask
            mask = np.zeros((target_h, target_w), dtype=np.uint8)
            
            for lane in lanes:
                pts = []
                for x, y in zip(lane, h_samples):
                    if x != -2:
                        pts.append([int(x * actual_scale_x), int(y * actual_scale_y)])
                if len(pts) > 1:
                    pts = np.array([pts], dtype=np.int32)
                    # user specification: thickness ≈ 5px
                    cv2.polylines(mask, pts, isClosed=False, color=255, thickness=5)
                    
            cv2.imwrite(out_mask_path, mask)

if __name__ == '__main__':
    generate_masks()
