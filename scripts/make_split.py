import json
import os
import random

def make_split():
    base_dir = '/home/subha08/my_projects/lane-detection-proj'
    combined_json = os.path.join(base_dir, 'data', 'combined_labels.json')
    splits_dir = os.path.join(base_dir, 'data', 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    if not os.path.exists(combined_json):
        print(f"Error: {combined_json} not found. Please run merge_labels.py first.")
        return

    with open(combined_json, 'r') as f:
        data_lines = f.readlines()
        
    print(f"Loaded {len(data_lines)} labels.")
    
    # Split by clip to avoid leakage.
    clip_map = {}
    for line in data_lines:
        data = json.loads(line)
        raw_file = data['raw_file']
        clip_id = os.path.dirname(raw_file)
        if clip_id not in clip_map:
            clip_map[clip_id] = []
        clip_map[clip_id].append(line)
        
    clips = list(clip_map.keys())
    # Shuffle clips securely
    random.seed(42)
    random.shuffle(clips)
    
    total_clips = len(clips)
    train_end = int(total_clips * 0.8)
    val_end = int(total_clips * 0.9)
    
    train_clips = clips[:train_end]
    val_clips = clips[train_end:val_end]
    test_clips = clips[val_end:]
    
    def write_split(clips_list, split_name):
        split_path = os.path.join(splits_dir, f"{split_name}.txt")
        count = 0
        with open(split_path, 'w') as f:
            for clip in clips_list:
                for line in clip_map[clip]:
                    f.write(line)
                    count += 1
        print(f"{split_name}: {len(clips_list)} clips, {count} images")
        
    write_split(train_clips, 'train')
    write_split(val_clips, 'val')
    write_split(test_clips, 'test')

if __name__ == '__main__':
    make_split()