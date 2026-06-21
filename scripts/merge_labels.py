import json
import os

def merge_labels():
    base_dir = '/home/subha08/my_projects/lane-detection-proj'
    train_json_paths = [
        'data/tusimple/train_set/label_data_0313.json',
        'data/tusimple/train_set/label_data_0531.json',
        'data/tusimple/train_set/label_data_0601.json'
    ]
    test_json_path = 'data/tusimple/test_set/test_label.json'
    
    combined_data = []
    seen_files = set()
    
    # Process train sets
    for p in train_json_paths:
        full_path = os.path.join(base_dir, p)
        if not os.path.exists(full_path):
            print(f"Warning: {full_path} not found")
            continue
        with open(full_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Prepend 'train_set/' to know where the clip came from
                raw_file = data['raw_file']
                new_raw_file = os.path.join('train_set', raw_file)
                if new_raw_file not in seen_files:
                    data['raw_file'] = new_raw_file
                    combined_data.append(data)
                    seen_files.add(new_raw_file)

    # Process test set
    full_path = os.path.join(base_dir, test_json_path)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                raw_file = data['raw_file']
                new_raw_file = os.path.join('test_set', raw_file)
                if new_raw_file not in seen_files:
                    data['raw_file'] = new_raw_file
                    combined_data.append(data)
                    seen_files.add(new_raw_file)
    else:
        print(f"Warning: {full_path} not found")
        
    out_path = os.path.join(base_dir, 'data', 'combined_labels.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for data in combined_data:
            f.write(json.dumps(data) + '\n')
            
    print(f"Merged {len(combined_data)} unique labels.")

if __name__ == '__main__':
    merge_labels()