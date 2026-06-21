import sys
import os
sys.path.append(os.path.abspath('.'))

import tensorflow as tf
from src.utils.dataset import get_segmentation_dataset
from src.objectives.losses import combo_loss
from src.objectives.metrics import iou_metric, f1_metric
from src.models.unet_mobilenetv2 import get_unet_mobilenetv2

def test_pipeline():
    # 1. Test dataset
    base_dir = '/home/subha08/my_projects/lane-detection-proj'
    split_dir = os.path.join(base_dir, 'data', 'train')
    
    ds = get_segmentation_dataset(split_dir, batch_size=4, is_training=True)
    
    for imgs, masks in ds.take(1):
        print("Batch shape:", imgs.shape, masks.shape)
        print("Image range:", tf.reduce_min(imgs).numpy(), "-", tf.reduce_max(imgs).numpy())
        print("Mask range:", tf.reduce_min(masks).numpy(), "-", tf.reduce_max(masks).numpy())
        
        # 2. Test model
        model = get_unet_mobilenetv2(input_shape=(288, 512, 3), max_lanes=4)
        print("Model params:", model.count_params())
        
        # Model returns a single mask output (no auxiliary branch is used for training)
        mask_out = model(imgs, training=False)
        print("Mask pred shape:", mask_out.shape)
        
        # 3. Test losses and metrics
        cl = combo_loss()
        loss = cl(masks, mask_out)
        print("Combo loss:", loss.numpy())
        
        iou = iou_metric(masks, mask_out)
        f1 = f1_metric(masks, mask_out)
        print("IoU:", iou.numpy())
        print("F1:", f1.numpy())

if __name__ == '__main__':
    test_pipeline()