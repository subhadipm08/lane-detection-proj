import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import tensorflow as tf
import json
from src.utils.dataset import get_segmentation_dataset
from src.objectives.losses import combo_loss
from src.objectives.metrics import iou_metric, f1_metric, fnr_metric, fpr_metric
from src.models.unet_mobilenetv2_reg import get_unet_mobilenetv2_reg
from src.models.unet_mobilenetv2 import get_unet_mobilenetv2
from src.utils.menu import select_model

def train():
    base_dir = '/home/subha08/my_projects/lane-detection-proj'
    train_dir = os.path.join(base_dir, 'data', 'train')
    val_dir = os.path.join(base_dir, 'data', 'val')
    
    # Load Config
    config_path = os.path.join(base_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    model_name = select_model()
    
    # Default float32 precision
    tf.keras.mixed_precision.set_global_policy('float32')
    
    batch_size = config['batch_size']
    frozen_epochs = config['frozen_epochs']
    unfrozen_epochs = config['unfrozen_epochs']
    
    train_ds = get_segmentation_dataset(train_dir, batch_size=batch_size, is_training=True)
    val_ds = get_segmentation_dataset(val_dir, batch_size=batch_size, is_training=False)
    
    if model_name == 'unet_mobilenetv2_reg':
        model = get_unet_mobilenetv2_reg(input_shape=(288, 512, 3), max_lanes=4, l2_reg=5e-4)
    elif model_name == 'unet_mobilenetv2':
        model = get_unet_mobilenetv2(input_shape=(288, 512, 3), max_lanes=4)
    else:
        print(f"Unknown model choice: {model_name}")
        return
    
    # Stage 1: Train decoder only
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=config['learning_rate_stage1'], 
        weight_decay=config['weight_decay']
    )
    model.compile(
        optimizer=optimizer,
        loss=combo_loss(),
        metrics=[iou_metric, f1_metric, fnr_metric, fpr_metric]
    )
    
    checkpoint_dir = os.path.join(base_dir, 'checkpoints', model_name)
    weights_dir = os.path.join(base_dir, 'checkpoints', f'{model_name}_weights')
    log_dir = os.path.join(base_dir, 'logs', model_name)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(weights_dir, 'epoch_{epoch:02d}_val_iou_{val_iou:.4f}.weights.h5'),
            monitor='val_iou',
            mode='max',
            save_best_only=False,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
            monitor='val_iou',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(log_dir, 'history.csv'), append=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_iou',
            mode='max',
            factor=0.5,
            patience=config['patience_lr'],
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_iou',
            mode='max',
            patience=config['patience_es'],
            verbose=1
        )
    ]
    
    print(f"Starting Stage 1: Warm-up Decoder ({frozen_epochs} epochs)")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=frozen_epochs,
        callbacks=callbacks
    )
    
    # Stage 2: Unfreeze all layers
    print("Starting Stage 2: Full Fine-tuning")
    for layer in model.layers:
        layer.trainable = True
        
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=config['learning_rate_stage2'], 
        weight_decay=config['weight_decay']
    )
    model.compile(
        optimizer=optimizer,
        loss=combo_loss(),
        metrics=[iou_metric, f1_metric, fnr_metric, fpr_metric]
    )
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=frozen_epochs,
        epochs=frozen_epochs + unfrozen_epochs,
        callbacks=callbacks
    )

if __name__ == '__main__':
    train()
