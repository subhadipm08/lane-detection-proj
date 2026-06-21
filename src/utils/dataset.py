import tensorflow as tf
import os
import glob

IMG_HEIGHT = 288
IMG_WIDTH = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def load_segmentation_example(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    # Normalize with ImageNet mean/std
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH], method='nearest')
    mask = tf.cast(mask > 127, tf.float32)
    return img, mask

def augment_segmentation(img, mask):
    if tf.random.uniform([]) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    return img, mask

def get_segmentation_dataset(split_dir, batch_size=16, is_training=True):
    img_dir = os.path.join(split_dir, 'images')
    mask_dir = os.path.join(split_dir, 'masks')
    
    # We sort to ensure consistent mapping between image and mask
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
    
    if len(img_paths) == 0:
        raise ValueError(f"No images found in {img_dir}")
        
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    dataset = dataset.map(load_segmentation_example, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(buffer_size=200)
        dataset = dataset.map(augment_segmentation, num_parallel_calls=tf.data.AUTOTUNE)
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

