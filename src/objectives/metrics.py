import tensorflow as tf

def iou_metric(y_true, y_pred):
    threshold = 0.5
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - inter
    return tf.reduce_mean((inter + 1e-7) / (union + 1e-7))

def f1_metric(y_true, y_pred):
    threshold = 0.5
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    return 2 * ((precision * recall) / (precision + recall + 1e-7))

def fnr_metric(y_true, y_pred):
    threshold = 0.5
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    return fn / (fn + tp + 1e-7)

def fpr_metric(y_true, y_pred):
    threshold = 0.5
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))
    return fp / (fp + tn + 1e-7)

# Shorten names for the Keras progress bar so it fits on one line
iou_metric.__name__ = 'iou'
f1_metric.__name__ = 'f1'
fnr_metric.__name__ = 'fnr'
fpr_metric.__name__ = 'fpr'
