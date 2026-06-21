import tensorflow as tf

def focal_loss(gamma=2.0, alpha=0.8):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -(alpha * y_true * tf.math.log(y_pred) +
                (1 - alpha) * (1 - y_true) * tf.math.log(1 - y_pred))
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return tf.reduce_mean(tf.pow(1 - p_t, gamma) * bce)
    return loss_fn

def dice_loss(smooth=1.0):
    def loss_fn(y_true, y_pred):
        yt, yp = tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1])
        inter = tf.reduce_sum(yt * yp)
        union = tf.reduce_sum(yt) + tf.reduce_sum(yp)
        return 1 - (2. * inter + smooth) / (union + smooth)
    return loss_fn

def combo_loss(focal_w=0.5, dice_w=0.5, gamma=2.0, alpha=0.8):
    f = focal_loss(gamma, alpha)
    d = dice_loss()
    def loss_fn(y_true, y_pred):
        return focal_w * f(y_true, y_pred) + dice_w * d(y_true, y_pred)
    return loss_fn
