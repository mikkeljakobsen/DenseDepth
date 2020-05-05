import keras.backend as K
import tensorflow as tf
import settings

def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=settings.MAX_DEPTH*settings.DEPTH_SCALE):
    if settings.USE_DEPTHNORM:
        maxDepthVal = settings.MAX_DEPTH/settings.MIN_DEPTH
    # Point-wise depth
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

    if settings.USE_L1_LOSS_ONLY:
        return l_depth
    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta

    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))