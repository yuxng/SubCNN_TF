import os.path
import tensorflow as tf

_roi_pooling_module = tf.load_op_library('roi_pooling.so')
roi_pool = _roi_pooling_module.roi_pool
roi_pool_grad = _roi_pooling_module.roi_pool_grad
