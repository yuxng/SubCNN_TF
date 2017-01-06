import tensorflow as tf
import os.path as osp

filename = osp.join(sop.dirname(__file__), 'localization.so')
_localization_module = tf.load_op_library(filename)
localization = _localization_module.localization
localization_grad = _localization_module.localization_grad
