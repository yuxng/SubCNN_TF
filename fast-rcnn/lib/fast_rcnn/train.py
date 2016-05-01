# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

from fast_rcnn.config import cfg
import gt_data_layer.roidb as gdl_roidb
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import sys

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, network, imdb, roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print 'Computing bounding-box regression targets...'
        if cfg.TRAIN.BBOX_REG:
            if cfg.IS_RPN:
                self.bbox_means, self.bbox_stds = gdl_roidb.add_bbox_regression_targets(roidb)
            else:
                self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print 'done'

        # For checkpoint
        self.saver = tf.train.Saver()
        self.ckpt_path = os.path.join(self.output_dir, 'model.ckpt')


    def snapshot(self, sess):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            # save original values
            with tf.variable_scope("bbox_pred", reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = tf.Variable(weights.initialized_value(), name="orig_0")
            orig_1 = tf.Variable(biases.initialized_value(), name="orig_1")

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights = weights * self.bbox_stds[:, np.newaxis]
            biases = biases * self.bbox_stds + self.bbox_means

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.saver.save(sess, self.ckpt_path)
        print 'Wrote snapshot to: {:s}'.format(self.ckpt_path)

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            # restore net to original state
            weights = orig_0
            biases = orig_1

    def train_model(self, sess, max_iters):
        """Network training loop."""

        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)

        # classification loss
        cls_score = self.net.get_output('cls_score')
        label = tf.placeholder(tf.int32, shape=[None])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(cls_score, label))

        # bounding box regression L2 loss
        bbox_pred = self.net.get_output('bbox_pred')
        bbox_targets = tf.placeholder(tf.float32, shape=[None, 4 * self.imdb.num_classes])
        bbox_weights = tf.placeholder(tf.float32, shape=[None, 4 * self.imdb.num_classes])
        loss_box = tf.reduce_mean(tf.reduce_sum(tf.mul(bbox_weights, tf.square(tf.sub(bbox_pred, bbox_targets))), reduction_indices=[1]))

        # multi-task loss
        loss = tf.add(cross_entropy, loss_box)

        lr = 1e-3
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)

        # intialize variables
        sess.run(tf.initialize_all_variables())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, True)

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):
            # get one batch
            blobs = data_layer.forward()

            # Make one SGD update
            feed_dict={self.net.data: blobs['data'], self.net.rois: blobs['rois'], self.net.keep_prob: 0.5, \
                       label: blobs['labels'], bbox_targets: blobs['bbox_targets'], bbox_weights: blobs['bbox_inside_weights']}
            
            timer.tic()
            loss_cls_value, loss_box_value, _ = sess.run([cross_entropy, loss_box, train_op], feed_dict=feed_dict)
            timer.toc()

            print 'iter: %d / %d, loss_cls: %.4f, loss_box: %.4f, lr: %f' %\
                    (iter+1, max_iters, loss_cls_value, loss_box_value, lr)

            if iter == 10:
                sys.exit()

            if (iter+1) % (10 * cfg.TRAIN.DISPLAY) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess)

        if last_snapshot_iter != iter:
            self.snapshot(sess)

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    if cfg.IS_RPN:
        if cfg.IS_MULTISCALE:
            gdl_roidb.prepare_roidb(imdb)
        else:
            rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb


def get_data_layer(roidb, num_classes):
    """return a data layer."""
    if cfg.IS_RPN:
        if cfg.IS_MULTISCALE:
            layer = GtDataLayer(roidb)
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer


def train_net(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""

    with tf.Session() as sess:
        sw = SolverWrapper(sess, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)

        print 'Solving...'
        sw.train_model(sess, max_iters)
        print 'done solving'
