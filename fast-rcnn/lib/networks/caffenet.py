import tensorflow as tf
from networks.network import Network

class caffenet(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.rois = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'rois':self.rois})
        self.trainable = trainable
        self.setup()

    def setup(self):
        (self.feed('data')
             .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .lrn(2, 2e-05, 0.75, name='norm1')
             .conv(5, 5, 256, 1, 1, group=2, name='conv2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .lrn(2, 2e-05, 0.75, name='norm2')
             .conv(3, 3, 384, 1, 1, name='conv3')
             .conv(3, 3, 384, 1, 1, group=2, name='conv4')
             .conv(3, 3, 256, 1, 1, group=2, name='conv5'))

        (self.feed(['conv5', 'rois'])
             .roi_pool(6, 6, 1.0/16, name='pool5')
             .fc(4096, name='fc6')
             .fc(4096, name='fc7')
             .fc(4, relu=False, name='cls_score')
             .softmax(name='prob_cls'))

        (self.feed(['fc7'])
             .fc(16, relu=False, name='bbox_pred'))
