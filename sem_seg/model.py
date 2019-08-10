import tensorflow as tf
import math
import time
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32,
                                    shape=(batch_size, num_point, 9))
    labels_pl = tf.placeholder(tf.int32,
                               shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ ConvNet baseline, input is BxNx9 gray image """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)

    point_cloud_xyz = point_cloud[:, :, 6:]

    edge_feature, idx = tf_util.pointSIFT_KNN(0.15, point_cloud_xyz)

    _, xyz = tf_util.group(point_cloud_xyz, idx)

    feature, _ = tf_util.group(point_cloud, idx)


    net = tf_util.ngcn_chebyshev(inputs=feature, num_output_channels=64,
                              local_cord=xyz,
                              bn=False, is_training=is_training,
                              scope='spec_conv%d' % (0), bn_decay=bn_decay)

    net_1 = tf.reduce_max(net, axis=-2, keep_dims=True)
    
    feature, _ = tf_util.group(tf.squeeze(net_1, axis=-2), idx)

    net = tf_util.ngcn_chebyshev(inputs=feature, num_output_channels=64,
                              local_cord=xyz,
                              bn=False, is_training=is_training,
                              scope='spec_conv%d' % (1), bn_decay=bn_decay)

    net_2 = tf.reduce_max(net, axis=-2, keep_dims=True)

    feature, _ = tf_util.group(tf.squeeze(net_2, axis=-2), idx)

    net = tf_util.ngcn_chebyshev(inputs=feature, num_output_channels=128,
                              local_cord=xyz,
                              bn=False, is_training=is_training,
                              scope='spec_conv%d' % (2), bn_decay=bn_decay)

    net_3 = tf.reduce_max(net, axis=-2, keep_dims=True)

    out7 = tf_util.conv2d(tf.concat([net_1, net_2, net_3], axis=-1), 1024, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=False, is_training=is_training,
                          scope='adj_conv7', bn_decay=bn_decay, is_dist=True)

    out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')

    expand = tf.tile(out_max, [1, num_point, 1, 1])

    concat = tf.concat(axis=3, values=[expand,
                                       net_1,
                                       net_2,
                                       net_3])

    # CONV
    net = tf_util.conv2d(concat, 512, [1, 1], padding='VALID', stride=[1, 1],
                         bn=False, is_training=is_training, scope='seg/conv1', is_dist=True)
    net = tf_util.conv2d(net, 256, [1, 1], padding='VALID', stride=[1, 1],
                         bn=False, is_training=is_training, scope='seg/conv2', is_dist=True)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    net = tf_util.conv2d(net, 13, [1, 1], padding='VALID', stride=[1, 1],
                         activation_fn=None, scope='seg/conv3', is_dist=True)
    net = tf.squeeze(net, [2])



    return net


def get_loss(pred, label):
    """ pred: B,N,13
        label: B,N """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    return tf.reduce_mean(loss)


if __name__ == "__main__":
    with tf.Graph().as_default():
        a = tf.placeholder(tf.float32, shape=(32, 4096, 9))
        net = get_model(a, tf.constant(True))
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            start = time.time()
            for i in range(100):
                print(i)
                sess.run(net, feed_dict={a: np.random.rand(32, 4096, 9)})
            print(time.time() - start)
