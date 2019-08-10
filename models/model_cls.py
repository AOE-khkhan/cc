import tensorflow as tf
import numpy as np
import math
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops'))
import tf_util


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    feature, idx = tf_util.pointSIFT_KNN(0.15, point_cloud)

    with tf.variable_scope('transform_net1') as sc:
        transform = tf_util.input_transform_net(feature, is_training, bn_decay, K=3)

    point_cloud_transformed = tf.matmul(point_cloud, transform)

    feature, xyz = tf_util.group(point_cloud_transformed, idx)

    adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
    nn_idx = tf_util.knn(adj_matrix, k=32)
    feature = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=32)
    net_dknn = tf_util.ngcn_chebyshev(inputs=feature, num_output_channels=64,
                                    local_cord=xyz,
                                    bn=True, is_training=is_training,
                                    scope='dknn%d' % (1), bn_decay=bn_decay)
    net_dknn_1 = tf.reduce_max(net_dknn, axis=-2, keep_dims=True)

    feature, b = tf_util.group(tf.squeeze(net_dknn_1, [2]), idx)
    net_knn = tf_util.ngcn_chebyshev(inputs=feature, num_output_channels=64,
                                       local_cord=xyz,
                                       bn=True, is_training=is_training,
                                       scope='knn%d' % (1), bn_decay=bn_decay)
    net_knn_1 = tf.reduce_max(net_knn, axis=-2, keep_dims=True)

    adj_matrix = tf_util.pairwise_distance(tf.squeeze(net_knn_1, [2]))
    nn_idx = tf_util.knn(adj_matrix, k=32)
    feature = tf_util.get_edge_feature(tf.squeeze(net_knn_1, [2]), nn_idx=nn_idx, k=32)

    net_dknn = tf_util.ngcn_chebyshev(inputs=feature, num_output_channels=128,
                                    local_cord=xyz,
                                    bn=True, is_training=is_training,
                                    scope='dknn%d' % (2), bn_decay=bn_decay)
    net_dknn_2 = tf.reduce_max(net_dknn, axis=-2, keep_dims=True)

    feature, b = tf_util.group(tf.squeeze(net_dknn_2, [2]), idx)
    net_knn = tf_util.ngcn_chebyshev(inputs=feature, num_output_channels=128,
                                       local_cord=xyz,
                                       bn=True, is_training=is_training,
                                       scope='knn%d' % (2), bn_decay=bn_decay)
    net_knn_2 = tf.reduce_max(net_knn, axis=-2, keep_dims=True)

    adj_matrix = tf_util.pairwise_distance(tf.squeeze(net_knn_2, [2]))
    nn_idx = tf_util.knn(adj_matrix, k=32)
    feature = tf_util.get_edge_feature(tf.squeeze(net_knn_2, [2]), nn_idx=nn_idx, k=32)

    net_dknn = tf_util.ngcn_chebyshev(inputs=feature, num_output_channels=256,
                                    local_cord=xyz,
                                    bn=True, is_training=is_training,
                                    scope='dknn%d' % (3), bn_decay=bn_decay)

    net_dknn_3 = tf.reduce_max(net_dknn, axis=-2, keep_dims=True)

    feature, b = tf_util.group(tf.squeeze(net_dknn_3, [2]), idx)

    net_knn = tf_util.ngcn_chebyshev(inputs=feature, num_output_channels=256,
                                       local_cord=xyz,
                                       bn=True, is_training=is_training,
                                       scope='knn%d' % (3), bn_decay=bn_decay)
    net_knn_3 = tf.reduce_max(net_knn, axis=-2, keep_dims=True)

    net = tf_util.conv2d(
        tf.concat([net_dknn_1, net_knn_1, net_dknn_2, net_knn_2, net_dknn_3, net_knn_3], axis=-1),
        1024, [1, 1],
        padding='VALID', stride=[1, 1],
        bn=True, is_training=is_training,
        scope='agg', bn_decay=bn_decay)

    net = tf.reduce_max(net, axis=1, keep_dims=True)

    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points

def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    return classify_loss


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
