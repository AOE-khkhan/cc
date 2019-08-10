import os
import sys
import tensorflow as tf
import numpy as np
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util



def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    cls_labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    
    return pointclouds_pl, labels_pl,cls_labels_pl

NUM_CATEGORIES = 16

def get_model(point_cloud,cls_label, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    point_cloud = point_cloud[:, :, 0:3]
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    input_label = cls_label
    cat_num = 16
    part_num = 50
    weight_decay = 0

    point_cloud_xyz = point_cloud[:, :, 0:3]

    edge_feature, idx = tf_util.pointSIFT_KNN(0.15, point_cloud_xyz)

    with tf.variable_scope('transform_net1') as sc:
        transform = tf_util.input_transform_net(edge_feature, is_training, bn_decay, K=3)

    point_cloud_transformed = tf.matmul(point_cloud, transform)

    feature, xyz = tf_util.group(point_cloud_transformed, idx)

    net = tf_util.ngcn_chebyshev(inputs=feature, num_output_channels=64,
                              local_cord=xyz,
                              bn=True, is_training=is_training,
                              scope='ngcn_chebyshev%d' % (0), bn_decay=bn_decay)

    net_max_1 = tf.reduce_max(net, axis=-2, keep_dims=True)
    net_mean_1 = tf.reduce_mean(net, axis=-2, keep_dims=True)

    out3 = tf_util.conv2d(tf.concat([net_max_1, net_mean_1], axis=-1), 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training, weight_decay=weight_decay,
                          scope='adj_conv3', bn_decay=bn_decay, is_dist=True)

    feature, _ = tf_util.group(tf.squeeze(out3, axis=-2), idx)

    net = tf_util.ngcn_chebyshev(inputs=feature, num_output_channels=128,
                              local_cord=xyz,
                              bn=True, is_training=is_training,
                              scope='ngcn_chebyshev%d' % (1), bn_decay=bn_decay)

    net_max_2 = tf.reduce_max(net, axis=-2, keep_dims=True)
    net_mean_2 = tf.reduce_mean(net, axis=-2, keep_dims=True)

    out5 = tf_util.conv2d(tf.concat([net_max_2, net_mean_2], axis=-1), 128, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training, weight_decay=weight_decay,
                          scope='adj_conv5', bn_decay=bn_decay, is_dist=True)

    feature, _ = tf_util.group(tf.squeeze(out5, axis=-2), idx)

    net = tf_util.ngcn_chebyshev(inputs=feature, num_output_channels=128,
                              local_cord=xyz,
                              bn=True, is_training=is_training,
                              scope='ngcn_chebyshev%d' % (2), bn_decay=bn_decay)


    net_max_3 = tf.reduce_max(net, axis=-2, keep_dims=True)
    net_mean_3 = tf.reduce_mean(net, axis=-2, keep_dims=True)

    out7 = tf_util.conv2d(tf.concat([net_max_3, net_mean_3], axis=-1), 128, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training, weight_decay=weight_decay,
                          scope='adj_conv7', bn_decay=bn_decay, is_dist=True)

    out8 = tf_util.conv2d(tf.concat([out3, out5, out7], axis=-1), 1024, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training,
                          scope='adj_conv13', bn_decay=bn_decay, is_dist=True)

    out_max = tf_util.max_pool2d(out8, [num_point, 1], padding='VALID', scope='maxpool')

    one_hot_label_expand = tf.one_hot(input_label, depth=NUM_CATEGORIES, on_value=1.0, off_value=0.0)
    one_hot_label_expand = tf.reshape(one_hot_label_expand, [batch_size, 1, 1, cat_num])
    one_hot_label_expand = tf_util.conv2d(one_hot_label_expand, 128, [1, 1],
                                          padding='VALID', stride=[1, 1],
                                          bn=True, is_training=is_training,
                                          scope='one_hot_label_expand', bn_decay=bn_decay, is_dist=True)
    out_max = tf.concat(axis=3, values=[out_max, one_hot_label_expand])
    expand = tf.tile(out_max, [1, num_point, 1, 1])

    concat = tf.concat(axis=3, values=[expand,
                                       net_max_1,
                                       net_mean_1,
                                       out3,
                                       net_max_2,
                                       net_mean_2,
                                       out5,
                                       net_max_3,
                                       net_mean_3,
                                       out7,
                                       out8])

    net2 = tf_util.conv2d(concat, 256, [1, 1], padding='VALID', stride=[1, 1], bn_decay=bn_decay,
                          bn=True, is_training=is_training, scope='seg/conv1', weight_decay=weight_decay, is_dist=True)
    net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp1')
    net2 = tf_util.conv2d(net2, 256, [1, 1], padding='VALID', stride=[1, 1], bn_decay=bn_decay,
                          bn=True, is_training=is_training, scope='seg/conv2', weight_decay=weight_decay, is_dist=True)
    net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp2')
    net2 = tf_util.conv2d(net2, 128, [1, 1], padding='VALID', stride=[1, 1], bn_decay=bn_decay,
                          bn=True, is_training=is_training, scope='seg/conv3', weight_decay=weight_decay, is_dist=True)
    net2 = tf_util.conv2d(net2, part_num, [1, 1], padding='VALID', stride=[1, 1], activation_fn=None,
                          bn=False, scope='seg/conv4', weight_decay=weight_decay, is_dist=True)

    net = tf.reshape(net2, [batch_size, num_point, part_num])




    return net, end_points


def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss



if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)

