import tensorflow as tf
import numpy as np
import sys, time, os

slim = tf.contrib.slim

## TODO: change to use those folders with __init__
sys.path.insert(0, '.')
sys.path.insert(0, '../utils')
from basemodel import BaseModel
from upsampling import bilinear_upsample_weights

""" Implements Fully Convolutional Networks 32, 16, 8s versions:
https://github.com/shelhamer/fcn.berkeleyvision.org

@inproceedings{long2015fully,
  title={Fully convolutional networks for semantic segmentation},
  author={Long, Jonathan and Shelhamer, Evan and Darrell, Trevor},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3431--3440},
  year={2015}
}
"""

class FCNModel(BaseModel):
    def __init__(self,
        sess = None,
        n_classes = 2,
        log_dir = None,
        dataset = None,
        save_dir = None,
        bayesian = False,
        input_dims = 512,
        mode = 'TRAINING',
        input_channel = 3,
        test_dataset = None,
        learning_rate = 1e-4,
        load_snapshot = None,
        load_snapshot_from = None,
        n_kernels = 32,
        adversarial_training = False,
        fcn_type = '32s'):

        ## TODO: Check args

        super(FCNModel, self).__init__(
            sess=sess,
            mode=mode,
            log_dir=log_dir,
            dataset=dataset,
            bayesian=bayesian,
            save_dir=save_dir,
            n_classes=n_classes,
            input_dims=input_dims,
            test_dataset=test_dataset,
            input_channel=input_channel,
            load_snapshot=load_snapshot,
            learning_rate=learning_rate,
            load_snapshot_from=load_snapshot_from,
            adversarial_training=adversarial_training)

        self.model_name = 'FCN'
        print 'Setting up Fully Convolutional Network model'

        ## Ops
        self._init_input()

        ## Custom things for this model
        self.n_kernels = n_kernels
        self.fcn_type = fcn_type
        with tf.name_scope('FCN'):
            self.y_hat = self.model(self.input_x, reuse=False)

        with tf.name_scope('output'):
            self.y_hat_sig = tf.nn.sigmoid(self.y_hat)
            self.output = tf.expand_dims(tf.argmax(self.y_hat_sig, 3), -1)
            self.output = tf.cast(self.output, tf.float32)
        self.inference_ops = [self.y_hat_sig, self.output]

        ## Generics
        with tf.name_scope('loss') as scope:
            self._init_training_ops()

        with tf.name_scope('testing') as scope:
            self._init_testing()
        self._init_summary_ops()

        self.init_op = tf.global_variables_initializer()
        self.sess.run([self.init_op])
        self._init_saver(self.model_name)


    def model(self, input_op, reuse=False):
        self.zed = self.encode_model(input_op, reuse=reuse)
        if self.fcn_type == '32s':
            y_hat = self.fcn32s(self.zed)
        elif self.fcn_type == '16s':
            y_hat = self.fcn16s(self.zed)
        elif self.fcn_type == '8s':
            y_hat = self.fcn8s(self.zed)
        else:
            ## TODO: use actual errors
            print 'MODE ERROR'
        return y_hat

    def encode_model(self, input_op, reuse=False):
        self.x_dim, self.y_dim = input_op.get_shape().as_list()[1:3]
        print 'x_dim', self.x_dim
        print 'y_dim', self.y_dim
        net = slim.convolution2d(input_op,
            num_outputs = self.n_kernels,
            kernel_size = 3,
            stride = 1,
            padding = 'SAME',
            scope = 'conv1',
            reuse = reuse)
        net = slim.max_pool2d(net, 2, scope='pool1')
        net = slim.convolution2d(net, self.n_kernels*2, 3, 1, padding='SAME', scope='conv2', reuse=reuse)
        net = slim.max_pool2d(net, 2, scope='pool2')
        net = slim.convolution2d(net, self.n_kernels*4, 3, 1, padding='SAME', scope='conv3', reuse=reuse)
        self.pool3 = slim.max_pool2d(net, 2, scope='pool3')
        net = slim.convolution2d(self.pool3, self.n_kernels*8, 3, 1, padding='SAME', scope='conv4', reuse=reuse)
        self.pool4 = slim.max_pool2d(net, 2, scope='pool4')
        net = slim.convolution2d(self.pool4, self.n_kernels*8, 3, 1, padding='SAME', scope='conv5', reuse=reuse)
        self.pool5 = slim.max_pool2d(net, 2, scope='pool5')
        net = slim.convolution2d(self.pool5, self.n_kernels*32, 1, 1, padding='SAME', scope='conv6', reuse=reuse)
        net = slim.convolution2d(net, self.n_kernels*32, 1, 1, padding='SAME', scope='conv7', reuse=reuse)
        net = slim.convolution2d(net, self.n_classes, 1, 1, padding='SAME', scope='conv_fr', reuse=reuse)

        return net


    def fcn32s(self, net, reuse=False):
        with tf.variable_scope('fcn32s') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()

            upsample_filter_np = bilinear_upsample_weights(32, self.n_classes)
            upsample_filter_tensor = tf.constant(upsample_filter_np)

            batch_size, h, w, channels = net.get_shape().as_list()
            output = tf.nn.conv2d_transpose(net, upsample_filter_tensor, [batch_size, h*32, w*32, channels], [1,32,32,1], reuse=reuse)
            output = tf.image.resize_image_with_crop_or_pad(output, self.x_dim, self.y_dim)
            print 'fcn32s output', output.get_shape()
        return output


    def fcn16s(self, net, reuse=False):
        with tf.variable_scope('fcn16s') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()

            upscore_weights = bilinear_upsample_weights(2, self.n_classes)
            output_weights = bilinear_upsample_weights(16, self.n_classes)
            upscore_weights_tensor = tf.constant(upscore_weights)
            output_weights_tensor = tf.constant(output_weights)

            ## Score from pool4:
            pool4_score = slim.convolution2d(self.pool4, self.n_classes, 1, 1, padding='SAME', scope='pool4_score', reuse=reuse)
            pool4_h, pool4_w = pool4_score.get_shape().as_list()[1:3]
            ## Upsample the stream
            batch_size, h, w, _ = net.get_shape().as_list()
            upscore = tf.nn.conv2d_transpose(net, upscore_weights_tensor,
                [batch_size, h*2, w*2, self.n_classes], [1,2,2,1], reuse=reuse)
            ## Crop pool4_score to be same shape as upscore
            upscore_pool4_crop = tf.image.resize_image_with_crop_or_pad(upscore, pool4_h, pool4_h)
            ## Order invariant combination
            upscore = pool4_score + upscore_pool4_crop
            ## Final upsample
            h, w = upscore.get_shape().as_list()[1:3]
            upscore = tf.nn.conv2d_transpose(upscore, output_weights_tensor,
                [batch_size, h*16, w*16, self.n_classes], [1,16,16,1], reuse=reuse) ## 4 * 16 = 64
            ## Force to be the same size as input
            upscore = tf.image.resize_image_with_crop_or_pad(upscore, self.x_dim, self.y_dim)
            print 'fcn16s upscore', upscore.get_shape()
        return upscore


    def fcn8s(self, net, reuse=False):
        with tf.variable_scope('fcn8s') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()

            upscore_weights_3 = bilinear_upsample_weights(2, self.n_classes)
            upscore_weights_4 = bilinear_upsample_weights(2, self.n_classes)
            output_weights = bilinear_upsample_weights(8, self.n_classes)
            upscore_weights_3_tensor = tf.constant(upscore_weights_3)
            upscore_weights_4_tensor = tf.constant(upscore_weights_4)
            output_weights_tensor = tf.constant(output_weights)

            ## Score from pool3:
            pool3_score = slim.convolution2d(self.pool3, self.n_classes, 1, 1, padding='SAME', scope='pool3_score', reuse=reuse)
            pool3_h, pool3_w = pool3_score.get_shape().as_list()[1:3]
            ## Score from pool4:
            pool4_score = slim.convolution2d(self.pool4, self.n_classes, 1, 1, padding='SAME', scope='pool4_score', reuse=reuse)
            pool4_h, pool4_w = pool4_score.get_shape().as_list()[1:3]
            ## Upsample the stream
            batch_size, h, w, _ = net.get_shape().as_list()
            upscore = tf.nn.conv2d_transpose(net, upscore_weights_4_tensor,
                [batch_size, h*2, w*2, self.n_classes], [1,2,2,1], reuse=reuse)
            ## Crop pool4_score to be same shape as upscore
            upscore_pool4_crop = tf.image.resize_image_with_crop_or_pad(upscore, pool4_h, pool4_w)
            ## Order invariant combination
            upscore = pool4_score + upscore_pool4_crop
            ## Repeat for pool3
            batch_size, h, w, _ = upscore.get_shape().as_list()
            upscore = tf.nn.conv2d_transpose(upscore, upscore_weights_3_tensor,
                [batch_size, h*2, w*2, self.n_classes], [1,2,2,1], reuse=reuse)
            ## Crop pool4_score to be same shape as upscore
            upscore_pool3_crop = tf.image.resize_image_with_crop_or_pad(upscore, pool3_h, pool3_w)
            ## Order invariant combination
            upscore = pool3_score + upscore_pool3_crop
            ## Final upsample
            h, w = upscore.get_shape().as_list()[1:3]
            upscore = tf.nn.conv2d_transpose(upscore, output_weights_tensor,
                [batch_size, h*8, w*8, self.n_classes], [1,8,8,1], reuse=reuse) ## 4 * 16 = 64
            ## Force to be the same size as input
            upscore = tf.image.resize_image_with_crop_or_pad(upscore, self.x_dim, self.y_dim)
            print 'fcn8s upscore', upscore.get_shape()
        return upscore
