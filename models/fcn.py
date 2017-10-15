import tensorflow as tf
import numpy as np
import sys, time, os

slim = tf.contrib.slim

sys.path.insert(0, '.')
sys.path.insert(0, '..')
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
        fcn_type = '32s',
        save_dir = None,
        bayesian = False,
        input_dims = 224,
        mode = 'TRAINING',
        input_channel = 3,
        learning_rate = 1e-4,
        load_snapshot = None,
        load_snapshot_from = None,):

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
            input_channel=input_channel,
            load_snapshot=load_snapshot,
            learning_rate=learning_rate,
            load_snapshot_from=load_snapshot_from)

        self.model_name = 'deconvolution'
        print 'Setting up deconvolution model'

        ## Ops
        self._init_input()

        ## Custom things for this model
        self.n_kernels = 8
        self.fcn_type = fcn_type
        with tf.name_scope('FCN'):
            self.zed = self.encode_model()

            if self.fcn_type == '32s':
                self.y_hat = self.fcn32s(self.zed)
            elif self.fcn_type == '16s':
                self.y_hat = self.fcn16s(self.zed)
            elif self.fcn_type == '8s':
                self.y_hat = self.fcn8s(self.zed)
            else:
                ## TODO: use actual errors
                print 'MODE ERROR'
                return

        with tf.name_scope('output'):
            self.y_hat_sig = tf.nn.sigmoid(self.y_hat)
            self.output = tf.expand_dims(tf.argmax(self.y_hat_sig, 3), -1)
            self.output = tf.cast(self.output, tf.float32)
        self.inference_ops = [self.y_hat_sig, self.output]

        ## Generics
        self._init_training_ops()
        self._init_summary_ops()

        self.init_op = tf.global_variables_initializer()

        self.sess.run([self.init_op])

        ## Saver things; TODO add logic that skips the dataset load if we restore
        self._init_saver(self.model_name)

        # self.sess.run([self.init_op])


    def encode_model(self):
        self.x_dim, self.y_dim = self.input_x.get_shape().as_list()[1:3]
        print 'x_dim', self.x_dim
        print 'y_dim', self.y_dim
        net = slim.convolution2d(self.input_x,
            num_outputs = self.n_kernels,
            kernel_size = 3,
            stride = 1,
            padding = 'SAME',
            scope = 'conv1')
        net = slim.max_pool2d(net, 2, scope='pool1')
        print 'pool1', net.get_shape()

        net = slim.convolution2d(net, self.n_kernels*2, 3, 1, padding='SAME', scope='conv2')
        net = slim.max_pool2d(net, 2, scope='pool2')
        print 'pool2', net.get_shape()

        net = slim.convolution2d(net, self.n_kernels*4, 3, 1, padding='SAME', scope='conv3')
        self.pool3 = slim.max_pool2d(net, 2, scope='pool3')
        print 'pool3', self.pool3.get_shape()

        net = slim.convolution2d(self.pool3, self.n_kernels*8, 3, 1, padding='SAME', scope='conv4')
        self.pool4 = slim.max_pool2d(net, 2, scope='pool4')
        print 'pool4', self.pool4.get_shape()

        net = slim.convolution2d(self.pool4, self.n_kernels*8, 3, 1, padding='SAME', scope='conv5')
        self.pool5 = slim.max_pool2d(net, 2, scope='pool5')
        print 'pool5', self.pool5.get_shape()

        net = slim.convolution2d(self.pool5, self.n_kernels*32, 1, 1, padding='SAME', scope='conv6')
        net = slim.convolution2d(net, self.n_kernels*32, 1, 1, padding='SAME', scope='conv7')
        print 'FC', net.get_shape()

        net = slim.convolution2d(net, self.n_classes, 1, 1, padding='SAME', scope='conv_fr')
        print 'FC_fr', net.get_shape()

        return net


    def fcn32s(self, net):
            ## The behaviour of conv2d_transpose is... weird
            ## It seems like the 4th arg defines the output size
        upsample_filter_np = bilinear_upsample_weights(32, self.n_classes)
        upsample_filter_tensor = tf.Variable(upsample_filter_np)
        batch_size, h, w, channels = net.get_shape().as_list()
        output_h = h * 32
        output_w = w * 32
        output = tf.nn.conv2d_transpose(net, upsample_filter_tensor, [batch_size, output_h, output_w, channels], [1,32,32,1])

        output = tf.image.resize_image_with_crop_or_pad(output, self.x_dim, self.y_dim)
        print 'fcn32s output', output.get_shape()
        return output


    def fcn16s(self, net):
        upscore_weights = bilinear_upsample_weights(2, self.n_classes)
        output_weights = bilinear_upsample_weights(16, self.n_classes)

        upscore_weights_tensor = tf.Variable(upscore_weights)
        output_weights_tensor = tf.Variable(output_weights)

        ## Score from pool4:
        pool4_score = slim.convolution2d(self.pool4, self.n_classes, 1, 1, padding='SAME', scope='pool4_score')
        pool4_h, pool4_w = pool4_score.get_shape().as_list()[1:3]
        print 'pool4_score', pool4_score.get_shape()

        ## Upsample the stream
        batch_size, h, w, _ = net.get_shape().as_list()
        upscore = tf.nn.conv2d_transpose(net, upscore_weights_tensor,
            [batch_size, h*2, w*2, self.n_classes], [1,2,2,1])
        print 'fcn16 upscore', upscore.get_shape()

        ## Crop pool4_score to be same shape as upscore
        upscore_pool4_crop = tf.image.resize_image_with_crop_or_pad(upscore, pool4_h, pool4_h)
        print 'upscore_pool4_crop', upscore_pool4_crop.get_shape()

        ## Order invariant combination
        upscore = pool4_score + upscore_pool4_crop
        print 'upscore ', upscore.get_shape()

        ## Final upsample
        h, w = upscore.get_shape().as_list()[1:3]
        upscore = tf.nn.conv2d_transpose(upscore, output_weights_tensor,
            [batch_size, h*16, w*16, self.n_classes], [1,16,16,1]) ## 4 * 16 = 64

        ## Force to be the same size as input
        upscore = tf.image.resize_image_with_crop_or_pad(upscore, self.x_dim, self.y_dim)
        print 'fcn16s upscore', upscore.get_shape()
        return upscore


    def fcn8s(self, net):
        upscore_weights_3 = bilinear_upsample_weights(2, self.n_classes)
        upscore_weights_4 = bilinear_upsample_weights(2, self.n_classes)
        output_weights = bilinear_upsample_weights(8, self.n_classes)

        upscore_weights_3_tensor = tf.Variable(upscore_weights_3)
        upscore_weights_4_tensor = tf.Variable(upscore_weights_4)
        output_weights_tensor = tf.Variable(output_weights)

        ## Score from pool3:
        pool3_score = slim.convolution2d(self.pool3, self.n_classes, 1, 1, padding='SAME', scope='pool3_score')
        pool3_h, pool3_w = pool3_score.get_shape().as_list()[1:3]
        print 'pool3_score', pool3_score.get_shape()

        ## Score from pool4:
        pool4_score = slim.convolution2d(self.pool4, self.n_classes, 1, 1, padding='SAME', scope='pool4_score')
        pool4_h, pool4_w = pool4_score.get_shape().as_list()[1:3]
        print 'pool4_score', pool4_score.get_shape()

        ## Upsample the stream
        batch_size, h, w, _ = net.get_shape().as_list()
        upscore = tf.nn.conv2d_transpose(net, upscore_weights_4_tensor,
            [batch_size, h*2, w*2, self.n_classes], [1,2,2,1])
        print 'stream upscore', upscore.get_shape()

        ## Crop pool4_score to be same shape as upscore
        upscore_pool4_crop = tf.image.resize_image_with_crop_or_pad(upscore, pool4_h, pool4_w)
        print 'upscore_pool4_crop', upscore_pool4_crop.get_shape()

        ## Order invariant combination
        upscore = pool4_score + upscore_pool4_crop
        print 'upscore ', upscore.get_shape()

        ## Repeat for pool3
        batch_size, h, w, _ = upscore.get_shape().as_list()
        upscore = tf.nn.conv2d_transpose(upscore, upscore_weights_3_tensor,
            [batch_size, h*2, w*2, self.n_classes], [1,2,2,1])
        print 'stream upscore', upscore.get_shape()

        ## Crop pool4_score to be same shape as upscore
        upscore_pool3_crop = tf.image.resize_image_with_crop_or_pad(upscore, pool3_h, pool3_w)
        print 'upscore_pool4_crop', upscore_pool3_crop.get_shape()

        ## Order invariant combination
        upscore = pool3_score + upscore_pool3_crop
        print 'upscore ', upscore.get_shape()

        ## Final upsample
        h, w = upscore.get_shape().as_list()[1:3]
        upscore = tf.nn.conv2d_transpose(upscore, output_weights_tensor,
            [batch_size, h*8, w*8, self.n_classes], [1,8,8,1]) ## 4 * 16 = 64

        ## Force to be the same size as input
        upscore = tf.image.resize_image_with_crop_or_pad(upscore, self.x_dim, self.y_dim)
        print 'fcn16s upscore', upscore.get_shape()
        return upscore
