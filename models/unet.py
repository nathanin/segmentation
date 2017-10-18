import tensorflow as tf
import numpy as np
import sys, time

slim = tf.contrib.slim

sys.path.insert(0, '.')
from basemodel import BaseModel


""" Implements U-Net (Ronneberger, et al 2015)

@inproceedings{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={234--241},
  year={2015},
  organization={Springer}
}

"""

class UNetModel(BaseModel):
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
        adversarial_training = False):

        ## TODO: Check args

        super(UNetModel, self).__init__(
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

        self.model_name = 'unet'
        self.IN_OUT_CROP = True
        print 'Setting up UNet model'

        ## Ops
        self._init_input()

        ## Custom things for this model
        self.n_kernels = n_kernels
        with tf.name_scope('UNet'):
            self.y_hat = self.model(self.input_x, reuse=False)
            target = self.y_hat.get_shape().as_list()[1]
            self.input_y = tf.image.resize_image_with_crop_or_pad(self.input_y, target, target)
            print 'crop_input_y', self.input_y.get_shape()

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

        ## Saver things; TODO add logic that skips the dataset load if we restore
        self._init_saver(self.model_name)


    def crop_tensor(self, tensor, target):
        h = target.shape[1]
        w = target.shape[2]
        h_ = tensor.shape[1]
        w_ = tensor.shape[2]
        print 'h:', h_, 'w:', w_, 'target H:',h, 'target W:', w
        # crop_fn = lambda x: tf.image.resize_image_with_crop_or_pad(x,h,w)
        # cropped_tensor = tf.map_fn(crop_fn, tensor)
        cropped_tensor = tf.image.resize_image_with_crop_or_pad(tensor, h, w)
        return cropped_tensor


    def model(self, input_op, reuse=False):
        with tf.name_scope('Unet') as scope:
            net = slim.convolution2d(input_op,
                num_outputs = self.n_kernels,
                kernel_size = 3,
                stride = 1,
                padding = 'VALID',
                scope = 'conv1_1',
                reuse=reuse)
            net1_2 = slim.convolution2d(net, self.n_kernels, 3, 1, padding='VALID', scope='conv1_2', reuse=reuse)

            net2_0 = slim.max_pool2d(net, 2, scope='pool1')
            net2_1 = slim.convolution2d(net2_0, self.n_kernels*2, 3, 1, padding='VALID', scope='conv2_1', reuse=reuse)
            net2_2 = slim.convolution2d(net2_1, self.n_kernels*2, 3, 1, padding='VALID', scope='conv2_2', reuse=reuse)

            net3_0 = slim.max_pool2d(net2_2, 2, scope='pool2')
            net3_1 = slim.convolution2d(net3_0, self.n_kernels*4, 3, 1, padding='VALID', scope='conv3_1', reuse=reuse)
            net3_2 = slim.convolution2d(net3_1, self.n_kernels*4, 3, 1, padding='VALID', scope='conv3_2', reuse=reuse)

            net4_0 = slim.max_pool2d(net3_2, 2, scope='pool3')
            net4_1 = slim.convolution2d(net4_0, self.n_kernels*8, 3, 1, padding='VALID', scope='conv4_1', reuse=reuse)
            net4_2 = slim.convolution2d(net4_1, self.n_kernels*8, 3, 1, padding='VALID', scope='conv4_2', reuse=reuse)

            net5_0 = slim.max_pool2d(net4_2, 2, scope='pool4')
            net5_1 = slim.convolution2d(net5_0, self.n_kernels*16, 3, 1, padding='VALID', scope='conv5_1', reuse=reuse)
            net5_2 = slim.convolution2d(net5_1, self.n_kernels*16, 3, 1, padding='VALID', scope='conv5_2', reuse=reuse)

            ## Upsample is by transpose deconvolution
            ## Features from downsampling pass are center-cropped & concatenated
            net6_0 = slim.convolution2d_transpose(net5_2, self.n_kernels*8, 2, 2, padding='VALID',scope='upconv1', reuse=reuse)
            target = net6_0.get_shape().as_list()[1]
            net4_2_crop = tf.image.resize_image_with_crop_or_pad(net4_2, target, target)
            net6_0 = tf.concat([net4_2_crop, net6_0], axis=-1, name='concat1')
            net6_1 = slim.convolution2d(net6_0, self.n_kernels*8, 3, 1, padding='VALID', scope='conv6_1', reuse=reuse)
            net6_2 = slim.convolution2d(net6_1, self.n_kernels*8, 3, 1, padding='VALID', scope='conv6_2', reuse=reuse)

            net7_0 = slim.convolution2d_transpose(net6_2, self.n_kernels*4, 2, 2, padding='VALID', scope='upconv2', reuse=reuse)
            target = net7_0.get_shape().as_list()[1]
            net3_2_crop = tf.image.resize_image_with_crop_or_pad(net3_2, target, target)
            net7_0 = tf.concat([net3_2_crop, net7_0], axis=-1, name='concat2')
            net7_1 = slim.convolution2d(net7_0, self.n_kernels*4, 3, 1, padding='VALID', scope='conv7_1', reuse=reuse)
            net7_2 = slim.convolution2d(net7_1, self.n_kernels*4, 3, 1, padding='VALID', scope='conv7_2', reuse=reuse)

            net8_0 = slim.convolution2d_transpose(net7_2, self.n_kernels*2, 2, 2, padding='VALID',scope='upconv3', reuse=reuse)
            target = net8_0.get_shape().as_list()[1]
            net2_2_crop = tf.image.resize_image_with_crop_or_pad(net2_2, target, target)
            net8_0 = tf.concat([net2_2_crop, net8_0], axis=-1, name='concat3')
            net8_1 = slim.convolution2d(net8_0, self.n_kernels*2, 3, 1, padding='VALID', scope='conv8_1', reuse=reuse)
            net8_2 = slim.convolution2d(net8_1, self.n_kernels*2, 3, 1, padding='VALID', scope='conv8_2', reuse=reuse)

            net9_0 = slim.convolution2d_transpose(net8_2, self.n_kernels, 2, 2, padding='VALID', scope='upconv4', reuse=reuse)
            target = net9_0.get_shape().as_list()[1]
            net1_2_crop = tf.image.resize_image_with_crop_or_pad(net1_2, target, target)
            net9_0 = tf.concat([net1_2_crop, net9_0], axis=-1, name='concat4')
            net9_1 = slim.convolution2d(net9_0, self.n_kernels, 3, 1, padding='VALID', scope='conv9_1', reuse=reuse)
            net9_2 = slim.convolution2d(net9_1, self.n_kernels, 3, 1, padding='VALID', scope='conv9_2', reuse=reuse)

            output = slim.convolution2d(net9_2, self.n_classes, 1, 1, padding='VALID', scope='output',
                activation_fn=None, reuse=reuse)
            print 'output', output.get_shape()

            ## Crop training mask to match
            if self.mode == 'TRAINING':
                target = output.get_shape().as_list()[1]
                self.input_y = tf.image.resize_image_with_crop_or_pad(self.input_y, target, target)
                print 'crop_input_y', self.input_y.get_shape()
        return output
