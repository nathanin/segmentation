import tensorflow as tf
import numpy as np
import sys, time, os

slim = tf.contrib.slim

sys.path.insert(0, '.')
from basemodel import BaseModel

""" Autoencoder Model class

We have two options:
    move the special autoencoder options into this file
    keep as-is with the if-case in the BaseModel


actually the good, proper, OOP solution is to factor like this:
But it only saves room with some smarter message passing structure

class Autoencoder(BaseModel):
    --- stuff

class StackedAutoencoder(Autoencoder):
    --- model()

class MultiScaleAutoencoder(Autoencoder):
    --- model()

"""


class StackedAutoencoder(BaseModel):
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
        autoencoder = True,
        adversarial_training = True,
        zed_dim = 64):


        ## TODO: Check args

        super(StackedAutoencoder, self).__init__(
            sess=sess,
            mode=mode,
            log_dir=log_dir,
            dataset=dataset,
            bayesian=bayesian,
            save_dir=save_dir,
            n_classes=n_classes,
            input_dims=input_dims,
            autoencoder=autoencoder,
            test_dataset=test_dataset,
            input_channel=input_channel,
            load_snapshot=load_snapshot,
            learning_rate=learning_rate,
            load_snapshot_from=load_snapshot_from,
            adversarial_training=adversarial_training)

        self.model_name = 'autoencoder'
        print 'Setting up conv / deconv autoencoder model'

        ## Ops
        self._init_input()

        ## Custom things for this model
        self.n_kernels = n_kernels
        self.zed_dim = zed_dim
        with tf.name_scope('Autoencoder') as scope:
            print 'Instantiating net'
            self.y_hat = self.model(self.input_x, reuse=False)

        with tf.name_scope('output') as scope:
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

        ## Saver last so all the variables exist
        self._init_saver(self.model_name)


    """ Implement the autoencoder

    An autoencoder squishes the signal through some vector space, zed,
    which can be from a dimensionality of our choosing.

    Some cases add multiplicative gaussian noise to the input
    Some cases gray a region, and ask the model to fill it in based on
    context.
    These are for later.
    """
    def model(self, input_op, reuse=False, training=True):
        x_dim = input_op.get_shape().as_list()[1]
        y_dim = input_op.get_shape().as_list()[2]
        print '\tx_dim', x_dim
        print '\ty_dim', y_dim
        net = slim.convolution2d(input_op,
            num_outputs = self.n_kernels,
            kernel_size = 5,
            stride = 2,
            padding = 'SAME',
            scope = 'conv1_0',
            reuse=reuse)
        net = slim.batch_norm(net, scope='bn1', reuse=reuse, is_training=training)
        print '\tconv1_0', net.get_shape()
        net = slim.max_pool2d(net, 2, 2, scope='pool1')
        print '\tpool1', net.get_shape()

        net = slim.convolution2d(net, self.n_kernels*2, 3, 1, padding='VALID', scope='conv2_0',reuse=reuse)
        net = slim.batch_norm(net, scope='bn2', reuse=reuse, is_training=training)
        print '\tconv2_0', net.get_shape()
        if self.bayesian:
            net = slim.dropout(net, scope='drop1')
        net = slim.max_pool2d(net, 3, 3, scope='pool2')
        print '\tpool2', net.get_shape()

        net = slim.convolution2d(net, self.n_kernels*4, 3, 1, padding='VALID', scope='conv3_0',reuse=reuse)
        net = slim.batch_norm(net, scope='bn3', reuse=reuse, is_training=training)
        print '\tconv3_0', net.get_shape()
        net = slim.max_pool2d(net, 3, 3, scope='pool3')
        print '\tpool3', net.get_shape()

        net = slim.convolution2d(net, self.n_kernels*8, 3, 1, padding='VALID', scope='conv4_0',reuse=reuse)
        net = slim.batch_norm(net, scope='bn4', reuse=reuse, is_training=training)
        if self.bayesian:
            net = slim.dropout(net, scope='drop1')
        print '\tconv4_0', net.get_shape()

        ## Squish through a vector space
        ## Store the shape
        conv_shape = net.get_shape().as_list()
        net = slim.flatten(net, scope='enc_flatten')
        print '\tflatten', net.get_shape()
        ## Store dimensionality
        pre_zed_dim = net.get_shape().as_list()[-1]
        self.zed = slim.fully_connected(net, self.zed_dim, scope='zed', reuse=reuse, activation_fn=None)
        print '\tzed', self.zed.get_shape()

        ## Shape back
        net = slim.fully_connected(self.zed, pre_zed_dim, scope='inflate', reuse=reuse)
        print '\tinflate', net.get_shape()
        net = tf.reshape(net, conv_shape)

        ## Decoding
        net = slim.convolution2d_transpose(net, self.n_kernels*2, 5, 2, padding='VALID', scope='deconv1_0', reuse=reuse)
        net = slim.batch_norm(net, scope='bn5', reuse=reuse, is_training=training)
        print '\tdeconv1_0', net.get_shape()
        if self.bayesian:
            net = slim.dropout(net, scope='drop2')

        net = slim.convolution2d_transpose(net, self.n_kernels, 5, 2, padding='VALID', scope='deconv2_0', reuse=reuse)
        net = slim.batch_norm(net, scope='bn6', reuse=reuse, is_training=training)
        print '\tdeconv2_0', net.get_shape()
        net = slim.convolution2d_transpose(net, self.n_kernels, 5, 2, padding='VALID', scope='deconv2_1', reuse=reuse)
        net = slim.batch_norm(net, scope='bn7', reuse=reuse, is_training=training)
        print '\tdeconv2_1', net.get_shape()

        net = tf.image.resize_bilinear(net, [x_dim//2, y_dim//2])
        print '\tresize', net.get_shape()

        net = slim.convolution2d_transpose(net, self.n_classes, 2, 2, padding='VALID', scope='deconv3_0', reuse=reuse)
        net = slim.batch_norm(net, scope='bn8', reuse=reuse, is_training=training)
        print '\tdeconv3_0', net.get_shape()

        ## Force to be the same size as input, if it's off by one
        net = tf.image.resize_image_with_crop_or_pad(net, x_dim, y_dim)
        print '\tforce_resize', net.get_shape()

        net = slim.convolution2d(net, self.n_classes, 3, 1, padding='SAME', scope='conv_out', reuse=reuse,
            activation_fn=None)
        print '\tconv_output', net.get_shape()

        return net




class MultiScaleAutoencoder(BaseModel):
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
        autoencoder = True,
        adversarial_training = True,
        zed_dim = 64,
        denoising = False):


        ## TODO: Check args

        super(MultiScaleAutoencoder, self).__init__(
            sess=sess,
            mode=mode,
            log_dir=log_dir,
            dataset=dataset,
            bayesian=bayesian,
            save_dir=save_dir,
            n_classes=n_classes,
            input_dims=input_dims,
            autoencoder=autoencoder,
            test_dataset=test_dataset,
            input_channel=input_channel,
            load_snapshot=load_snapshot,
            learning_rate=learning_rate,
            load_snapshot_from=load_snapshot_from,
            adversarial_training=adversarial_training)

        self.model_name = 'multiscale AE'
        print 'Setting up multi-scale autoencoder model'

        ## Ops
        self._init_input()

        ## Custom things for this model
        self.n_kernels = n_kernels
        self.zed_dim = zed_dim
        with tf.name_scope('ConvDeconv') as scope:
            print 'Instantiating net'
            self.y_hat = self.model(self.input_x, reuse=False)

        with tf.name_scope('output') as scope:
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

        ## Saver last so all the variables exist
        ## Saver things; TODO add logic that skips the dataset load if we restore
        self._init_saver(self.model_name)


    """ Implement the model

    Multi scale version convolves the image at multiple scales,
    then A) concat or B) sums them into zed.

    Deconvolution proceeds from zed as a single stack.
    -- can just try something else too. I mean, we're pretty free here.

    Some cases add multiplicative gaussian noise to the input
    Some cases gray a region, and ask the model to fill it in based on
    context.
    These are for later.
    """
    def model(self, input_op, reuse=False, training=True):
        self.x_dim = input_op.get_shape().as_list()[1]
        self.y_dim = input_op.get_shape().as_list()[2]
        print '\tx_dim', self.x_dim
        print '\ty_dim', self.y_dim

        with tf.name_scope('Multi_Encode') as scope:
            self.zed = self.encoder(input_op, reuse=reuse, training=training)

        with tf.name_scope('Multi_Decode') as scope:
            y_hat = self.decoder(self.zed, reuse=reuse, training=training)

        return y_hat


    def encoder(self, input_op, reuse=False, training=True):
        ## Scale 1
        conv1_0 = slim.convolution2d(input_op,
            num_outputs = self.n_kernels,
            kernel_size = 3,
            stride = 2,
            padding = 'SAME',
            scope = 'conv1_0',
            reuse=reuse)
        print '\tconv1_0', conv1_0.get_shape()
        conv1_1 = slim.convolution2d(conv1_0, self.n_kernels, 3, 2, padding='SAME', scope='conv1_1',reuse=reuse)
        # if self.bayesian:
        #     conv1_2 = slim.dropout(conv1_2, scope='drop1')
        print '\tconv1_1', conv1_1.get_shape()
        conv1_2 = slim.convolution2d(conv1_1, self.n_kernels, 3, 1, padding='SAME', scope='conv1_2',reuse=reuse)
        print '\tconv1_2', conv1_2.get_shape()
        conv1_2 = slim.batch_norm(conv1_2, scope='bn1', reuse=reuse, is_training=training)
        pool1 = slim.max_pool2d(conv1_2, 3, 3, scope='pool1')

        zed_1 = slim.flatten(pool1, scope='flat1')
        zed_1 = slim.fully_connected(zed_1, self.zed_dim, scope='zed_1', reuse=reuse)
        print '\tzed_1', zed_1.get_shape()


        conv2_0 = slim.convolution2d(pool1, self.n_kernels*2, 3, 2, padding='VALID', scope='conv2_0',reuse=reuse)
        print '\tconv2_0', conv2_0.get_shape()
        conv2_1 = slim.convolution2d(conv2_0, self.n_kernels*2, 3, 1, padding='SAME', scope='conv2_1',reuse=reuse)
        print '\tconv2_1', conv2_1.get_shape()
        conv2_2 = slim.convolution2d(conv2_1, self.n_kernels*2, 3, 1, padding='SAME', scope='conv2_2',reuse=reuse)
        print '\tconv2_2', conv2_2.get_shape()
        conv2_2 = slim.batch_norm(conv2_2, scope='bn2', reuse=reuse, is_training=training)
        # if self.bayesian:
        #     net = slim.dropout(net, scope='drop1')
        # pool2 = slim.max_pool2d(conv2_2, 3, 3, scope='pool2')
        # print '\tpool2', pool2.get_shape()

        zed_2 = slim.flatten(conv2_2, scope='flat2')
        zed_2 = slim.fully_connected(zed_2, self.zed_dim, scope='zed_2', reuse=reuse)
        print '\tzed_2', zed_2.get_shape()

        conv3_0 = slim.convolution2d(conv2_2, self.n_kernels*4, 3, 1, padding='VALID', scope='conv3_0',reuse=reuse)
        # if self.bayesian:
        #     net = slim.dropout(net, scope='drop1')
        print '\tconv3_0', conv3_0.get_shape()
        conv3_1 = slim.convolution2d(conv3_0, self.n_kernels*4, 3, 1, padding='SAME', scope='conv3_1',reuse=reuse)
        print '\tconv3_1', conv3_1.get_shape()
        conv3_2 = slim.convolution2d(conv3_1, self.n_kernels*4, 3, 1, padding='SAME', scope='conv3_2',reuse=reuse)
        print '\tconv3_2', conv3_2.get_shape()
        conv3_2 = slim.batch_norm(conv3_2, scope='bn3', reuse=reuse, is_training=training)
        # pool3 = slim.max_pool2d(conv3_2, 3, 3, scope='pool3')
        # print '\tpool3', pool3.get_shape()

        zed_3 = slim.flatten(conv3_2, scope='flat2')
        zed_3 = slim.fully_connected(zed_3, self.zed_dim, scope='zed_3', reuse=reuse)

        zed = tf.concat([zed_1, zed_2, zed_3], axis=-1)
        print '\tzed_concat', zed.get_shape()
        zed = slim.fully_connected(zed, self.zed_dim, scope='zed', reuse=reuse, activation_fn=None)
        print '\tzed', zed.get_shape()

        return zed



    def decoder(self, zed, reuse=False, training=True):
        ## Decoding
        ## Reshape zed
        ## Use zed as a number of kernels
        zed = tf.expand_dims(zed, axis=1)
        zed = tf.expand_dims(zed, axis=2)
        print '\tzed_expand', zed.get_shape()

        zed = slim.convolution2d_transpose(zed, self.zed_dim, 7, 1, padding='VALID', scope='zed_upsample', reuse=reuse)
        print '\tzed_upsample', zed.get_shape()

        net = slim.convolution2d_transpose(zed, self.n_kernels*2, 5, 2, padding='VALID', scope='deconv1_0', reuse=reuse)
        net = slim.batch_norm(net, scope='bn5', reuse=reuse, is_training=training)
        print '\tdeconv1_0', net.get_shape()
        if self.bayesian:
            net = slim.dropout(net, scope='drop3')

        net = slim.convolution2d_transpose(net, self.n_kernels, 5, 2, padding='VALID', scope='deconv2_0', reuse=reuse)
        net = slim.batch_norm(net, scope='bn6', reuse=reuse, is_training=training)
        print '\tdeconv2_0', net.get_shape()
        net = slim.convolution2d_transpose(net, self.n_kernels, 5, 2, padding='VALID', scope='deconv2_1', reuse=reuse)
        net = slim.batch_norm(net, scope='bn7', reuse=reuse, is_training=training)
        print '\tdeconv2_1', net.get_shape()

        net = tf.image.resize_bilinear(net, [self.x_dim//2, self.y_dim//2])
        print '\tresize', net.get_shape()

        net = slim.convolution2d_transpose(net, self.n_classes, 2, 2, padding='VALID', scope='deconv3_0', reuse=reuse)
        net = slim.batch_norm(net, scope='bn8', reuse=reuse, is_training=training)
        print '\tdeconv3_0', net.get_shape()

        ## Force to be the same size as input, if it's off by one
        net = tf.image.resize_image_with_crop_or_pad(net, self.x_dim, self.y_dim)
        print '\tforce_resize', net.get_shape()

        net = slim.convolution2d(net, self.n_classes, 3, 1, padding='SAME', scope='conv_out', reuse=reuse,
            activation_fn=None)
        print '\tconv_output', net.get_shape()

        return net
