import tensorflow as tf
import numpy as np
import sys, time, os

slim = tf.contrib.slim

sys.path.insert(0, '.')
from basemodel import BaseModel

""" Generative Adversarial Network class

It should implement a method model()

Variational mode:
model() should pass the signal through a vector space, zed
Then we replace the loss ops, to add the KL divergence
of zed with Gaussian normal, and minimize the whole thing.


We split model() into encoder() and generator()
encoder is the one that produces zed
generator is the one that generates a new image from
sampled zed


Then, call model() that sets the attribute y_hat:

y_hat = model(...)

Model should have method for reuse
def model(input, reuse=False):
    ...


Also: there has to be a better option for message passing arguments


It's like conv/deconv except with batch norm and also more layers

TODO move self.x_dim and self.y_dim assignment to constructor
"""

class GAN(BaseModel):
    def __init__(self,
        sess = None,
        n_classes = 2,
        log_dir = None,
        dataset = None,
        save_dir = None,
        bayesian = False,
        input_dims = 256,
        mode = 'TRAINING',
        input_channel = 3,
        test_dataset = None,
        learning_rate = 1e-4,
        load_snapshot = None,
        load_snapshot_from = None,
        n_kernels = 32,
        autoencoder = False,  ## Usually true
        adversarial_training = True,  ## Always true
        zed_dim = 64):


        ## TODO: Check args

        super(GAN, self).__init__(
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

        self.model_name = 'GAN'
        print 'Setting up Generative Adversarial Network model'

        ## Ops
        self._init_input()

        ## Custom things for this model
        self.n_kernels = n_kernels
        with tf.name_scope('ConvDeconv') as scope:
            print 'Instantiating GAN net'
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


    """ Implements a GAN """

    def model(self, input_op, reuse=False):
        self.zed = self.encoder(input_op, reuse=reuse)

        ## Sample from N(0, 1), conditioned on zed

        y_hat = self.generator(self.zed_sample, reuse=reuse)

        return y_hat

    def encoder(self, input_op, reuse=False):
        # self.batch_size = self.input_x.get_shape()[0]
        # x_dim = self.input_x.get_shape().as_list()[1]
        # y_dim = self.input_x.get_shape().as_list()[2]
        self.x_dim = input_op.get_shape().as_list()[1]
        self.y_dim = input_op.get_shape().as_list()[2]
        print '\tx_dim', x_dim
        print '\ty_dim', y_dim
        net = slim.convolution2d(input_op,
            num_outputs = self.n_kernels,
            kernel_size = 5,
            stride = 1,
            padding = 'SAME',
            scope = 'conv1_0',
            reuse=reuse)
        net = slim.batch_norm(net, scope='bn1', reuse=reuse)
        print '\tconv1_0', net.get_shape()
        net = slim.convolution2d(net, self.n_kernels, 3, 2, padding='VALID', scope='conv1_1',reuse=reuse)
        net = slim.batch_norm(net, scope='bn2', reuse=reuse)
        print '\tconv1_1', net.get_shape()

        net = slim.convolution2d(net, self.n_kernels*2, 3, 1, padding='VALID', scope='conv2_0',reuse=reuse)
        print '\tconv2_0', net.get_shape()
        net = slim.convolution2d(net, self.n_kernels*2, 3, 2, padding='VALID', scope='conv2_1',reuse=reuse)
        net = slim.batch_norm(net, scope='bn3', reuse=reuse)
        print '\tconv2_1', net.get_shape()
        if self.bayesian:
            net = slim.dropout(net, scope='drop1')
        net = slim.max_pool2d(net, 3, 3, scope='pool3')
        print '\tpool3', net.get_shape()

        net = slim.convolution2d(net, self.n_kernels*4, 3, 1, padding='VALID', scope='conv3_0',reuse=reuse)
        print '\tconv3_0', net.get_shape()
        net = slim.convolution2d(net, self.n_kernels*4, 3, 2, padding='VALID', scope='conv3_1',reuse=reuse)
        net = slim.batch_norm(net, scope='bn4', reuse=reuse)
        print '\tconv3_1', net.get_shape()
        if self.bayesian:
            net = slim.dropout(net, scope='drop1')

        net = slim.convolution2d(net, self.n_kernels*8, 3, 1, padding='VALID', scope='conv4_0',reuse=reuse)
        net = slim.batch_norm(net, scope='bn5', reuse=reuse)
        print '\tconv4_0', net.get_shape()


        """ We can be creative with the shape of zed

        It can be a vector, 2-D, or 3-D image with as many kernels as we want.
        It can be an exterior random variable if we condition zed to be useful
        for e.g. some external classification.
        """
        zed = slim.flatten(net, scope='gan_zed')

        return zed


        


    def generator(self, zed, reuse=False):
        net = slim.convolution2d_transpose(zed, self.n_kernels*2, 5, 3, padding='VALID', scope='deconv1_0', reuse=reuse)
        net = slim.batch_norm(net, scope='bn6', reuse=reuse)
        print '\tdeconv1_0', net.get_shape()
        if self.bayesian:
            net = slim.dropout(net, scope='drop2')

        net = slim.convolution2d_transpose(net, self.n_kernels, 5, 2, padding='VALID', scope='deconv2_0', reuse=reuse)
        print '\tdeconv2_0', net.get_shape()
        net = slim.convolution2d_transpose(net, self.n_kernels, 5, 2, padding='VALID', scope='deconv2_1', reuse=reuse)
        net = slim.batch_norm(net, scope='bn7', reuse=reuse)
        print '\tdeconv2_1', net.get_shape()
        ## Set to 1/2 input size for 2x upconv
        net = tf.image.resize_bilinear(net, [self.x_dim//2, self.y_dim//2])
        print '\tresize', net.get_shape()

        net = slim.convolution2d_transpose(net, self.n_classes, 2, 2, padding='VALID', scope='deconv3_0', reuse=reuse)
        net = slim.batch_norm(net, scope='bn8', reuse=reuse)
        print '\tdeconv3_0', net.get_shape()

        ## Force to be the same size as input, if it's off by one
        net = tf.image.resize_image_with_crop_or_pad(net, self.x_dim, self.y_dim)
        print '\tforce_resize', net.get_shape()

        net = slim.convolution2d(net, self.n_classes, 3, 1, padding='SAME', scope='conv_out', reuse=reuse,
            activation_fn=None)
        print '\tconv_output', net.get_shape()

        return net
