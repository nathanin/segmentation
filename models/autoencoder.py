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


class Autoencoder(BaseModel):
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
        denoising = False,
        variational = False,
        encoder_type = 'stacked'):


        ## TODO: Check args

        super(Autoencoder, self).__init__(
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
        self.variational = variational
        self.denoising = denoising
        self.encoder_type = encoder_type

        ## Weight for xentropy loss
        self.xent_alpha = 1.5

        with tf.name_scope('Autoencoder') as scope:
            print 'Instantiating net'
            self.y_hat = self.model(self.input_x, reuse=False)

        with tf.name_scope('output') as scope:
            self.y_hat_sig = tf.nn.sigmoid(self.y_hat)
            self.output = tf.expand_dims(tf.argmax(self.y_hat_sig, 3), -1)
            self.output = tf.cast(self.output, tf.float32)
        self.inference_ops = [self.y_hat_sig, self.output]

        if self.encoder_type == 'small':
            self._adversarial_net_fn = self._small_adversary

        ## Generics
        with tf.name_scope('loss') as scope:
            self._init_training_ops()

        with tf.name_scope('testing') as scope:
            self._init_testing()

        ## Ammend training and testing ops if we're doing variational mode
        if self.variational:
            self._add_variational_loss()
        #     self._make_testing_variational()

        ## Summary, saver & initialize
        self._init_summary_ops()
        self.init_op = tf.global_variables_initializer()
        self.sess.run([self.init_op])
        self._init_saver(self.model_name)




    """ Add the KL term to seg_loss_op """
    def _add_variational_loss(self):
        if self.adversarial_training:
            self.KLD_seg = tf.stop_gradient(self.KLD)
            self.seg_loss_op = tf.reduce_mean(
                self.xent_alpha*self.xentropy_loss_op + \
                self.adv_lambda*self.l_bce_fake_one + \
                self.KLD_seg)

            ## Split up KLD optimization
            self.KLD_optimizer = tf.train.AdamOptimizer(1e-5)
            self.KLD_train_op = self.KLD_optimizer.minimize(self.KLD)
            self.train_op_list.append(self.KLD_train_op)
        else:
            self.seg_loss_op = tf.reduce_mean(self.xentropy_loss_op + self.KLD)




    """ Add variational daydream op
    The new op should sample from normal distribution and compute y_hat
    """
    def _make_testing_variational():
        epsilon = tf.random_normal([self.batch_size, self.zed_dim], name='epsilon')
        pass



    def _small_adversary(self, tensor_in, reuse=False, training=True):
        adv_kernels = 8
        with tf.name_scope('Adversary') as scope:
            conv1_0 = slim.convolution2d(tensor_in, adv_kernels, 3, 2, padding='VALID', reuse=reuse, scope='adv_conv1_0')
            pool1 = slim.max_pool2d(conv1_0, 2, 2, scope='adv_pool1')
            conv1_1 = slim.convolution2d(pool1, adv_kernels*2, 3, 2, padding='VALID', reuse=reuse, scope='adv_conv1_1')

            flat = slim.flatten(conv1_1)
            decision = slim.fully_connected(flat, 128, reuse=reuse, scope='adv_decision_0')
            decision = slim.fully_connected(flat, 2, reuse=reuse, scope='adv_decision_1', activation_fn=None)

        print '\tconv1_0', conv1_0.get_shape()
        print '\tconv1_1', conv1_1.get_shape()
        print '\tflat', flat.get_shape()
        print '\tdecision', decision.get_shape()

        return decision




    """ Implement the autoencoder

    An autoencoder squishes the signal through some vector space, zed,
    which can be from a dimensionality of our choosing.

    Some cases add multiplicative gaussian noise to the input
    Some cases gray a region, and ask the model to fill it in based on
    context.
    These are for later.
    """
    def model(self, input_op, reuse=False, training=True):
        if self.encoder_type == 'stacked':
            self.zed = self.stacked_encoder(input_op, reuse=reuse, training=training)
        elif self.encoder_type == 'multiscale':
            self.zed = self.multiscale_encoder(input_op, reuse=reuse, training=training)
        elif self.encoder_type == 'small':
            self.zed = self.small_stacked_encoder(input_op, reuse=reuse, training=training)


        ## Get mean and std from z and sample from normal distribution
        if self.variational:
            self.mu = slim.fully_connected(self.zed, self.zed_dim, scope='mu', reuse=reuse,
                activation_fn=None)
            self.logvar = slim.fully_connected(self.zed, self.zed_dim, scope='logvar', reuse=reuse,
                activation_fn=None)
            ## Contratin logvar to positive and add a small epsilon
            self.logvar = tf.nn.softplus(self.logvar) + 1e-9

            epsilon = tf.random_normal(tf.shape(self.logvar), name='epsilon')
            self.zed = self.mu + tf.exp(0.5 * self.logvar) * epsilon

            self.KLD = -0.5 * tf.reduce_sum(1 + self.logvar - tf.pow(self.mu, 2) - tf.exp(self.logvar), reduction_indices=1)
            # self.KLD = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.exp(self.logvar) - self.logvar -1, reduction_indices=1)
            print '\tSetting up variational autoencoder'
            print '\tmu', self.mu.get_shape()
            print '\tlogvar', self.logvar.get_shape()
            print '\tzed', self.zed.get_shape()
            print '\tKLD', self.KLD.get_shape()

        if self.encoder_type == 'small':
            y_hat = self.small_decoder(self.zed, reuse=reuse, training=training)
        else:
            y_hat = self.decoder(self.zed, reuse=reuse, training=training)


        return y_hat


    """ Small, for MNIST """
    def small_stacked_encoder(self, input_op, reuse=False, training=True):
        print 'SMALL ENCODER'
        self.x_dim = input_op.get_shape().as_list()[1]
        self.y_dim = input_op.get_shape().as_list()[2]
        print '\tx_dim', self.x_dim
        print '\ty_dim', self.y_dim
        conv1_0 = slim.convolution2d(input_op,
            num_outputs = self.n_kernels,
            kernel_size = 5,
            stride = 1,
            padding = 'SAME',
            scope = 'conv1_0',
            reuse=reuse)
        conv1_0 = slim.batch_norm(conv1_0, scope='bn1', reuse=reuse, is_training=training)
        pool1 = slim.max_pool2d(conv1_0, 2, 2, scope='pool1')

        conv2_0 = slim.convolution2d(pool1, self.n_kernels*2, 5, 1, padding='VALID', scope='conv2_0',reuse=reuse)
        conv2_0 = slim.batch_norm(conv2_0, scope='bn2', reuse=reuse, is_training=training)
        if self.bayesian:
            conv2_0 = slim.dropout(conv2_0, scope='drop1')
        pool2 = slim.max_pool2d(conv2_0, 2, 2, scope='pool2')

        ## Squish through a vector space
        ## Store the shape
        self.conv_shape = pool2.get_shape().as_list()
        enc_flatten = slim.flatten(pool2, scope='enc_flatten')
        ## Store dimensionality
        self.pre_zed_dim = enc_flatten.get_shape().as_list()[-1]

        if self.variational:
            zed = slim.fully_connected(enc_flatten, self.zed_dim*2, scope='zed', reuse=reuse)
        else:
            zed = slim.fully_connected(enc_flatten, self.zed_dim, scope='zed', reuse=reuse,
                activation_fn=None, biases_initializer=None)

        print '\tconv1_0', conv1_0.get_shape()
        print '\tpool1', pool1.get_shape()
        print '\tconv2_0', conv2_0.get_shape()
        print '\tpool2', pool2.get_shape()
        print '\tflatten', enc_flatten.get_shape()
        print '\tzed', zed.get_shape()
        return zed





    def small_decoder(self, zed, reuse=False, training=True):
        print 'SMALL DECODER'
        ## Shape back
        inflate = slim.fully_connected(zed, self.pre_zed_dim, scope='inflate', reuse=reuse)
        inflate = tf.reshape(inflate, self.conv_shape)

        ## Decoding
        deconv1_0 = slim.convolution2d_transpose(inflate, self.n_kernels*2, 5, 2, padding='VALID', scope='deconv1_0', reuse=reuse)
        deconv1_1 = slim.convolution2d_transpose(deconv1_0, self.n_kernels*2, 5, 2, padding='VALID', scope='deconv1_1', reuse=reuse)
        deconv1_1 = slim.batch_norm(deconv1_1, scope='bn_up1', reuse=reuse, is_training=training)
        if self.bayesian:
            deconv1_1 = slim.dropout(deconv1_1, scope='drop2')

        upsample = tf.image.resize_image_with_crop_or_pad(deconv1_1, self.x_dim+2, self.y_dim+2)
        deconv_out = slim.convolution2d(upsample, self.input_channel, 3, 1, padding='VALID', scope='conv_out',
            reuse=reuse, activation_fn=None, biases_initializer=False)

        print '\tinflate', inflate.get_shape()
        print '\tdeconv1_0', deconv1_0.get_shape()
        print '\tdeconv1_1', deconv1_1.get_shape()
        print '\tupsample', upsample.get_shape()
        print '\tdeconv_out', deconv_out.get_shape()

        return deconv_out



    

    """ Stacked convolutions

    Encoder should return an encoded vector [batch_size, self.zed_dim]
    And define:
    conv_shape - the shape of the L-1 convolution layer
    pre_zed_dim - the shape of the flattened L-1 conv layer
        (pre_zed_dim = prod(conv_shape.shape[1:]))
    """
    def stacked_encoder(self, input_op, reuse=False, training=True):
        # self.batch_size = input_op.get_shape().as_list()[0]
        self.x_dim = input_op.get_shape().as_list()[1]
        self.y_dim = input_op.get_shape().as_list()[2]
        print '\tx_dim', self.x_dim
        print '\ty_dim', self.y_dim

        conv1_0 = slim.convolution2d(input_op,
            num_outputs = self.n_kernels,
            kernel_size = 5,
            stride = 2,
            padding = 'SAME',
            scope = 'conv1_0',
            reuse=reuse)
        conv1_0 = slim.batch_norm(conv1_0, scope='bn1', reuse=reuse, is_training=training)
        pool1 = slim.max_pool2d(conv1_0, 2, 2, scope='pool1')

        conv2_0 = slim.convolution2d(pool1, self.n_kernels*2, 3, 1, padding='VALID', scope='conv2_0',reuse=reuse)
        conv2_0 = slim.batch_norm(conv2_0, scope='bn2', reuse=reuse, is_training=training)
        if self.bayesian:
            conv2_0 = slim.dropout(conv2_0, scope='drop1')
        pool2 = slim.max_pool2d(conv2_0, 3, 3, scope='pool2')

        conv3_0 = slim.convolution2d(pool2, self.n_kernels*4, 3, 1, padding='VALID', scope='conv3_0',reuse=reuse)
        conv3_0 = slim.batch_norm(conv3_0, scope='bn3', reuse=reuse, is_training=training)
        pool3 = slim.max_pool2d(conv3_0, 3, 3, scope='pool3')

        conv4_0 = slim.convolution2d(conv3_0, self.n_kernels*8, 3, 1, padding='VALID', scope='conv4_0',reuse=reuse)
        conv4_0 = slim.batch_norm(conv4_0, scope='bn4', reuse=reuse, is_training=training)
        if self.bayesian:
            conv4_0 = slim.dropout(conv4_0, scope='drop1')

        ## Squish through a vector space
        ## Store the shape
        self.conv_shape = conv4_0.get_shape().as_list()
        enc_flatten = slim.flatten(conv4_0, scope='enc_flatten')
        ## Store dimensionality
        self.pre_zed_dim = enc_flatten.get_shape().as_list()[-1]

        if self.variational:
            zed = slim.fully_connected(enc_flatten, self.zed_dim*2, scope='zed', reuse=reuse)
        else:
            zed = slim.fully_connected(enc_flatten, self.zed_dim, scope='zed', reuse=reuse,
                activation_fn=None, biases_initializer=None)

        print '\tconv1_0', conv1_0.get_shape()
        print '\tpool1', pool1.get_shape()
        print '\tconv2_0', conv2_0.get_shape()
        print '\tpool2', pool2.get_shape()
        print '\tconv3_0', conv3_0.get_shape()
        print '\tpool3', pool3.get_shape()
        print '\tconv4_0', conv4_0.get_shape()
        print '\tflatten', enc_flatten.get_shape()
        print '\tzed', zed.get_shape()
        return zed





    """ Multiple scales

    Like FCN-8s, or U-Net, message passing from early convolutions downward
    Combination by reshape & concat
    """
    def multiscale_encoder(self, input_op, reuse=False, training=True):
        ## Scale 1
        print '\tMULTISCALE ENCODER'
        conv1_0 = slim.convolution2d(input_op,
            num_outputs = self.n_kernels,
            kernel_size = 3,
            stride = 2,
            padding = 'SAME',
            scope = 'conv1_0',
            reuse=reuse)
        conv1_1 = slim.convolution2d(conv1_0, self.n_kernels, 3, 2, padding='SAME', scope='conv1_1',reuse=reuse)
        # if self.bayesian:
        #     conv1_2 = slim.dropout(conv1_2, scope='drop1')
        conv1_2 = slim.convolution2d(conv1_1, self.n_kernels, 3, 1, padding='SAME', scope='conv1_2',reuse=reuse)
        conv1_2 = slim.batch_norm(conv1_2, scope='bn1', reuse=reuse, is_training=training)
        pool1 = slim.max_pool2d(conv1_2, 3, 3, scope='pool1')

        ## Make zed an image
        zed_1 = slim.flatten(pool1, scope='flat1')
        zed_1 = slim.fully_connected(zed_1, self.zed_dim, scope='zed_1', reuse=reuse,
            activation_fn=tf.nn.sigmoid)
        zed_1 = tf.reshape(zed_1, [self.batch_size, 8, 8, 1])


        conv2_0 = slim.convolution2d(pool1, self.n_kernels*2, 3, 2, padding='VALID', scope='conv2_0',reuse=reuse)
        conv2_1 = slim.convolution2d(conv2_0, self.n_kernels*2, 3, 1, padding='SAME', scope='conv2_1',reuse=reuse)
        conv2_2 = slim.convolution2d(conv2_1, self.n_kernels*2, 3, 1, padding='SAME', scope='conv2_2',reuse=reuse)
        conv2_2 = slim.batch_norm(conv2_2, scope='bn2', reuse=reuse, is_training=training)
        # if self.bayesian:
        #     net = slim.dropout(net, scope='drop1')

        ## Make zed an image
        zed_2 = slim.flatten(conv2_2, scope='flat2')
        zed_2 = slim.fully_connected(zed_2, self.zed_dim, scope='zed_2', reuse=reuse,
            activation_fn=tf.nn.sigmoid)
        zed_2 = tf.reshape(zed_2, [self.batch_size, 8, 8, 1])

        conv3_0 = slim.convolution2d(conv2_2, self.n_kernels*4, 3, 1, padding='VALID', scope='conv3_0',reuse=reuse)
        # if self.bayesian:
        #     net = slim.dropout(net, scope='drop1')
        conv3_1 = slim.convolution2d(conv3_0, self.n_kernels*4, 3, 1, padding='SAME', scope='conv3_1',reuse=reuse)
        conv3_2 = slim.convolution2d(conv3_1, self.n_kernels*4, 3, 1, padding='SAME', scope='conv3_2',reuse=reuse)
        conv3_2 = slim.batch_norm(conv3_2, scope='bn3', reuse=reuse, is_training=training)

        zed_3 = slim.flatten(conv3_2, scope='flat2')
        zed_3 = slim.fully_connected(zed_3, self.zed_dim, scope='zed_3', reuse=reuse,
            activation_fn=tf.nn.sigmoid)
        zed_3 = tf.reshape(zed_3, [self.batch_size, 8, 8, 1])

        ## zed is 4D: [batch, sqrt(z_dim), sqrt(z_dim), 3]
        zed_concat = tf.concat([zed_1, zed_2, zed_3], axis=-1)
        zed = slim.convolution2d(zed_concat, 1, 3, 1, padding='SAME', scope='zed_conv', reuse=reuse)
        self.conv_shape = zed.get_shape().as_list()
        zed = slim.flatten(zed)
        ## pre_zed_dim = 2*sqrt(z_dim) * 3
        self.pre_zed_dim = zed.get_shape().as_list()[-1]

        if self.variational:
            zed = slim.fully_connected(zed, self.zed_dim*2, scope='zed', reuse=reuse)
        else:
            zed = slim.fully_connected(zed, self.zed_dim, scope='zed', reuse=reuse,
                activation_fn=None, biases_initializer=None)

        print '\tconv1_0', conv1_0.get_shape()
        print '\tconv1_1', conv1_1.get_shape()
        print '\tconv1_2', conv1_2.get_shape()
        print '\tzed_1', zed_1.get_shape()
        print '\tconv2_0', conv2_0.get_shape()
        print '\tconv2_1', conv2_1.get_shape()
        print '\tconv2_2', conv2_2.get_shape()
        print '\tzed_2', zed_2.get_shape()
        print '\tconv3_0', conv3_0.get_shape()
        print '\tconv3_1', conv3_1.get_shape()
        print '\tconv3_2', conv3_2.get_shape()
        print '\tzed_concat', zed_concat.get_shape()
        print '\tzed', zed.get_shape()

        return zed




    def decoder(self, zed, reuse=False, training=True):
        ## Shape back
        inflate = slim.fully_connected(zed, self.pre_zed_dim, scope='inflate', reuse=reuse)
        inflate = tf.reshape(inflate, self.conv_shape)

        ## Decoding
        deconv1_0 = slim.convolution2d_transpose(inflate, self.n_kernels*2, 5, 2, padding='VALID', scope='deconv1_0', reuse=reuse)
        deconv1_0 = slim.batch_norm(deconv1_0, scope='bn5', reuse=reuse, is_training=training)
        if self.bayesian:
            deconv1_0 = slim.dropout(deconv1_0, scope='drop2')

        deconv2_0 = slim.convolution2d_transpose(deconv1_0, self.n_kernels, 5, 2, padding='VALID', scope='deconv2_0', reuse=reuse)
        deconv2_0 = slim.batch_norm(deconv2_0, scope='bn6', reuse=reuse, is_training=training)
        deconv2_1 = slim.convolution2d_transpose(deconv2_0, self.n_kernels, 5, 2, padding='VALID', scope='deconv2_1', reuse=reuse)
        deconv2_1 = slim.batch_norm(deconv2_1, scope='bn7', reuse=reuse, is_training=training)
        deconv2_1 = tf.image.resize_bilinear(deconv2_1, [self.x_dim//2, self.y_dim//2])

        deconv3_0 = slim.convolution2d_transpose(deconv2_1, self.n_classes, 2, 2, padding='VALID', scope='deconv3_0', reuse=reuse)
        deconv3_0 = slim.batch_norm(deconv3_0, scope='bn8', reuse=reuse, is_training=training)
        ## Force to be the same size as input, if it's off by one
        deconv3_0 = tf.image.resize_image_with_crop_or_pad(deconv3_0, self.x_dim, self.y_dim)

        conv_out = slim.convolution2d(deconv3_0, self.n_classes, 3, 1, padding='SAME', scope='conv_out', reuse=reuse,
            activation_fn=None, biases_initializer=None)

        print '\tinflate', inflate.get_shape()
        print '\tdeconv1_0', deconv1_0.get_shape()
        print '\tdeconv2_0', deconv2_0.get_shape()
        print '\tdeconv2_1', deconv2_1.get_shape()
        print '\tdeconv2_0', deconv2_0.get_shape()
        print '\tdeconv2_1', deconv2_1.get_shape()
        print '\tdeconv3_0', deconv3_0.get_shape()
        print '\tconv_output', conv_out.get_shape()

        return conv_out
