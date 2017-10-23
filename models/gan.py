import tensorflow as tf
import numpy as np
import sys, time, os

slim = tf.contrib.slim

sys.path.insert(0, '.')
from basemodel import BaseModel

""" Generative Adversarial Network class

It should implement a method model() and define y_hat,
along with special losses and overload attributes for training
to follow the GAN training algo.
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
        learning_rate = 1e-4,
        load_snapshot = None,
        load_snapshot_from = None,
        n_kernels = 32,
        autoencoder = True,  ## True just for input reasons; don't really need
        adversarial_training = True,  ## Always true
        zed_dim = 64,
        gan_type = 'small'):


        ## TODO: Check args
        assert adversarial_training == True

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
        self.zed_dim = zed_dim
        self.gan_type = gan_type


        if self.gan_type == 'small':
            self._generator_fn = self._generator_small
            self._adversarial_net_fn = self._descriminator_small
        else:
            self._generator_fn = self._generator
            self._adversarial_net_fn = self._descriminator


        with tf.name_scope('ConvDeconv') as scope:
            print 'Instantiating GAN'
            self.y_hat = self.model(reuse=False)

        ## Generics
        with tf.name_scope('loss') as scope:
            self._init_training_ops()

        ## 'dreaming' instead of 'testing'
        with tf.name_scope('dreaming') as scope:
            self._init_dream_ops()


        self._init_summary_ops()

        self.init_op = tf.global_variables_initializer()
        self.sess.run([self.init_op])

        ## Saver last so all the variables exist
        ## Saver things; TODO add logic that skips the dataset load if we restore
        self._init_saver(self.model_name)



    """ Overloading from BaseModel """
    def _init_summary_ops(self):
        self.y_hat_summary = tf.summary.image('y_hat', self.y_hat, max_outputs=4)
        self.x_real_summary = tf.summary.image('y_real', self.input_y, max_outputs=4)
        self.summary_op = tf.summary.merge([self.loss_summary,
            self.bce_real_summary,
            self.bce_fake_summary,
            self.gen_loss_summary,
            self.y_hat_summary,
            self.x_real_summary])
        self.summary_op = tf.Print(self.summary_op, ['Writing summary', self.global_step])


    """ Initialize the training ops
    creates self.train_op_list with a list of training ops
    we need one op that updates the generator
    and one op that updates the descriminator

    train_step() calls: self.train_op_list, and self.adversarial_train_list

    """
    def _init_training_ops(self):
        print 'Setting up GAN training'

        with tf.variable_scope('Adversarial') as scope:
            print 'Adversarial x ~ p(data)'
            self.real_adv = self._adversarial_net_fn(self.input_y, reuse=False)

            print 'Adversarial x ~ p(z)'
            ## Update G(z) only
            self.fake_adv = self._adversarial_net_fn(self.y_hat, reuse=True)
            ## Stop the gradient from updating G(z) when we just want to use G(z) to generate some fakes
            self.y_hat_nograd = tf.stop_gradient(self.y_hat)
            # self.fake_adv_nograd = self._adversarial_net_fn(self.y_hat_nograd, reuse=True)


        ## Separate variables for the optimizers to use - no more tf.stop_gradient !!
        ## https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'dis_' in var.name]
        self.g_vars = [var for var in t_vars if 'gen_' in var.name]

        print self.d_vars
        print self.g_vars

        ## seg_optimizer used to train the generator
        self.gen_optimizer = tf.train.AdamOptimizer(self.learning_rate, name='genAdam')
        self.adversarial_optimizer = tf.train.AdamOptimizer(self.adversarial_lr, name='descAdam')

        ## Trainig objectives for real and fake images
        self.real_ex = tf.one_hot(tf.ones_like(tf.argmax(self.real_adv, 1)), 2)
        self.fake_ex = tf.one_hot(tf.zeros_like(tf.argmax(self.fake_adv, 1)), 2)

        ## Real should all be real
        ## E_{x ~ p(data)}[log D(x)]
        self.l_bce_real = tf.nn.softmax_cross_entropy_with_logits(labels=self.real_ex, logits=self.real_adv)

        ## Fakes should all be fake
        ## E_{x ~ p(z)}[log 1 - D(G(z))]
        self.l_bce_fake = tf.nn.softmax_cross_entropy_with_logits(labels=self.fake_ex, logits=self.fake_adv)
        # self.l_bce_fake_nograd = tf.nn.softmax_cross_entropy_with_logits(labels=self.fake_ex, logits=self.fake_adv_nograd)

        ## Losses are the same; but gradients should flow differently
        ## adv_loss should only flow back to update D(~)
        self.adv_loss_op = tf.reduce_mean(self.l_bce_real + self.l_bce_fake)
        ## gen_loss should only flow back to update G(~)
        self.gen_loss_op = tf.reduce_mean(self.l_bce_fake)

        self.adv_loss_op = tf.Print(self.adv_loss_op, ['Adv Loss', self.adv_loss_op, self.global_step])
        self.gen_loss_op = tf.Print(self.gen_loss_op, ['Gen Loss', self.gen_loss_op, self.global_step])

        ## ?? slim documentation says to do this for batch_norm layers
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.gen_train_op = self.gen_optimizer.minimize(self.gen_loss_op, var_list=self.g_vars)
            self.adv_train_op = self.adversarial_optimizer.minimize(self.adv_loss_op, var_list=self.d_vars)


        self.loss_summary = tf.summary.scalar('loss', self.adv_loss_op)
        self.train_op_list = [self.gen_train_op,
                              self.gs_increment]
        self.adversarial_train_list = [self.adv_train_op]

        self.bce_real_summary = tf.summary.scalar('l_bce_real', tf.reduce_mean(self.l_bce_real))
        self.bce_fake_summary = tf.summary.scalar('l_bce_fake', tf.reduce_mean(self.l_bce_fake))
        self.gen_loss_summary = tf.summary.scalar('gen_loss', self.gen_loss_op)
        # self.train_op_list = [self.train_op, self.gs_increment]





    """ Dream is to generate z and pass through the generator

    Sample: x ~ G(z ~ N(0,1))
    """
    def _init_dream_ops(self):
        print 'Initializing dream ops'
        zed_shape = [self.batch_size, self.zed_dim]
        zed_sample = tf.random_normal(zed_shape, name='dream_z')

        self.y_dream = self._generator_fn(zed_sample, reuse=True, training=False)
        self.y_dream_summary = tf.summary.image('y_dream', self.y_dream, max_outputs=8)



    """ Execute the dreaming op. Use summary writer """
    def dream(self):
        self.write_summary(self.y_dream_summary)




    """ Construct the Generator, G(x)
    Here we don't have an encoder.
    Instead, there is a Generator, and a Descriminator
    We sample from zed ~ N(0,1), and generate samples from P(X)
    X are our training data

    The discriminator wants to distinguish samples y_hat ~ G(X)
    from y_hat ~ P(X). That is, the reals from the fakes.
    D(y_hat) is built into basemodel so we force adversarial_training = True
    """
    def model(self, reuse=False, training=True):
        ## Shape:
        zed_shape = [self.batch_size, self.zed_dim]
        zed_sample = tf.random_normal(zed_shape, name='epsilon')

        y_hat = self._generator_fn(zed_sample, reuse=reuse, training=training)

        return y_hat


    """ Small generator for MNIST
    architecture from the InfoGAN paper
    """
    def _generator_small(self, zed, reuse=False, training=True):
        zed_expand = slim.fully_connected(zed, 1024, scope='gen_zed_expand', reuse=reuse)
        zed_expand = slim.batch_norm(zed_expand, scope='gen_bn1', reuse=reuse, is_training=training)

        zed_expand = tf.expand_dims(zed_expand, 1)
        zed_expand = tf.expand_dims(zed_expand, 2)
        deconv1_0 = slim.convolution2d_transpose(zed_expand, 128, 7, 1, padding='VALID', scope='gen_deconv1_0', reuse=reuse)
        deconv1_0 = slim.batch_norm(deconv1_0, scope='gen_bn2', reuse=reuse, is_training=training)

        deconv1_1 = slim.convolution2d_transpose(deconv1_0, 64, 4, 2, padding='SAME', scope='gen_deconv1_1', reuse=reuse)
        deconv1_1 = slim.batch_norm(deconv1_1, scope='gen_bn3', reuse=reuse, is_training=training)

        deconv_out = slim.convolution2d_transpose(deconv1_1, 1, 4, 2, padding='SAME', scope='gen_deconv_out',
            reuse=reuse, activation_fn=None)

        print 'gan/_generator_small():'
        print '\tzed', zed.get_shape()
        print '\tzed_expand', zed_expand.get_shape()
        print '\tdeconv1_0', deconv1_0.get_shape()
        print '\tdeconv1_1', deconv1_1.get_shape()
        print '\tdeconv_out', deconv_out.get_shape()

        return deconv_out


    def _generator(self, zed, reuse=False, training=True):
        net = slim.convolution2d_transpose(zed, self.n_kernels*2, 5, 3, padding='VALID', scope='deconv1_0', reuse=reuse)
        net = slim.batch_norm(net, scope='bn6', reuse=reuse, is_training=training)
        print '\tdeconv1_0', net.get_shape()
        if self.bayesian:
            net = slim.dropout(net, scope='drop2')

        net = slim.convolution2d_transpose(net, self.n_kernels, 5, 2, padding='VALID', scope='deconv2_0', reuse=reuse)
        print '\tdeconv2_0', net.get_shape()
        net = slim.convolution2d_transpose(net, self.n_kernels, 5, 2, padding='VALID', scope='deconv2_1', reuse=reuse)
        net = slim.batch_norm(net, scope='bn7', reuse=reuse, is_training=training)
        print '\tdeconv2_1', net.get_shape()
        ## Set to 1/2 input size for 2x upconv
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


    """ Lightweight for MNIST dataset
    architecture from the InfoGAN paper
    """
    def _descriminator_small(self, input_tensor, reuse=False, training=True):
        conv1_0 = slim.convolution2d(input_tensor, 64, 4, 2, padding='SAME', scope='dis_conv1_0', reuse=reuse)
        conv1_1 = slim.convolution2d(conv1_0, 128, 4, 2, padding='SAME', scope='dis_conv1_1', reuse=reuse)
        conv1_1 = slim.batch_norm(conv1_1, scope='dis_bn1', reuse=reuse, is_training=training)

        pool = slim.max_pool2d(conv1_1, 7, 1, scope='dis_pool')

        conv1_1_flat = slim.flatten(pool)
        fc_1 = slim.fully_connected(conv1_1_flat, 1024, scope='dis_fc1', reuse=reuse)
        fc_1 = slim.batch_norm(fc_1, scope='dis_bn2', reuse=reuse, is_training=training)
        decision = slim.fully_connected(fc_1, 2, scope='dis_out', reuse=reuse)

        print 'gan/_descriminator_small():'
        print '\tinput_tensor', input_tensor.get_shape()
        print '\tconv1_0', conv1_0.get_shape()
        print '\tconv1_1', conv1_1.get_shape()
        print '\tpool', pool.get_shape()
        print '\tconv1_1_flat', conv1_1_flat.get_shape()
        print '\tfc_1', fc_1.get_shape()
        print '\tdecision', decision.get_shape()

        return decision




    def _descriminator(self, input_tensor, reuse=False, training=True):

        pass
