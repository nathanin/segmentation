import tensorflow as tf
import numpy as np
import sys, time, os

slim = tf.contrib.slim

sys.path.insert(0, '.')
from basemodel import BaseModel

# sys.path.insert(0, '..')
# from utils.ops import *

""" Generative Adversarial Network class

It should implement a method model() and define y_hat,
along with special losses and overload attributes for training
to follow the GAN training algo.

copied so many tricks from
https://github.com/carpedm20/DCGAN-tensorflow

GAN's aren't easy to train..
"""

class InfoGAN(BaseModel):
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
        generator_fc = 512,
        discriminator_fc = 512,
        label_dim = None,
        c_discrete = None,
        c_ctns = None):


        ## TODO: Check args
        assert adversarial_training == True

        super(InfoGAN, self).__init__(
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
        self.adversarial_lr = 2e-4
        self.label_dim = label_dim

        self._generator_fn = self._generator
        self._adversarial_net_fn = self._discriminator

        with tf.name_scope('ConvDeconv') as scope:
            print 'Instantiating GAN'
            self.zed_sample = tf.placeholder('float32', [None, self.zed_dim], name='zed_sample')

            if self.label_dim:
                print 'Setting up CGAN with {} labels'.format(self.label_dim)
                self.label_in = tf.placeholder('uint8', [self.batch_size])
                self.label_in_onehot = tf.squeeze(tf.one_hot(self.label_in, self.label_dim))
                print 'Set up labels: {} one hot: {}'.format(
                    self.label_in.get_shape(), self.label_in_onehot.get_shape())
                self.y_hat = self._generator_fn(self.zed_sample, self.label_in_onehot)
            else:
                print 'Setting up GAN with {} latent variables'.format(self.zed_dim)
                self.y_hat = self._generator_fn(self.zed_sample)

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


        ## Set up a constant for dream feeding
        self.dream_z_sample = np.random.uniform(-1, 1, [self.batch_size, self.zed_dim]).astype(np.float32)
        _, self.dream_label_sample = self.dataset.mnist.train.next_batch(self.batch_size)
        # self.dream_label_sample = tf.one_hot(self.dream_label_sample, self.label_dim)
        print 'Dream sample z:', self.dream_z_sample.shape
        print 'Dream sample label:', self.dream_label_sample.shape



    def train_step(self):
        batch_x, batch_labels = self.dataset.mnist.train.next_batch(self.batch_size)
        batch_x = self.dataset._reshape_batch(batch_x)
        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.zed_dim]).astype(np.float32)

        if self.label_dim:
            feed_dict = {self.input_x: batch_x, self.label_in: batch_labels, self.zed_sample: batch_z}
        else:
            feed_dict = {self.input_x: batch_x, self.zed_sample: batch_z}

        _, _, D_loss, G_loss, _, _ = self.sess.run(self.train_op_list, feed_dict=feed_dict)

        #print 'D loss: {} G loss: {}'.format(D_loss, G_loss)

        self.write_summary(self.summary_op, feed_dict=feed_dict)



    """ Overloading from BaseModel """
    def _init_summary_ops(self):
        self.real_loss_summary = tf.summary.scalar('loss_real', self.loss_real)
        self.fake_loss_summary = tf.summary.scalar('loss_fake', self.loss_fake)
        self.real_adv_summary = tf.summary.histogram('real_adv', self.real_adv)
        self.fake_adv_summary = tf.summary.histogram('fake_adv', self.fake_adv)
        self.zed_sample_summary = tf.summary.histogram('z_sample', self.zed_sample)

        grad_sum_list = []
        for grad, var in self.grads:
            if 'BatchNorm' not in var.name:
                grad_sum_list.append(tf.summary.histogram(var.name + '/gradient', grad))

        var_sum_list = []
        for var in tf.trainable_variables():
            if 'BatchNorm' not in var.name:
                var_sum_list.append(tf.summary.histogram(var.name, var))

        self.summary_op = tf.summary.merge([
            self.adv_loss_summary,
            self.gen_loss_summary,
            self.real_loss_summary,
            self.fake_loss_summary,
            self.real_adv_summary,
            self.fake_adv_summary,
            self.zed_sample_summary] + \
            var_sum_list + \
            grad_sum_list)
        # self.summary_op = tf.summary.merge_all()


    """ Initialize the training ops
    creates self.train_op_list with a list of training ops
    we need one op that updates the generator
    and one op that updates the discriminator

    train_step() calls: self.train_op_list, and self.adversarial_train_list

    """
    def _init_training_ops(self):
        print 'Setting up GAN training'
        ## seg_optimizer used to train the generator
        self.gen_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name='genAdam')
        self.adversarial_optimizer = tf.train.AdamOptimizer(self.adversarial_lr, beta1=0.5, name='descAdam')


        if self.label_dim:
            print 'Adversarial x ~ p(data)'
            self.real_adv, self.real_adv_logit = self._adversarial_net_fn(self.input_x, self.label_in_onehot, reuse=False)
            print 'Adversarial x ~ G(z)'
            self.fake_adv, self.fake_adv_logit = self._adversarial_net_fn(self.y_hat, self.label_in_onehot, reuse=True)
        else:
            print 'Adversarial x ~ p(data)'
            self.real_adv, self.real_adv_logit = self._adversarial_net_fn(self.input_x, reuse=False)
            print 'Adversarial x ~ G(z)'
            self.fake_adv, self.fake_adv_logit = self._adversarial_net_fn(self.y_hat, reuse=True)


        ## Separate variables for the optimizers to use - no more tf.stop_gradient !!
        ## https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'dis_' in var.name]
        self.g_vars = [var for var in t_vars if 'gen_' in var.name]

        print 'd_vars\n', self.d_vars
        print 'g_vars\n', self.g_vars

        ## Trainig objectives for real and fake images
        # self.real_label = tf.one_hot(tf.ones_like(tf.argmax(self.real_adv, 1)), 2)
        # self.fake_label = tf.one_hot(tf.zeros_like(tf.argmax(self.fake_adv, 1)), 2)
        self.real_label = tf.ones_like(self.real_adv)
        self.fake_label = tf.zeros_like(self.fake_adv)

        self.loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.real_label, logits=self.real_adv_logit))
        self.loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.fake_label, logits=self.fake_adv_logit))

        self.adv_loss_op = self.loss_real + self.loss_fake
        self.gen_loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.real_label, logits=self.fake_adv_logit))

        self.grads = tf.gradients(self.adv_loss_op, tf.trainable_variables())
        self.grads = list(zip(self.grads, tf.trainable_variables()))

        ## ?? slim documentation says to do this for batch_norm layers
        self.adv_train_op = self.adversarial_optimizer.minimize(self.adv_loss_op, var_list=self.d_vars)
        self.gen_train_op = self.gen_optimizer.minimize(self.gen_loss_op, var_list=self.g_vars)

        self.train_op_list = [self.adv_train_op, self.gen_train_op, self.gen_train_op,
            self.adv_loss_op, self.gen_loss_op, self.gs_increment]
        # self.adversarial_train_list = [self.adv_train_op]

        self.adv_loss_summary = tf.summary.scalar('adv_loss', self.adv_loss_op)
        self.gen_loss_summary = tf.summary.scalar('gen_loss', self.gen_loss_op)
        self.input_image_summary = tf.summary.image('input_x', self.input_x, max_outputs=4)
        self.generated_image_summary = tf.summary.image('G(z)', self.y_hat, max_outputs=4)




    """ Dream is to generate z and pass through the generator

    Sample: x ~ G(z ~ N(0,1))
    """
    def _init_dream_ops(self):
        print 'Initializing dream ops'
        # zed_shape = [self.batch_size, self.zed_dim]
        # zed_sample = tf.random_normal(zed_shape, name='dream_z')
        # self.dream_z = tf.random_uniform(zed_shape, -1.0, 1.0, name='dream_z')
        # zed_sample = tf.Print(zed_sample, ['dream sampling'])
        self.dream_z = tf.placeholder('float32', [self.batch_size, self.zed_dim], name='dream_z')

        if self.label_dim:
            self.dream_label = tf.placeholder('uint8', [self.batch_size], name='dream_label')
            self.dream_label_onehot = tf.one_hot(self.dream_label, self.label_dim)
            self.y_dream = self._generator_fn(self.dream_z, self.dream_label_onehot, reuse=True,
                training=False)
        else:
            self.y_dream = self._generator_fn(self.dream_z, reuse=True, training=False)

        self.y_dream_summary = tf.summary.image('y_dream', self.y_dream, max_outputs=8)



    """ Execute the dreaming op. Use summary writer """
    def dream(self):
        if self.label_dim:
            feed_dict = {self.dream_z: self.dream_z_sample,
                         self.dream_label: self.dream_label_sample}
        else:
            feed_dict = {self.dream_z: self.dream_z_sample}

        self.write_summary(self.y_dream_summary, feed_dict=feed_dict)




    """ Small generator for MNIST
    architecture from the InfoGAN paper
    """
    def _generator(self, zed, label=None, reuse=False, training=True):
        ## 7 * 7 * 128 = 6272
        ## 7 * 7 * 1024 = 50176
        ## 7 * 7 * 512 = 25088
        with tf.variable_scope('Generator') as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.fully_connected, slim.convolution2d_transpose],
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),
                normalizer_fn=slim.batch_norm,
                normalizer_params={'is_training': training, 'decay': 0.9,
                    'updates_collections': None,
                    'epsilon': 1e-5,
                    'scale': True},
                reuse=reuse):

                if self.label_dim:
                    zed = tf.concat([zed, label], axis=1)
                    label_tensor = tf.reshape(label, [self.batch_size, 1, 1, self.label_dim])
                zed_project = slim.fully_connected(zed, 512, scope='gen_zed_project')

                # if self.label_dim:
                #     zed_project = tf.concat([zed_project, label], axis=1)
                fc1 = slim.fully_connected(zed_project, 64*7*7, scope='gen_fc1')
                g0 = tf.reshape(fc1, [-1, 7, 7, 64], name='gen_fc1_reshape')

                # if self.label_dim:
                #     g0 = self.concat_tensor_label(g0, label_tensor)
                g1 = slim.convolution2d_transpose(g0, 64, 4, 2, padding='SAME', scope='gen_deconv1',
                    weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False)) ## 14 x 14

                # if self.label_dim:
                #     g1 = self.concat_tensor_label(g1, label_tensor)
                deconv_out = slim.convolution2d_transpose(g1, 1, 4, 2, padding='SAME',
                    weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False), ## 14 x 14
                    scope='gen_deconv_out', activation_fn=tf.nn.tanh) ## 28 x 28

        print 'gan/_generator_small():'
        print '\t zed', zed.get_shape()
        print '\t zed_project', zed_project.get_shape()
        print '\t fc1', fc1.get_shape()
        print '\t g0', g0.get_shape()
        print '\t g1', g1.get_shape()
        #print '\t g2', g2.get_shape()
        #print '\t g3', g3.get_shape()
        print '\t deconv_out', deconv_out.get_shape()

        return deconv_out

    """ Lightweight for MNIST dataset
    architecture from the InfoGAN paper
    """
    def _discriminator(self, image, label=None, reuse=False, training=True):
        with tf.variable_scope('Discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            with slim.arg_scope([slim.convolution2d, slim.fully_connected],
                weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02),
                normalizer_fn=slim.batch_norm,
                normalizer_params={'is_training': training, 'decay': 0.9,
                    'updates_collections': None,
                    'epsilon': 1e-5,
                    'scale': True},
                activation_fn=self.leaky_relu,
                reuse=reuse):

                if self.label_dim:
                    print '\t DISCRIMINATOR LABEL', label.get_shape()
                    image = self.concat_tensor_label(image, label)
                    print '\t image concat', image.get_shape()
                conv0 = slim.convolution2d(image, 11, 5, 2, padding='SAME', scope='dis_conv0',
                    weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                    normalizer_fn=None)

                # if self.label_dim:
                #     conv0 = self.concat_tensor_label(conv0, label)
                #     print '\t conv0 concat', conv0.get_shape()
                conv1 = slim.convolution2d(conv0, 24, 5, 2, padding='SAME', scope='dis_conv1',
                    weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                conv1_flat = slim.flatten(conv1)

                # if self.label_dim:
                #     conv1_flat = tf.concat([conv1_flat, label], axis=1)
                fc1 = slim.fully_connected(conv1_flat, 128, scope='dis_fc1')
                adv = slim.fully_connected(fc1, 1, scope='dis_out', activation_fn=None, normalizer_fn=None)
                adv_sig = tf.nn.sigmoid(adv)

        print 'gan/_discriminator_small():'
        print '\t image', image.get_shape()
        print '\t conv0', conv1.get_shape()
        print '\t conv1', conv1.get_shape()
        print '\t conv1_flat', conv1_flat.get_shape()
        print '\t fc1', fc1.get_shape()
        print '\t adv', adv.get_shape()

        return adv_sig, adv

    """ leaky_relu from:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_ops.py """
    def leaky_relu(self, features, alpha=0.2):
        features = tf.convert_to_tensor(features)
        alpha = tf.convert_to_tensor(alpha)
        return tf.maximum(alpha * features, features)


    def concat_tensor_label(self, image, label):
        label = tf.reshape(label, [self.batch_size, 1, 1, self.label_dim])
        x_shapes = image.get_shape()
        y_shapes = label.get_shape()
        # print 'concat_tensor_label x_shapes', x_shapes
        # print 'concat_tensor_label y_shapes', y_shapes

        ## [batch, h, w, label_dim]
        return tf.concat([
        image, label*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


    def dream_manifold(self):
        pass
