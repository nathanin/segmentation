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

copied so many tricks from
https://github.com/carpedm20/DCGAN-tensorflow

GAN's aren't easy to train..
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
        label_dim = None):


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
        self.adversarial_lr = 2e-4
        self.label_dim = label_dim

        self._generator_fn = self._generator
        self._adversarial_net_fn = self._discriminator

        with tf.name_scope('ConvDeconv') as scope:
            print 'Instantiating GAN'
            self.model(reuse=False, training=True)

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


    def train_step(self):
        batch_x, batch_labels = self.dataset.mnist.train.next_batch(self.batch_size)
        batch_x = self.dataset._reshape_batch(batch_x)
        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.zed_dim]).astype(np.float32)
        feed_dict = {self.input_x: batch_x, self.label_in: batch_labels, self.zed_sample: batch_z}

        # print 'batch_x', batch_x.shape, batch_x.dtype, batch_x.min(), batch_x.max()

        _, _, D_loss, G_loss, _ = self.sess.run(self.train_op_list, feed_dict=feed_dict)
        # _ = self.sess.run(self.gs_increment)
        # _ = self.sess.run(self.adv_train_op, feed_dict=feed_dict)
        # _ = self.sess.run(self.gen_train_op, feed_dict=feed_dict)
        # _ = self.sess.run(self.gen_train_op, feed_dict=feed_dict)

        print 'D loss: {} G loss: {}'.format(D_loss, G_loss)

        self.write_summary(self.summary_op, feed_dict=feed_dict)



    """ Overloading from BaseModel """
    def _init_summary_ops(self):
        self.real_loss_summary = tf.summary.scalar('loss_real', self.loss_real)
        self.fake_loss_summary = tf.summary.scalar('loss_fake', self.loss_fake)
        self.real_adv_summary = tf.summary.histogram('real_adv', self.real_adv)
        self.fake_adv_summary = tf.summary.histogram('fake_adv', self.fake_adv)
        self.zed_sample_summary = tf.summary.histogram('z_sample', self.zed_sample)
        if self.label_in:
            self.label_in_summary = tf.summary.histogram('label_hist', tf.argmax(self.label_in, 1))

        # for grad, var in self.grads:
        #     tf.summary.histogram(var.name + '/gradient', grad)
        #
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.name, var)
        self.summary_op = tf.summary.merge([self.adv_loss_summary, self.gen_loss_summary,
            self.real_loss_summary,
            self.fake_loss_summary,
            self.real_adv_summary,
            self.fake_adv_summary,
            self.zed_sample_summary])
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
            self.label_in = tf.placeholder('float32', [self.batch_size, 1])
            self.label_in = tf.one_hot(self.label_in)
        else:
            self.label_in = None

        with tf.variable_scope('Adversarial') as scope:
            print 'Adversarial x ~ p(data)'
            self.real_adv, self.real_adv_logit = self._adversarial_net_fn(self.input_x, self.label_in, reuse=False)

            print 'Adversarial x ~ G(z)'
            self.fake_adv, self.fake_adv_logit = self._adversarial_net_fn(self.y_hat, self.label_in, reuse=True)

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

        # self.adv_loss_op = -tf.reduce_mean(tf.log(self.real_adv) + tf.log(1. - self.fake_adv))
        # self.gen_loss_op = -tf.reduce_mean(tf.log(self.fake_adv))

        # self.grads = tf.gradients(self.adv_loss_op, tf.trainable_variables())
        # self.grads = list(zip(self.grads, tf.trainable_variables()))
        # print 'Grads\n', self.grads

        ## ?? slim documentation says to do this for batch_norm layers
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # print 'update ops\n', update_ops
        # with tf.control_dependencies(update_ops):

        self.adv_train_op = slim.learning.create_train_op(self.adv_loss_op, self.adversarial_optimizer)
        self.gen_train_op = slim.learning.create_train_op(self.gen_loss_op, self.gen_optimizer)
        # self.adv_train_op = self.adversarial_optimizer.minimize(self.adv_loss_op, var_list=self.d_vars)
        # self.gen_train_op = self.gen_optimizer.minimize(self.gen_loss_op, var_list=self.g_vars)

        self.train_op_list = [self.adv_train_op, self.gen_train_op,
            self.adv_loss_op, self.gen_loss_op, self.gs_increment]
        # self.adversarial_train_list = [self.adv_train_op]

        # self.bce_real_summary = tf.summary.scalar('l_bce_real', tf.reduce_mean(self.l_bce_real))
        # self.bce_fake_summary = tf.summary.scalar('l_bce_fake', tf.reduce_mean(self.l_bce_fake))
        self.adv_loss_summary = tf.summary.scalar('adv_loss', self.adv_loss_op)
        self.gen_loss_summary = tf.summary.scalar('gen_loss', self.gen_loss_op)





    """ Dream is to generate z and pass through the generator

    Sample: x ~ G(z ~ N(0,1))
    """
    def _init_dream_ops(self):
        print 'Initializing dream ops'
        # zed_shape = [self.batch_size, self.zed_dim]
        # zed_sample = tf.random_normal(zed_shape, name='dream_z')
        # self.dream_z = tf.random_uniform(zed_shape, -1.0, 1.0, name='dream_z')
        # zed_sample = tf.Print(zed_sample, ['dream sampling'])
        self.dream_z = tf.placeholder('float32', [None, self.zed_dim], name='dream_z')

        self.y_dream = self._generator_fn(self.dream_z, reuse=True)
        self.y_dream_summary = tf.summary.image('y_dream', self.y_dream, max_outputs=4)





    """ Execute the dreaming op. Use summary writer """
    def dream(self):
        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.zed_dim]).astype(np.float32)
        feed_dict = {self.dream_z: batch_z}
        self.write_summary(self.y_dream_summary, feed_dict=feed_dict)





    """ Construct the Generator, G(x)
    Here we don't have an encoder.
    Instead, there is a Generator, and a discriminator
    We sample from zed ~ N(0,1), and generate samples from P(X)
    X are our training data

    The discriminator wants to distinguish samples y_hat ~ G(X)
    from y_hat ~ P(X). That is, the reals from the fakes.
    D(y_hat) is built into basemodel so we force adversarial_training = True
    """
    def model(self, reuse=False, training=True):
        ## Shape:
        # zed_shape = [self.batch_size, self.zed_dim]
        # zed_sample = tf.random_normal(zed_shape, name='epsilon')
        # self.zed_sample = tf.random_uniform(zed_shape, -1.0, 1.0, name='epsilon')
        self.zed_sample = tf.placeholder('float32', [None, self.zed_dim], name='zed_sample')
        # self.zed_sample = tf.Print(self.zed_sample, ['zed sample'])

        self.y_hat = self._generator_fn(self.zed_sample, reuse=reuse, training=training)
        # self.y_hat = tf.Print(self.y_hat, ['yhat'])







    """ Small generator for MNIST
    architecture from the InfoGAN paper
    """
    def _generator(self, zed, reuse=False, training=True):
        # zed_expand = slim.fully_connected(zed, 1024, scope='gen_zed_expand',
        #     weights_initializer=tf.random_normal_initializer,
        #     reuse=reuse)
        # zed_expand = slim.batch_norm(zed_expand, scope='gen_bn1', reuse=reuse, is_training=training, updates_collections=None)

        ## 7 * 7 * 128 = 6272
        # 7 * 7 * 1024 = 50176
        with slim.arg_scope([slim.fully_connected, slim.convolution2d_transpose],
            weights_initializer=tf.random_normal_initializer,
            normalizer_fn=slim.batch_norm,
            normalizer_params={'is_training': training, 'decay': 0.95},
            reuse=reuse):
            zed_project = slim.fully_connected(zed, 50176, scope='gen_zed_project')
            zed_reshape = tf.reshape(zed_project, [-1, 7, 7, 1024], name='gen_zed_reshape')
            g1 = slim.convolution2d_transpose(zed_reshape, 512, 4, 2, padding='SAME', scope='gen_deconv1')
            g2 = slim.convolution2d_transpose(g1, 512, 4, 2, padding='SAME', scope='gen_deconv2')
            g3 = slim.convolution2d_transpose(g2, 128, 4, 2, padding='SAME', scope='gen_deconv3')

            deconv_out = slim.convolution2d_transpose(g3, 1, 4, 2, padding='SAME', scope='gen_deconv_out', activation_fn=tf.nn.tanh)

        print 'gan/_generator_small():'
        print '\t zed', zed.get_shape()
        print '\t zed_expand', zed_expand.get_shape()
        print '\t zed_project', zed_project.get_shape()
        print '\t zed_reshape', zed_reshape.get_shape()

        return deconv_out

    """ Lightweight for MNIST dataset
    architecture from the InfoGAN paper
    """
    def _discriminator(self, image, label=None, reuse=False, training=True):
        with slim.arg_scope([slim.convolution2d, slim.fully_connected],
            weights_initializer=tf.truncated_normal,
            normalizer_fn=slim.batch_norm,
            normalizer_params={'is_training': training, 'decay': 0.95, 'updates_collections': None},
            activation_fn=self.leaky_relu,
            reuse=reuse):
            if label:
                image = self.concat_tensor_label(image, label)
                print '\t image concat', image.get_shape()
            conv1 = slim.convolution2d(image, 64, 4, 2, padding='SAME', scope='dis_conv1')

            if label:
                conv1 = self.concat_tensor_label(conv1, label)
                print '\t conv1 concat', conv1.get_shape()
            conv2 = slim.convolution2d(conv1, 128, 4, 2, padding='SAME', scope='dis_conv2')
            conv2_flat = slim.flatten(conv2)

            fc1 = slim.fully_connected(conv2_flat, 1024, scope='dis_fc1')
            adv = slim.fully_connected(fc1, 1, scope='dis_out', activation_fn=None, normalizer_fn=None)
            adv_sig = tf.nn.sigmoid(adv)

        print 'gan/_discriminator_small():'
        print '\t image', image.get_shape()
        print '\t conv1', conv1.get_shape()
        print '\t conv2', conv2.get_shape()
        print '\t conv2_flat', conv2_flat.get_shape()
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

        ## [batch, h, w, label_dim]
        return tf.concat([
        image, label*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)
