"""
Scripty lightweight GAN for MNIST
"""

import tensorflow as tf
import numpy as np
import os, glob, cv2, datetime

from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape, name='weight'):
    return tf.get_variable(name, shape=shape,
        initializer=tf.random_normal_initializer(stddev=0.02))

def bias_variable(shape, name='bias'):
    return tf.get_variable(name, shape=shape,
        initializer=tf.constant_initializer(0.0))

source_dir = '/Users/nathaning/Envs/tensorflow/MNIST_data'
mnist = input_data.read_data_sets(source_dir)

batch_size = 64
zed_dim = 2
hidden_fc = [128, 256]
out_dim = 784
disc_fc = 512

g_lr = 1e-4
d_lr = 1e-5
epochs = 300
iterations= 250
timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_dir = './gan/logs/{}'.format(timestamp)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def lrelu(in_tensor, alpha=0.2):
    in_tensor = tf.convert_to_tensor(in_tensor)
    alpha = tf.convert_to_tensor(alpha)
    return tf.maximum(in_tensor*alpha, in_tensor)

def generator(z_in, y_in=None, reuse=False):
    with tf.variable_scope('Generator') as scope:
        if reuse:
            scope.reuse_variables()

        h0_weight = weight_variable([zed_dim, hidden_fc[0]], name='gen_w0')
        h0_bias = bias_variable([hidden_fc[0]], name='gen_b0')

        h1_weight = weight_variable([hidden_fc[0], hidden_fc[1]], name='gen_w1')
        h1_bias = bias_variable([hidden_fc[1]], name='gen_b1')

        out_weight = weight_variable([hidden_fc[1], out_dim], name='gen_w_out')
        out_bias = bias_variable([out_dim], name='gen_b_out')

        h0 = tf.nn.relu(tf.matmul(z_in, h0_weight, name='gen_h0') + h0_bias)
        h1 = tf.nn.relu(tf.matmul(h0, h1_weight, name='gen_h0') + h1_bias)
        out = tf.nn.tanh(tf.matmul(h1, out_weight, name='gen_out') + out_bias)

    return out


def discriminator(x_in, y_in=None, reuse=False):
    with tf.variable_scope('Discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        h_weight = weight_variable([out_dim, disc_fc], name='dis_w0')
        h_bias = bias_variable([disc_fc], name='dis_b0')

        out_weight = weight_variable([disc_fc, 1], name='dis_w_out')
        out_bias = bias_variable([1], name='dis_b_out')

        h0 = lrelu(tf.matmul(x_in, h_weight, name='dis_h0') + h_bias)
        out_logit = tf.matmul(h0, out_weight, name='dis_w0') + out_bias
        out_sig = tf.nn.sigmoid(out_logit)

        return out_sig, out_logit


if __name__ == '__main__':

    z_in = tf.placeholder('float', [batch_size, zed_dim], name='z_in')
    x_real = tf.placeholder('float', [batch_size, out_dim], name='x_real')
    G_z = generator(z_in, reuse=False)
    D_fake, D_fake_logit = discriminator(G_z, reuse=False)
    D_real, D_real_logit = discriminator(x_real, reuse=True)

    with tf.name_scope('loss'):
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(D_fake), logits=D_fake_logit))
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(D_real), logits=D_real_logit))

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(D_fake), logits=D_fake_logit))
        d_loss = d_loss_real + d_loss_fake

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis_' in var.name]
    g_vars = [var for var in t_vars if 'gen_' in var.name]
    print 'd_vars\n', d_vars
    print 'g_vars\n', g_vars

    d_solver = tf.train.AdamOptimizer(d_lr)
    g_solver = tf.train.AdamOptimizer(g_lr)
    d_train_op = d_solver.minimize(d_loss, var_list=d_vars)
    g_train_op = g_solver.minimize(g_loss, var_list=g_vars)

    d_loss_sum = tf.summary.scalar('d_loss', d_loss)
    g_loss_sum = tf.summary.scalar('g_loss', g_loss)

    grads = tf.gradients(d_loss, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    grad_sum_list = []
    for grad, var in grads:
        if 'BatchNorm' not in var.name:
            grad_sum_list.append(tf.summary.histogram(var.name + '/gradient', grad))

    var_sum_list = []
    for var in tf.trainable_variables():
        if 'BatchNorm' not in var.name:
            var_sum_list.append(tf.summary.histogram(var.name, var))

    x_real_img = tf.reshape(x_real, [batch_size, 28, 28, 1])
    g_z_img = tf.reshape(G_z, [batch_size, 28, 28, 1])
    x_real_sum = tf.summary.image('x_real', x_real_img, max_outputs=4)
    g_z_sum = tf.summary.image('g_z', g_z_img, max_outputs=4)

    summary_op = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()


    test_z = np.random.uniform(-1.0, 1.0, [batch_size, zed_dim]).astype(np.float32)
    test_x, _ = mnist.test.next_batch(batch_size)
    test_feed_dict = {z_in: test_z, x_real: test_x}
    with tf.Session(config=config) as sess:
        summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)

        sess.run([init_op])

        global_ix = 0
        for epx in xrange(epochs):
            for itx in xrange(iterations):
                global_ix += 1
                z_sample = np.random.uniform(-1.0, 1.0, [batch_size, zed_dim]).astype(np.float32)
                real_batch, _ = mnist.train.next_batch(batch_size)

                feed_dict={z_in: z_sample, x_real: real_batch}
                _= sess.run([d_train_op],
                    feed_dict=feed_dict)
                _= sess.run([g_train_op],
                    feed_dict=feed_dict)
                _= sess.run([g_train_op],
                    feed_dict=feed_dict)

            d_loss_, g_loss_, summary_str = sess.run(
                [d_loss, g_loss, summary_op],
                feed_dict=test_feed_dict)

            summary_writer.add_summary(summary_str, global_ix)
            print 'Epoch {} [D loss]: {} [G loss]: {}'.format(
                epx, d_loss_, g_loss_ )
