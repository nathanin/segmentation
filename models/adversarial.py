import tensorflow as tf
import numpy as np
import sys, time, os

slim = tf.contrib.slim

""" Adversarial loss functions:

@article{luc2016semantic,
  title={Semantic segmentation using adversarial networks},
  author={Luc, Pauline and Couprie, Camille and Chintala, Soumith and Verbeek, Jakob},
  journal={arXiv preprint arXiv:1611.08408},
  year={2016}
}

network - a class extending BaseModel that has a method network.model()

To be run instead of network._init_training_ops; must define:
network.loss_op
network.optimizer ((optional))
network.train_op
network.train_op_list
"""

class AdversarialTraining(object):
    def __init__(self, sess, network):
        self.sess = sess
        self.network = network
        network.adversarial_input = self._create_adv_in()
        network.adversarial_op = self._create_adversary()


    """ Construct the input for adversarial descriminator:

    For each element k in network.batch_size, decide to pass:
    1) network.y_hat[k, ...]
    or
    2) network.input_y[k, ...]

    ## input_y should be 4-D: [k, h, w, n_classes] if its label image, convert to one-hot image
    """
    def _create_adv_in(self):
        ## Convert input_y to one-hot image if necessary

        ## Create one-hot vector labels and randomly fill it with 0, 1

        ## 0 = use generated, 1 = use ground-truth

        ## construct new tensor from y_hat, replacing element k from input_y[k, ...] if labels[k] = 1

    def _label2onehot(self, y_in):
        pass


    def _create_adversary(self):
        net = slim.convolution2d(self.network.adversarial_input,
            num_outputs = 64,
            kernel_size = 5,
            stride = 2,
            padding = 'VALID',
            scope = 'adv_conv1')
        net = slim.max_pool2d(net, 2, scope='adv_pool1')
        net = slim.convolution2d(net, 64, 5, 2, padding= 'VALID', scope='adv_conv2')
        net = slim.max_pool2d(net, 2, scope='adv_pool2')

        net = slim.flatten(net, scope='adv_flat')
        net = slim.fully_connected(net, 128, scope='adv_fc1')
        net = slim.fully_connected(net, 256, scope='adv_fc2')
        net = slim.fully_connected(net, 2, scope='adv_output')
        return net

    def _create_adversarial_loss(self):
        pass
