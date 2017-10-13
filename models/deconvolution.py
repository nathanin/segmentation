import tensorflow as tf
import numpy as np
import sys, time, os

slim = tf.contrib.slim

sys.path.insert(0, '.')
from basemodel import BaseModel

class DeconvModel(BaseModel):
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
        learning_rate = 1e-4,
        load_snapshot = None,
        load_snapshot_from = None,):

        ## TODO: Check args

        super(DeconvModel, self).__init__(
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
        self.n_kernels = 16
        with tf.name_scope('ConvDeconv'):
            self.y_hat = self.model()

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


    """ Implements some generic convolution / deconvolution model """
    def model(self):
        with tf.name_scope('Encoder') as scope:
            input_x = self.input_x.get_shape().as_list()[1]
            input_y = self.input_x.get_shape().as_list()[2]
            net = slim.convolution2d(self.input_x,
                num_outputs = self.n_kernels,
                kernel_size = 5,
                stride = 2,
                padding = 'SAME',
                scope = 'conv1')
            net = slim.convolution2d(net, self.n_kernels,  5, 2, padding='VALID', scope='conv2')
            net = slim.max_pool2d(net, 2, scope='pool2')
            net = slim.convolution2d(net, self.n_kernels, 5, 2, padding='VALID', scope='conv3')
            if self.bayesian:
                net = slim.dropout(net, scope='drop1')

        with tf.name_scope('Decoder') as scope:
            net = slim.convolution2d_transpose(net, self.n_kernels, 3, 2, padding='VALID', scope='deconv1')
            net = slim.convolution2d_transpose(net, self.n_kernels, 7, 3, padding='VALID', scope='deconv2')
            ## Set to 1/2 input size for 2x upconv
            net = tf.image.resize_bilinear(net, [input_x//2, input_y//2])
            net = slim.convolution2d_transpose(net, self.n_classes, 2, 2,
                padding='SAME', scope='deonv3', activation_fn=None)
            if self.bayesian:
                net = slim.dropout(net, scope='drop2')

            ## Force to be the same size as input
            net = tf.images.resize_image_with_crop_or_pad(net, input_x, input_y)

            ## Debugging
            # net = tf.Print(net, [net, 'Model processing'])

        return net
