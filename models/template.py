import tensorflow as tf
import numpy as np
import sys, time, os

slim = tf.contrib.slim

sys.path.insert(0, '.')
from basemodel import BaseModel

class Template(BaseModel):
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




    def model(self):
        """
        Implement the tensorflow graph ops

        At the point model is called, the input functions are all defined
        """
        pass
