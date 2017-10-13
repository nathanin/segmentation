import tensorflow as tf
import numpy as np
import sys, time, os

slim = tf.contrib.slim

## TODO: defaults for all arguments/clean arg passing
## TODO: clean workflow/use cases/
class BaseModel(object):
    def __init__(self,
                 sess,
                 mode,
                 log_dir,
                 dataset,
                 bayesian,
                 save_dir,
                 n_classes,
                 input_dims,
                 input_channel,
                 load_snapshot,
                 learning_rate,
                 load_snapshot_from,):
        ## Constants / hyperparameters
        self.mode = mode
        self.log_dir = log_dir
        self.dataset = dataset
        self.save_dir = save_dir
        self.bayesian = bayesian
        self.n_classes = n_classes
        self.input_dims = input_dims
        self.learning_rate = learning_rate
        self.input_channel = input_channel

        ## Move somewhere else
        if load_snapshot:
            self.load_snapshot = load_snapshot
        else:
            self.load_snapshot = None
        ## Mode makes load_snapshot true
        if mode=='INFERENCE': self.load_snapshot = True

        if load_snapshot_from:
            self.load_snapshot_from = load_snapshot_from
        else:
            self.load_snapshot_from = None
        ## /end optionals

        self.summary_iter = 50
        self.init_op = tf.global_variables_initializer()

        ## Always needed
        self._init_session(sess)
        # self.sess.run([self.init_op])

    """ Setup functions """
    def _init_session(self, sess):
        self.sess = sess

        if self.dataset is not None:
            self.dataset.set_tf_sess(self.sess)

        if self.log_dir is not None:
            self.summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.gs_increment = tf.assign_add(self.global_step, 1)


    ## Always needed for training or inference
    def _init_saver(self, name='model'):
        self.saver = tf.train.Saver(max_to_keep=1)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_path = os.path.join(self.save_dir, '{}.ckpt'.format(name))

        if self.load_snapshot:
            try:
                if self.load_snapshot_from:
                    print 'Loading latest snapshot from {}'.format(self.load_snapshot_from)
                    self.saver.restore(self.sess, self.load_snapshot_from)
                    load_from = load_snapshot_from
                else:
                    print 'Loading latest snapshot from {}'.format(self.save_dir)
                    self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir))

                print 'Success! Resuming from global step {}'.format(
                        tf.train.global_step(self.sess, self.global_step))
            except:
                print 'Failed to load snapshot; proceed with training'



    def _init_input(self):
        if self.mode == 'INFERENCE':
            ## TODO: deal with variable input shapes; call resize() before inference
            self.input_x = tf.placeholder('float',
                [None, self.input_dims, self.input_dims, self.input_channel])
            return

        print 'Setting up TRAINING mode input ops'
        self.input_x = self.dataset.image_op
        input_y = self.dataset.mask_op
        self.input_y = tf.cast(input_y, tf.uint8)

        print 'input_x:', self.input_x.get_shape(), self.input_x.dtype
        print 'input_y:', self.input_y.get_shape(), self.input_y.dtype


    """ Define some other loss functions by overriding these:
    loss_op
    optimizer
    train_op
    train_op_list
    """
    def _init_training_ops(self):
        ## Skip for inference mode
        if self.mode == 'INFERENCE':
            return

        labels_onehot = tf.reshape(tensor = self.input_y, shape=(-1, self.n_classes))
        labels_onehot = tf.one_hot(labels_onehot, self.n_classes)
        y_hat_flat = tf.reshape(tensor = self.y_hat, shape=(-1, self.n_classes))

        self.loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels_onehot, logits=y_hat_flat))

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)
        self.train_op_list = [self.train_op,
                              self.gs_increment]

    def _init_summary_ops(self):
        ## Skip for inference mode
        if self.mode == 'INFERENCE':
            return

        self.loss_summary = tf.summary.scalar('loss', self.loss_op)
        self.output_summary = tf.summary.image('mask', self.output, max_outputs=4)
        self.input_summary = tf.summary.image('img', self.input_x, max_outputs=4)

        self.summary_op = tf.summary.merge_all()

    def train_step(self):
        ## Skip for inference mode
        if self.mode == 'INFERENCE':
            return

        _ = self.sess.run(self.train_op_list)
        ## Check if we should summarize --- I think it takes a lot of time
        gs = tf.train.global_step(self.sess, self.global_step)
        if gs % self.summary_iter == 0:
            print gs, 'writing summary'
            summary_str = self.sess.run([self.summary_op])[0]
            self.summary_writer.add_summary(summary_str,
                tf.train.global_step(self.sess, self.global_step))

    def snapshot(self):
        ## Skip for inference mode
        if self.mode == 'INFERENCE':
            return

        gs = tf.train.global_step(self.sess, self.global_step)
        print 'Global step {}, snapshotting to {}'.format(gs, self.save_path)
        self.saver.save(self.sess, self.save_path, global_step=gs)


    def test(self):
        ## Skip for inference mode
        if self.mode == 'INFERENCE':
            return

        loss = self.sess.run([self.loss_op])
        return loss[0]


    """ v1: accept a 4D nparray as input """
    def infer(self, imgs):
        feed_dict = {self.input_x: imgs}
        output = self.sess.run(self.inference_ops, feed_dict=feed_dict)

        return output
