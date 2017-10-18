import tensorflow as tf
import numpy as np
import sys, time, os

slim = tf.contrib.slim

## TODO: defaults for all arguments/clean arg passing
## TODO: clean workflow/use cases/
class BaseModel(object):
    def __init__(self,
                 sess,
                 mode = 'TRAINING',
                 log_dir = './logs',
                 dataset = None,
                 test_dataset = None,
                 bayesian = False,
                 save_dir = './snapshot',
                 n_classes = None,
                 input_dims = None,
                 input_channel = 3,
                 load_snapshot = True,
                 learning_rate = 1e-3,
                 load_snapshot_from = None,
                 adversarial_training = False):
        ## Constants / hyperparameters
        self.mode = mode
        self.log_dir = log_dir
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.save_dir = save_dir
        self.bayesian = bayesian
        self.n_classes = n_classes
        self.input_dims = input_dims
        self.learning_rate = learning_rate
        self.input_channel = input_channel
        self.adversarial_training = adversarial_training


        ## Flags for defining the relationship of input to output
        ## The child class should set one
        self.IN_OUT_EQUAL = False
        self.IN_OUT_CROP = False
        self.IN_OUT_RATIO = False

        ## Move if/else somewhere else
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

        if self.adversarial_training:
            ## TODO add support for external aversary ("discriminator") net
            self._adversarial_net_fn = self._adversarial_net

        ## /end if/else

        self.summary_iter = 5
        self.init_op = tf.global_variables_initializer()

        ## Always needed
        self._init_session(sess)



    """ Setup functions """
    def _init_session(self, sess):
        self.sess = sess

        if self.dataset is not None:
            self.dataset.set_tf_sess(self.sess)

        if self.log_dir is not None:
            self.summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.gs_increment = tf.assign_add(self.global_step, 1)


    """ Set up a saver; load from checkpoint file if it exists:

    Failure modes:
    - the requested checkpoint is not found
    - the checkpoint has some different variables, or there is shape mismatch
    - edge-case failure if the checkpoint has a subset of the correct variables
        in this case, the newly defined variables will be trained from scratch
    """
    def _init_saver(self, name='model'):
        self.saver = tf.train.Saver(max_to_keep=1)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_path = os.path.join(self.save_dir, '{}.ckpt'.format(name))

        if self.load_snapshot:
            try:
                if self.load_snapshot_from:
                    print 'Loading latest snapshot from {}'.format(
                        self.load_snapshot_from)
                    self.saver.restore(self.sess, self.load_snapshot_from)
                    load_from = load_snapshot_from
                else:
                    print 'Loading latest snapshot from {}'.format(self.save_dir)
                    self.saver.restore(self.sess,
                        tf.train.latest_checkpoint(self.save_dir))

                print 'Success! Resuming from global step {}'.format(
                        tf.train.global_step(self.sess, self.global_step))
            except:
                print 'Failed to load snapshot; proceed with training'




    """ Instantiate input functions from a dataset """
    def _init_input(self):
        if self.mode == 'INFERENCE':
            ## TODO: deal with variable input shapes; call resize() before inference
            self.input_x = tf.placeholder('float',
                [None, self.input_dims, self.input_dims, self.input_channel])
            return

        print 'Setting up TRAINING mode input ops'
        self.input_x = self.dataset.image_op
        self.input_y = self.dataset.mask_op
        # self.input_y = tf.cast(input_y, tf.uint8)

        print 'input_x:', self.input_x.get_shape(), self.input_x.dtype
        print 'input_y:', self.input_y.get_shape(), self.input_y.dtype


    """ Instantiate basic loss function:
    loss_op
    optimizer
    train_op
    train_op_list
    """
    def _init_xentropy_loss(self):
        if self.mode == 'INFERENCE':
            return

        self.input_y_onehot = tf.squeeze(tf.one_hot(self.input_y, self.n_classes))
        print 'input_y_onehot', self.input_y_onehot.get_shape()

        self.xentropy_loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits( labels=self.input_y_onehot, logits=self.y_hat))

        self.inpt_y_0, self.inpt_y_1 = tf.split(self.input_y_onehot, 2, axis=-1)
        self.xentropy_summary = tf.summary.scalar('seg_xentropy', self.xentropy_loss_op)
        print 'xentropy loss op:', self.xentropy_loss_op.dtype




    """ Adversarial loss functions:

    @article{luc2016semantic,
      title={Semantic segmentation using adversarial networks},
      author={Luc, Pauline and Couprie, Camille and Chintala, Soumith and Verbeek, Jakob},
      journal={arXiv preprint arXiv:1611.08408},
      year={2016}
    }

    Some random set of convolutions /pools not necessarily lifted from their paper
    Experiment with downsampling the input
        - This net should be lightning fast. Passing the whole sized map is pretty big.
    """
    def _adversarial_net(self, tensor_in, reuse=False):
        with tf.variable_scope('adversary') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h, w = tensor_in.get_shape().as_list()[1:3]
            print '\tadv input', tensor_in.get_shape(), tensor_in.dtype
            tensor_in = tf.image.resize_bilinear(tensor_in, [h//4, w//4])

            print '\tadv resize', tensor_in.get_shape(), tensor_in.dtype
            net = slim.convolution2d(tensor_in,
                num_outputs = 8,
                kernel_size = 5,
                stride = 2,
                padding = 'SAME',
                scope = 'adv_conv1',
                reuse = reuse)
            net = slim.max_pool2d(net, 3, 3, scope='adv_pool1')
            print '\tadv pool1', net.get_shape()

            net = slim.convolution2d(net, 32, 5, 2, padding='SAME', scope='adv_conv2', reuse=reuse)
            net = slim.max_pool2d(net, 3, 3, scope='adv_pool2')
            print '\tadv pool2', net.get_shape()

            # net = slim.convolution2d(net, 32, 5, 2, padding= 'VALID', scope='adv_conv3')
            # net = slim.max_pool2d(net, 2, scope='adv_pool3')
            # print '\tadv pool3', net.get_shape()

            net = slim.flatten(net, scope='adv_flat')
            print '\tadv flat', net.get_shape()
            net = slim.fully_connected(net, 512, scope='adv_fc1', reuse=reuse)
            print '\tadv fc', net.get_shape()
            ## Add some bias = [1,0] so that all images are called fake at first
            net = slim.fully_connected(net, 2, scope='adv_output', activation_fn=None, reuse=reuse)
            print '\tadv out', net.get_shape()
        return net



    """ Implements the "basic" strategy from Luc, et al

    The adversary should be able to distinguish between real and fake segmentations
    By including the fake prediction in the segmentation related loss,
    we can game the training to produce fakes that maximally look like the real ones

    As Luc et al point out, if the learning rate for the adversary is too high
    it can learn some trivial distinguisher like the absolute values (1.0, vs 0.0 < x < 1.0)
    so we set a low learning rate to give the segmentation net a chance to catch up.
    """
    def _init_adversarial_loss(self):
        self.adv_lambda = tf.constant(1.5)

        with tf.name_scope('Adversarial') as scope:
            print 'Adversarial real input'
            self.real_adv = self._adversarial_net_fn(self.input_y_onehot, reuse=False)
            print 'Adversarial generated mask'
            self.fake_adv = self._adversarial_net_fn(self.y_hat, reuse=True)

        ## Trainig objectives for real and fake images
        self.real_ex = tf.one_hot(tf.ones_like(tf.argmax(self.real_adv, 1)), 2)
        self.fake_ex = tf.one_hot(tf.zeros_like(tf.argmax(self.fake_adv, 1)), 2)

        ## Real should all be real
        self.l_bce_real = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.real_ex, logits=self.real_adv))

        ## Fakes should all be fake
        self.l_bce_fake = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.fake_ex, logits=self.fake_adv))

        ## Flip the direction for traning
        ## "maximize the prob. that the fake images are called real"
        l_bce_fake_one = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits( labels=self.real_ex, logits=self.fake_adv))
            ## << the fake one is used for segmentation loss & shouldnt be updated
        self.l_bce_fake_one = tf.stop_gradient(l_bce_fake_one, name='stop_grad')

        self.bce_real_summary = tf.summary.scalar('l_bce_real', self.l_bce_real)
        self.bce_fake_summary = tf.summary.scalar('l_bce_fake', self.l_bce_fake)
        self.bce_fake_one_summary = tf.summary.scalar('l_bce_fake_one', self.l_bce_fake_one)



    """ Initialize the training ops

    creates self.train_op_list with the proper list of training ops

    """
    def _init_training_ops(self):
        if self.mode == 'INFERENCE': return
        self.seg_optimizer = tf.train.AdamOptimizer(self.learning_rate, name='segAdam')

        if self.adversarial_training:
            print 'Using adversarial training'
            ## Turns out the key is to have a second optimizer for the adversarial net
            self.adversarial_optimizer = tf.train.AdamOptimizer(1e-7, name='advAdam')

            self._init_xentropy_loss()
            self._init_adversarial_loss()

            ## \sum_{n=1}^{N} l_{mce}(s(x_n), y_n) + \lambda (a(x_n, s(x_n)), 1)
            ## --------- or ----------
            ## \sum_{n=1}^{N} l_{mce}(s(x_n), y_n) - \lambda (a(x_n, s(x_n)), 0)
            self.seg_loss_op = self.xentropy_loss_op + self.adv_lambda*self.l_bce_fake_one
            # self.seg_loss_op = self.xentropy_loss_op - self.adv_lambda*self.l_bce_fake
            self.seg_train_op = self.seg_optimizer.minimize(self.seg_loss_op)

            ## \sum_{n=1}^{N} l_{bce}(a(x_n, y_n), 1) + l_{bce}(a(x_n, s(x_n)), 0)
            self.adv_loss_op = self.l_bce_real + self.l_bce_fake
            self.adv_train_op = self.adversarial_optimizer.minimize(self.adv_loss_op)
            # self.adv_train_op = self.seg_optimizer.minimize(self.adv_loss_op)

            ## \sum_{n=1}^{N} l_{mce}(s(x_n), y_n) - ...
            ##      \lambda [l_{bce}(a(x_n, y_n), 1) + l_{bce}(a(x_n, s(x_n)), 0)]
            self.loss_op = self.xentropy_loss_op - self.adv_lambda*(self.l_bce_real + self.l_bce_fake)

            self.loss_summary = tf.summary.scalar('loss', self.loss_op)
            self.seg_loss_summary = tf.summary.scalar('seg_loss', self.seg_loss_op)
            self.adv_loss_summary = tf.summary.scalar('adv_loss', self.adv_loss_op)
            self.train_op_list = [self.adv_train_op,
                                  self.seg_train_op,
                                #   self.adv_train_op,
                                  self.gs_increment]
            # self.train_op_list = [self.train_op, self.gs_increment]

        else:
            print 'Using standard x-entropy training'
            self._init_xentropy_loss()
            self.loss_op = self.xentropy_loss_op
            self.train_op = self.seg_optimizer.minimize(self.loss_op)
            self.train_op_list = [self.train_op, self.gs_increment]

            self.loss_summary = tf.summary.scalar('loss', self.loss_op)


    """ Initialize testing ops

    Decide if there's a testing dataset, or if we just run the net forward
    """
    def _init_testing(self):
        if self.test_dataset is None:
            self.test_ops = [self.loss_op]
            return

        ## When we have a test dataset
        self.test_dataset.set_tf_sess(self.sess)
        self.test_x = self.test_dataset.image_op
        self.test_y = self.test_dataset.mask_op
        self.test_y_onehot = tf.one_hot(self.test_y, self.n_classes)

        self.test_y_hat = self.model(self.test_x, reuse=True)
        self.test_output = tf.argmax(self.test_y_hat, axis=-1)
        self.test_output = tf.expand_dims(tf.cast(self.test_output, tf.float32), -1)

        if self.IN_OUT_CROP:
            target = self.test_y_hat.get_shape().as_list()[1]
            self.test_output = tf.image.resize_image_with_crop_or_pad(self.test_output, target, target)
        elif self.IN_OUT_RATIO:
            target_h, target_w = self.test_y_hat.get_shape().as_list()[1:3]
            self.test_output = tf.image.resize_bilinear(tf.test_output, [target_h, target_w])

        self.test_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.test_y_onehot, logits=self.test_y_hat))
        self.test_loss = tf.Print(self.test_loss, ['TEST LOSS', self.test_loss])

        self.test_y_hat_summary = tf.summary.image('test_out', self.test_output, max_outputs=3)
        self.test_y_summary = tf.summary.image('test_y', tf.cast(self.test_y, tf.float32), max_outputs=3)
        self.test_x_summary = tf.summary.image('test_x', self.test_x, max_outputs=3)
        self.test_loss_summary = tf.summary.scalar('test_loss', self.test_loss)
        self.test_ops = [self.test_y_hat, self.test_loss]

    """ Initializer for summary ops

    Adds some generic summary ops which we're pretty sure can always be called
    creates self.summary_op

    The previous initializers should create their own summary ops
    And this function should be called right before init_op
    """
    def _init_summary_ops(self):
        if self.mode == 'INFERENCE': return

        self.gt_summary = tf.summary.image('input_y', tf.cast(self.input_y, tf.float32), max_outputs=3)
        self.input_summary = tf.summary.image('input_x', self.input_x, max_outputs=3)
        self.output_summary = tf.summary.image('y_hat', self.output, max_outputs=3)
        # self.output_summary = tf.summary.image('mask', self.y_hat_sig, max_outputs=4)

        self.summary_op = tf.summary.merge_all()
#/END initializers




#/START methods
    def write_summary(self, op):
        summary_str = self.sess.run([op])[0]
        self.summary_writer.add_summary(summary_str,
            tf.train.global_step(self.sess, self.global_step))

    def train_step(self):
        ## Skip for inference mode
        if self.mode == 'INFERENCE':
            return

        _ = self.sess.run(self.train_op_list)
        ## Check if we should summarize --- I think it takes a lot of time
        gs = tf.train.global_step(self.sess, self.global_step)
        if gs % self.summary_iter == 0:
            self.write_summary(self.summary_op)
            # summary_str = self.sess.run([self.summary_op])[0]
            # self.summary_writer.add_summary(summary_str,
            #     tf.train.global_step(self.sess, self.global_step))


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

        print 'Testing'
        loss = self.sess.run(self.test_ops)
        self.write_summary(self.summary_op)


    """ Accepts a 4D nparray as input

    NOTE inference_ops set by the child class

    """
    def infer(self, imgs):
        feed_dict = {self.input_x: imgs}
        output = self.sess.run(self.inference_ops, feed_dict=feed_dict)

        return output
