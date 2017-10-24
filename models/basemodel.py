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
                 autoencoder = False,
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
        self.autoencoder = autoencoder
        self.learning_rate = learning_rate
        self.input_channel = input_channel
        self.adversarial_training = adversarial_training
        self.batch_size = self.dataset.batch_size


        ## Flags for defining the relationship of input to output
        ## The child class should set one
        ## Wherever this is used IN_OUT_EQUAL should be no-action ..????
        self.IN_OUT_EQUAL = False
        self.IN_OUT_CROP = False
        self.IN_OUT_RATIO = False


        ## Argument parsing and if/else flag setting
        self.load_snapshot = load_snapshot if load_snapshot else False

        if self.mode=='INFERENCE':
            print 'NOTICE: INFERENCE MODE load snapshot forced True'
            self.load_snapshot = True

        self.load_snapshot_from = load_snapshot_from if load_snapshot_from else False

        if self.autoencoder:
            print 'AUTOENCODER MODE'
            print '\tSetting objective function to MSE'
            print '\tSetting n_classes to input_channel ({})'.format(input_channel)
            self.objective_fn = lambda y,y_hat: tf.losses.mean_squared_error(
                labels=y, predictions=y_hat)
            self.n_classes = self.input_channel
        else:
            print 'Setting objective function to Softmax x-entropy'
            nested_yprep_fn = lambda y: tf.squeeze(tf.one_hot(y, self.n_classes))
            self.objective_fn = lambda y,y_hat: tf.nn.softmax_cross_entropy_with_logits(
                labels=nested_yprep_fn(y), logits=y_hat)
        ## /end if/else


        ## Controls how often we write the summary. Summary writing impacts speed maybe?
        self.summary_iter = 25

        ## Always needed
        self.init_op = tf.global_variables_initializer()
        self._init_session(sess)





    """ Setup functions """
    def _init_session(self, sess):
        self.sess = sess
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.gs_increment = tf.assign_add(self.global_step, 1)

        if self.mode == 'INFERENCE':
            print 'INFERENCE MODE skipping dataset initializations'
            return

        self.dataset.set_tf_sess(self.sess)

        if self.log_dir is not None:
            self.summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)





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
            ## Probably not the best solution. at least catch some specific errors
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
        else:
            print 'Training from scratch. Set load_snapshot = True to resume training.'







    """ Instantiate input functions from a dataset """
    def _init_input(self):
        tensor_dims = [
            self.batch_size,
            self.input_dims[0],
            self.input_dims[1],
            self.input_channel ]

        if self.mode == 'INFERENCE':
            print 'INFERENCE MODE skipping input'
            ## TODO: deal with variable input shapes; call resize() before inference
            self.input_x = tf.placeholder('float', tensor_dims, name='in_x')
            return

        print 'Setting up TRAINING mode input ops'
        if self.dataset.use_feed:
            self.input_x = tf.placeholder('float32', tensor_dims, name='in_x')
        else:
            self.input_x = self.dataset.image_op

        if self.autoencoder:
            print 'AUTOENCODER mode settings input_y = input_x'
            self.input_y = self.input_x
        else:
            if dataset.has_masks:
                self.input_y = self.dataset.mask_op
            else:
                raise Exception('No dataset.mask_op found. Is it ImageMaskDataSet?')


        # self.input_y = tf.cast(input_y, tf.uint8)

        print 'input_x:', self.input_x.get_shape(), self.input_x.dtype
        print 'input_y:', self.input_y.get_shape(), self.input_y.dtype






    """ Instantiate basic loss function """
    def _init_xentropy_loss(self):
        if self.mode == 'INFERENCE':
            print 'INFERENCE MODE skipping loss'
            return

        # self.input_y_onehot = tf.squeeze(tf.one_hot(self.input_y, self.n_classes))
        # print 'input_y_onehot', self.input_y_onehot.get_shape()

        # self.xentropy_loss_op = tf.reduce_mean( self.objective_fn( y=self.input_y, y_hat=self.y_hat))
        self.xentropy_loss_op = self.objective_fn( y=self.input_y, y_hat=self.y_hat)

        self.xentropy_summary = tf.summary.scalar('seg_xentropy', self.xentropy_loss_op)




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
        - The training benefits from a strong adversary.
    """
    def _adversarial_net(self, tensor_in, reuse=False):
        n_kernels = 36
        dadv = 4
        with tf.variable_scope('adversary') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            ## Downsample this input for faster and more accurate
            h, w = tensor_in.get_shape().as_list()[1:3]
            print '\tadv input', tensor_in.get_shape(), tensor_in.dtype
            tensor_in = tf.image.resize_bilinear(tensor_in, [h//dadv, w//dadv])

            print '\tadv resize', tensor_in.get_shape(), tensor_in.dtype
            net = slim.convolution2d(tensor_in,
                num_outputs = n_kernels,
                kernel_size = 3,
                stride = 2,
                padding = 'VALID',
                scope = 'adv_conv1',
                reuse = reuse)
            net = slim.batch_norm(net, scope='adv_bn1', reuse=reuse)
            net = slim.max_pool2d(net, 2, 2, scope='adv_pool1')
            print '\tadv pool1', net.get_shape()

            net = slim.convolution2d(net, n_kernels*2, 3, 2, padding='VALID', scope='adv_conv2', reuse=reuse)
            net = slim.batch_norm(net, scope='adv_bn2', reuse=reuse)
            net = slim.max_pool2d(net, 2, 2, scope='adv_pool2')
            print '\tadv pool2', net.get_shape()

            ## Can never decide if i want this last convolution or not
            # net = slim.convolution2d(net, n_kernels*4, 3, 1, padding= 'VALID', scope='adv_conv3')
            # net = slim.max_pool2d(net, 2, 2, scope='adv_pool3')
            # print '\tadv pool3', net.get_shape()

            net = slim.flatten(net, scope='adv_flat')
            net = slim.batch_norm(net, scope='adv_bn3', reuse=reuse)
            print '\tadv flat', net.get_shape()

            net = slim.fully_connected(net, 1024, scope='adv_fc1', reuse=reuse)
            net = slim.batch_norm(net, scope='adv_bn4', reuse=reuse)
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
        self.adv_lambda = tf.constant(2.0)

        with tf.name_scope('Adversarial') as scope:
            print 'Adversarial real input'
            self.real_adv = self._adversarial_net_fn(self.input_y, reuse=False)
            print 'Adversarial generated mask'
            self.fake_adv = self._adversarial_net_fn(self.y_hat, reuse=True)

        ## Trainig objectives for real and fake images
        self.real_ex = tf.one_hot(tf.ones_like(tf.argmax(self.real_adv, 1)), 2)
        self.fake_ex = tf.one_hot(tf.zeros_like(tf.argmax(self.fake_adv, 1)), 2)

        ## Real should all be real
        self.l_bce_real = tf.nn.softmax_cross_entropy_with_logits(labels=self.real_ex, logits=self.real_adv)

        ## Fakes should all be fake
        self.l_bce_fake = tf.nn.softmax_cross_entropy_with_logits(labels=self.fake_ex, logits=self.fake_adv)

        l_bce_fake_one = tf.nn.softmax_cross_entropy_with_logits( labels=self.real_ex, logits=self.fake_adv)
        ## << the fake one is used for segmentation loss & should not pass gradients backwards
        self.l_bce_fake_one = tf.stop_gradient(l_bce_fake_one, name='stop_grad')

        self.bce_real_summary = tf.summary.scalar('l_bce_real', tf.reduce_mean(self.l_bce_real))
        self.bce_fake_summary = tf.summary.scalar('l_bce_fake', tf.reduce_mean(self.l_bce_fake))
        self.bce_fake_one_summary = tf.summary.scalar('l_bce_fake_one', tf.reduce_mean(self.l_bce_fake_one))
        # self.fake_detection = tf.summary.scalar('fake_acc', self.accuracy_fake_detection)
        # self.real_detection = tf.summary.scalar('real_acc', self.accuracy_real_detection)






    """ Initialize the training ops

    creates self.train_op_list with the proper list of training ops

    """
    def _init_training_ops(self):
        if self.mode == 'INFERENCE':
            print 'INFERENCE MODE skip training ops'
            return
        self.seg_optimizer = tf.train.AdamOptimizer(self.learning_rate, name='segAdam')

        if self.adversarial_training:
            print 'Using adversarial training'
            ## Turns out the key is to have a second optimizer for the adversarial net
            ## with a LOW learning rate!!
            self.adversarial_optimizer = tf.train.AdamOptimizer(
                self.adversarial_lr, name='advAdam')

            self._init_xentropy_loss()
            self._init_adversarial_loss()

            ## \sum_{n=1}^{N} l_{mce}(s(x_n), y_n) + \lambda (a(x_n, s(x_n)), 1)
            self.seg_loss_op = tf.reduce_mean(self.xentropy_loss_op + self.adv_lambda*self.l_bce_fake_one)

            ## \sum_{n=1}^{N} l_{bce}(a(x_n, y_n), 1) + l_{bce}(a(x_n, s(x_n)), 0)
            self.adv_loss_op = tf.reduce_mean(self.l_bce_real + self.l_bce_fake)

            ## ?? slim documentation says to do this for batch_norm layers
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.seg_train_op = self.seg_optimizer.minimize(self.seg_loss_op)
                self.adv_train_op = self.adversarial_optimizer.minimize(self.adv_loss_op)

            ## \sum_{n=1}^{N} l_{mce}(s(x_n), y_n) - ...
            ##      \lambda [l_{bce}(a(x_n, y_n), 1) + l_{bce}(a(x_n, s(x_n)), 0)]
            self.loss_op = tf.reduce_mean(self.xentropy_loss_op - self.adv_lambda*(self.l_bce_real + self.l_bce_fake))

            self.loss_summary = tf.summary.scalar('loss', self.loss_op)
            self.seg_loss_summary = tf.summary.scalar('seg_loss', self.seg_loss_op)
            self.adv_loss_summary = tf.summary.scalar('adv_loss', self.adv_loss_op)
            self.train_op_list = [self.seg_train_op,
                                  self.gs_increment]
            self.adversarial_train_list = [self.adv_train_op]
            # self.train_op_list = [self.train_op, self.gs_increment]

        else:
            print 'Using standard x-entropy training'
            self._init_xentropy_loss()
            self.seg_loss_op = tf.reduce_mean(self.xentropy_loss_op)
            self.seg_loss_summary = tf.summary.scalar('seg_loss', self.seg_loss_op)

            ## Needed to update batch_norm
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.seg_train_op = self.seg_optimizer.minimize(self.seg_loss_op)

            self.train_op_list = [self.seg_train_op, self.gs_increment]
            self.loss_summary = tf.summary.scalar('seg_loss', self.seg_loss_op)




    """ Initialize testing ops """
    def _init_testing(self):
        if self.mode == 'INFERENCE':
            print 'INFERENCE MODE skip testing ops'
            return

        if self.test_dataset is None:
            self.test_ops = [self.loss_op]
            return

        ## When we have a test dataset
        print 'Initializing TEST net (sharing weights ON)'
        self.test_dataset.set_tf_sess(self.sess)

        if self.test_dataset.use_feed:
            self.test_x = tf.placeholder('float', [self.batch_size, 28, 28, 1], name='test_x')
        else:
            self.test_x = self.test_dataset.image_op

        # self.test_y_onehot = tf.one_hot(self.test_y, self.n_classes)

        ## Turn on reuse
        ## Set training=False
        self.test_y_hat = self.model(self.test_x, reuse=True, training=False)

        if self.autoencoder:
            ## This use case is pixel-wise regression
            self.test_y = self.test_x
            self.test_output = self.test_y_hat
        else:
            ## The other use case is segmentation
            if not self.test_dataset.has_masks:
                raise Exception('No mask_op in test_dataset. Is it ImageMaskDataSet?')

            self.test_y = self.test_dataset.mask_op
            self.test_output = tf.argmax(self.test_y_hat, axis=-1)
            self.test_output = tf.expand_dims(tf.cast(self.test_output, tf.float32), -1)

        if self.IN_OUT_CROP:
            target = self.test_y_hat.get_shape().as_list()[1]
            self.test_output = tf.image.resize_image_with_crop_or_pad(self.test_output, target, target)
        elif self.IN_OUT_RATIO:
            target_h, target_w = self.test_y_hat.get_shape().as_list()[1:3]
            self.test_output = tf.image.resize_bilinear(tf.test_output, [target_h, target_w])

        ## Just use straight-up objective loss in testing
        self.test_loss = tf.reduce_mean(
            self.objective_fn(y=self.test_y, y_hat=self.test_y_hat))
        self.test_loss = tf.Print(self.test_loss, ['TEST LOSS', self.test_loss, self.global_step])

        self.test_y_hat_summary = tf.summary.image('test_out', self.test_output, max_outputs=3)
        self.test_y_summary = tf.summary.image('test_y', tf.cast(self.test_y, tf.float32), max_outputs=3)
        self.test_x_summary = tf.summary.image('test_x', self.test_x, max_outputs=3)
        self.test_loss_summary = tf.summary.scalar('test_loss', self.test_loss)
        self.test_ops = [self.test_y_hat, self.test_loss]

        ## I think it's sufficient to run test_summary_op since it has to compute
        ## all the things associated with it also
        self.test_summary_op = tf.summary.merge([
            self.test_y_hat_summary,
            # self.test_y_summary,
            self.test_x_summary,
            self.test_loss_summary ])




    def _init_summary_ops(self):

        raise Exception('USING DEPRECIATED SUMMARY CREATOR. The new way is to define summary ops in child classes')

#/END initializers


















#/START methods
    def write_summary(self, op, feed_dict=None):
        ## TODO check out if the listy-ness of this one is really needed
        if feed_dict:
            summary_str = self.sess.run([op], feed_dict=feed_dict)[0]
        else:
            summary_str = self.sess.run([op])[0]

        self.summary_writer.add_summary(summary_str,
            tf.train.global_step(self.sess, self.global_step))



    def train_step(self):
        raise Exception('ERROR depreciated train_step(). Implement train_step in child')

        # if self.dataset.use_feed:
        #     batch = self.dataset.mnist.train.next_batch(self.batch_size)[0]
        #     batch = self.dataset._reshape_batch(batch)
        #     feed_dict = {self.input_x: batch}
        #     _ = self.sess.run(self.train_op_list, feed_dict=feed_dict)
        #     self.write_summary(self.summary_op, feed_dict=feed_dict)
        #
        # else:
        #     _ = self.sess.run(self.train_op_list)
        #     self.write_summary(self.summary_op)




    def snapshot(self):
        if self.mode == 'INFERENCE':
            print 'snapshot() with INFERENCE mode invalid'
            return

        gs = tf.train.global_step(self.sess, self.global_step)
        print 'Global step {}, snapshotting to {}'.format(gs, self.save_path)
        self.saver.save(self.sess, self.save_path, global_step=gs)




    def test(self):
        ## Skip for inference mode
        if self.mode == 'INFERENCE':
            print 'test() with INFERENCE mode invalid'
            return
        if self.test_dataset.use_feed:
            batch = self.dataset.mnist.test.next_batch(self.batch_size)[0]
            batch = self.dataset._reshape_batch(batch)
            # print 'batch:', batch.shape, batch.dtype
            feed_dict = {self.test_x: batch/255.0}
            self.write_summary(self.test_summary_op, feed_dict=feed_dict)
        else:
            self.write_summary(self.test_summary_op)



    """ Accepts a 4D nparray as input

    NOTE inference_ops set by the child class

    """
    def infer(self, imgs):
        feed_dict = {self.input_x: imgs}
        output = self.sess.run(self.inference_ops, feed_dict=feed_dict)

        return output
