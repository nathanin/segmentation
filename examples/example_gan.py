import sys, time, random, cv2, os, glob, datetime
import tensorflow as tf
import numpy as np

sys.path.insert(0, '.')
from models.gan import GAN
from utils.datasets import ImageDataSet, MNISTDataSet, load_images

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

itert = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
experiment = 'gan'

# img_list = sorted(glob.glob(os.path.join(feat_test_dir, '*.'+image_ext)))
inference_dir = 'examples/{}/inference'.format(experiment)
log_dir = 'examples/{}/logs/{}'.format(experiment, itert)
save_dir = 'examples/{}/snapshots'.format(experiment)

test_iter = 100
batch_size = 64
crop_size = 128


# with tf.Graph().as_default():
with tf.Session(config=config) as sess:

    """ TRAINING MODE
    Dataset used to define batch and input shape
    """
    # dataset = ImageDataSet(feat_dir,
    #     image_ext  = image_ext,
    #     n_classes  = 2,
    #     batch_size = batch_size,
    #     crop_size  = crop_size,
    #     ratio      = 0.5,
    #     capacity   = batch_size*10,
    #     min_holding= batch_size*5,
    #     threads    = 8)
    #
    # test_dataset = ImageDataSet(feat_test_dir,
    #     image_ext  = image_ext,
    #     n_classes  = 2,
    #     batch_size = batch_size,
    #     crop_size  = crop_size,
    #     ratio      = 0.5,
    #     capacity   = batch_size*5,
    #     min_holding= batch_size,
    #     threads    = 4)

    dataset = MNISTDataSet('../MNIST_data',
        batch_size = 96)

    network = GAN(
        sess = sess,
        n_classes = 2,
        zed_dim = 64,
        n_kernels = 16,
        bayesian = False,
        dataset = dataset,
        gan_type = 'small',
        input_channel = 1,
        log_dir = log_dir,
        save_dir = save_dir,
        input_dims = [28,28],
        load_snapshot = False,
        learning_rate = 1e-5,
        adversarial_training = True)

    ## Has to come after init_op ???
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)


    """ Training loop.

    Call network.train_step() once for each global step.
    Insert testing / snapshotting however you want.
    """
    tstart = time.time()
    for _ in range(50):
        t_outer_loop = time.time()
        for k in range(1000):
            t_inner_loop = time.time()
            network.train_step()
            if k % test_iter == 0:
                print 'Dreaming'
                network.dream()

        print 'Time: {}'.format(time.time() - t_outer_loop)
        network.snapshot()

    #/end training loop

    print 'Time: {}'.format(time.time()-tstart)
    print 'Done'
    coord.request_stop()
    coord.join(threads)
