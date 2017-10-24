#!/home/nathan/envs/tensorflow/bin/python

"""
Working towards the following API:

with tf.Session() as sess:
    ## Training
    dataset = image_dataset('path')
    model = modelClass(sess, dataset, hyperparams)
    model.train(settings)

    ## Or testing
    model = modelClass(sess, saved_weights)

    ## Inference
    testimg = load_img('path')
    output = model.inference(testimg)

"""
import sys, time, random, cv2, os, glob, datetime
import tensorflow as tf
import numpy as np

sys.path.insert(0, '.')
# sys.path.insert(0, 'models')
# from autoencoder import StackedAutoencoder
from models.autoencoder import Autoencoder

# sys.path.insert(0, 'utils')
# from datasets import ImageMaskDataSet, load_images
from utils.datasets import ImageDataSet, MNISTDataSet, load_images

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

itert = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
feat_dir = '/Users/nathaning/_original_data/nuclei/pca/img'
feat_test_dir = '/Users/nathaning/_original_data/nuclei/pca/img'
image_ext = 'png'
# feat_dir = '../data/feature'
# feat_test_dir = '../data/test/feature'
# image_ext = 'jpg'
experiment = 'auto'

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

    test_dataset = MNISTDataSet('../MNIST_data',
        batch_size = 96)

    network = Autoencoder(
        sess = sess,
        n_classes = 2,
        zed_dim = 8,
        n_kernels = 16,
        bayesian = False,
        dataset = dataset,
        input_channel = 1,
        log_dir = log_dir,
        variational = True,
        save_dir = save_dir,
        input_dims = [28,28],
        load_snapshot = False,
        learning_rate = 1e-3,
        encoder_type = 'small'
        test_dataset = test_dataset,
        adversarial_training = True)

    ## Has to come after init_op ???
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)



    """ Training loop.

    Call network.train_step() once for each global step.
    Insert testing / snapshotting however you want.
    """
    tstart = time.time()
    for _ in range(150):
        t_outer_loop = time.time()
        for k in range(1000):
            t_inner_loop = time.time()
            network.train_step()
            if k % test_iter == 0:
                network.test()
                network.dream()

        print 'Time: {}'.format(time.time() - t_outer_loop)
        network.snapshot()

        """ INFERENCE MODE
        There is a problem when re-instantiating the net but it totally works like this

        probably don't have to anyway.. inference flag just stops things
        from being instantiated
        can still call model.infer() from TRAINING mode

        .......... was INFERENCE mode unnecessary ???
        """
        # print 'Inferencing from current snapshot'
        # img_tensor = load_images(img_list, batch_size, crop_size)
        # output = network.infer(img_tensor)
        #
        # for k in range(output[0].shape[0]):
        #     img = np.squeeze(output[0][k,:,:,1])
        #     img_ = np.squeeze(img_tensor[k,:,:,:])
        #
        #     outname = os.path.join(inference_dir, '{}.jpg'.format(k))
        #     outname_ = os.path.join(inference_dir, '{}_.jpg'.format(k))
        #     cv2.imwrite(outname, img*255)
        #     cv2.imwrite(outname_, img_[:,:,::-1]*255)
    #/end training loop

    print 'Time: {}'.format(time.time()-tstart)
    print 'Done'
    coord.request_stop()
    coord.join(threads)
