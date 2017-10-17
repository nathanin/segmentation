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
import sys, time, random, cv2, os, glob
import tensorflow as tf
import numpy as np

sys.path.insert(0, '../models')
from deconvolution import DeconvModel
from unet import UNetModel
from fcn import FCNModel

sys.path.insert(0, '../utils')
from datasets import ImageMaskDataSet, load_images

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# feat_dir = '/Users/nathaning/_original_data/nuclei/pca/img'
# mask_dir = '/Users/nathaning/_original_data/nuclei/pca/mask'
# image_ext = 'png'

feat_dir = '../../data/feature'
mask_dir = '../../data/label'
image_ext = 'jpg'

img_list = sorted(glob.glob(os.path.join(feat_dir, '*.'+image_ext)))
inference_dir = './adv/inference'

itert=0
test_iter = 50
batch_size = 64
crop_size = 256


with tf.Graph().as_default():
    with tf.Session(config=config) as sess:

        """ TRAINING MODE
        Dataset used to define batch and input shape
        """
        dataset = ImageMaskDataSet(feat_dir, mask_dir,
            image_ext  = image_ext,
            n_classes  = 2,
            batch_size = batch_size,
            crop_size  = crop_size,
            ratio      = 0.5,
            capacity   = 7500,
            min_holding= 1250,
            threads    = 8)


        network = DeconvModel(
            sess = sess,
            dataset = dataset,
            n_classes = 2,
            save_dir = './adv/snapshots',
            log_dir = './adv/logs/{}'.format(itert),
            load_snapshot = True,
            learning_rate = 1e-4,
            n_kernels = 32,
            adversarial_training = True)

        # network = UNetModel(
        #     sess = sess,
        #     dataset = dataset,
        #     n_classes = 2,
        #     save_dir = './unet/snapshots',
        #     log_dir = './unet/logs/{}'.format(itert),
        #     load_snapshot = True,
        #     mode = 'TRAINING')

        # network = FCNModel(
        #     sess = sess,
        #     dataset = dataset,
        #     n_classes = 2,
        #     learning_rate = 1e-4,
        #     fcn_type = '8s',
        #     save_dir = './fcn/snapshots',
        #     log_dir = './fcn/logs/{}'.format(itert),
        #     load_snapshot = True,
        #     mode = 'TRAINING')

        ## Has to come after init_op ???
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print 'Train/test loop:'
        tstart = time.time()
        for _ in range(50):
            tloop = time.time()
            for k in range(500):
                network.train_step()
                if k % test_iter == 0:
                    print 'Testing... loss={:3.5f}\tTime:{}'.format(network.test(),
                        time.time()-tloop)

            network.snapshot()

            """ INFERENCE MODE
            There is a problem when re-instantiating the net but it totally works like this

            probably don't have to anyway.. inference flag just stops things
            from being instantiated
            can still call model.infer() from TRAINING mode

            .......... was INFERENCE mode unnecessary ???
            """
            img_tensor = load_images(img_list, batch_size, crop_size)
            output = network.infer(img_tensor)

            for k in range(output[0].shape[0]):
                img = np.squeeze(output[0][k,:,:,1])
                img_ = np.squeeze(img_tensor[k,:,:,:])

                outname = os.path.join(inference_dir, '{}.jpg'.format(k))
                outname_ = os.path.join(inference_dir, '{}_.jpg'.format(k))
                cv2.imwrite(outname, img*255)
                cv2.imwrite(outname_, img_[:,:,::-1]*255)
        #/end training loop

        print 'Time: {}'.format(time.time()-tstart)
        print 'Done'
        coord.request_stop()
        coord.join(threads)
