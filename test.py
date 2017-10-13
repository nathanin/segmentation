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

sys.path.insert(0, './models')
from deconvolution import DeconvModel
from unet import UNetModel

sys.path.insert(0, '.')
from utils import ImageMaskDataSet, load_images


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# init_op = tf.global_variables_initializer()

itert=1


with tf.Graph().as_default():
    with tf.Session(config=config) as sess:

        """ TRAINING MODE """
        dataset = ImageMaskDataSet('../data/feature', '../data/label',
            n_classes  = 2,
            batch_size = 16,
            crop_size  = 512,
            capacity   = 1000)

        # network = DeconvModel(
        #     sess = sess,
        #     dataset = dataset,
        #     n_classes = 2,
        #     save_dir = './deconv/snapshots',
        #     log_dir = './deconv/logs/{}'.format(itert),
        #     load_snapshot = True,
        #     mode = 'TRAINING')

        network = UNetModel(
            sess = sess,
            dataset = dataset,
            n_classes = 2,
            save_dir = './unet/snapshots',
            log_dir = './unet/logs/{}'.format(itert),
            load_snapshot = True,
            mode = 'TRAINING')

        ## Has to come after init_op ???
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print 'Train/test loop:'
        tstart = time.time()
        for _ in range(2):
            tloop = time.time()
            for _ in range(50):
                network.train_step()

            print 'Testing... loss={:3.5f}\tTime:{}'.format( network.test(),
                time.time()-tloop)
            network.snapshot()

        print 'Time: {}'.format(time.time()-tstart)
        print 'Done'
        coord.request_stop()
        coord.join(threads)




        """ INFERENCE MODE
        There is a problem when re-instantiating the net

        probably don't have to anyway.. inference flag just stops things
        from being instantiated
        can still call model.infer() from TRAINING mode

        ... INFERENCE mode was unnecessary ???
        """

        img_list = sorted(glob.glob('../data/feature/*jpg'))
        inference_dir = './unet/inference'
        # network = DeconvModel(
        #     sess=sess,
        #     save_dir = './unet/snapshots',
        #     mode='INFERENCE' )

        # network = UNetModel(
        #     sess = sess,
        #     save_dir = './unet/snapshots',
        #     mode = 'INFERENCE')

        img_tensor = load_images(img_list, 16, 512)
        output = network.infer(img_tensor)

        print len(output)
        print output[0].shape
        print output[1].shape

        for k in range(output[0].shape[0]):
            img = np.squeeze(output[0][k,:,:,1])
            img_ = np.squeeze(img_tensor[k,:,:,:])

            outname = os.path.join(inference_dir, '{}.jpg'.format(k))
            outname_ = os.path.join(inference_dir, '{}_.jpg'.format(k))
            cv2.imwrite(outname, img*255)
            cv2.imwrite(outname_, img_[:,:,::-1]*255)
