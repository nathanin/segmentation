import sys, time, random, cv2, os, glob, datetime
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

feat_dir = '../../data/feature'
image_ext = 'jpg'

img_list = sorted(glob.glob(os.path.join(feat_dir, '*.'+image_ext)))
inference_dir = './adv/inference'

# itert=4
itert = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
test_iter = 100
batch_size = 72
crop_size = 256


# with tf.Graph().as_default():
with tf.Session(config=config) as sess:
    network = DeconvModel(
        sess = sess,
        mode = 'INFERENCE',
        n_classes = 2,
        load_snapshot = True,
        load_snapshot_from = './adv/snapshots/deconvolution.ckpt',
        n_kernels = 16,
        bayesian = True,)

    ## Add process for batching
    print 'test loop:'
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
