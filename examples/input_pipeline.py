#!/home/nathan/envs/tensorflow/bin/python

import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import time

'''
Read data from a queue
'''


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

feat_dir = '/home/nathan/segmentation/nuclei/data/feature'
mask_dir = '/home/nathan/segmentation/nuclei/data/label'
debugdir = '/home/nathan/segmentation/nuclei/debug'

crop_size = 512
process_size = 128
batch_size = 128


seed = 5555
image_names = tf.convert_to_tensor(sorted(glob.glob(
    os.path.join(feat_dir, '*.jpg')
)))
mask_names = tf.convert_to_tensor(sorted(glob.glob(
    os.path.join(mask_dir, '*.png')
)))

feature_queue = tf.train.string_input_producer(
    image_names,
    shuffle=True,
    seed=seed)

mask_queue = tf.train.string_input_producer(
    mask_names,
    shuffle=True,
    seed=seed)

image_reader = tf.WholeFileReader()
image_key, image_file = image_reader.read(feature_queue)
image = tf.image.decode_image(image_file)

mask_reader = tf.WholeFileReader()
mask_key, mask_file = image_reader.read(mask_queue)
mask = tf.image.decode_image(mask_file)

image = tf.divide(image, 255)
mask = tf.divide(mask, 255)

## Do a random crop
## Important? Do the crop before batching.
image_mask = tf.concat([image, mask], 2)
image_mask = tf.random_crop(image_mask, [crop_size, crop_size, 4])
image, mask = tf.split(image_mask, [3,1], axis=2)

image, mask = tf.train.shuffle_batch([image, mask],
    # shapes=[(1000,1000,3), (1000,1000,1)],
    batch_size=48,
    capacity=5000,
    min_after_dequeue=1000)

# image_mask = tf.concat([image, mask], 3)
# image_mask = tf.random_crop(image_mask, [-1, crop_size, crop_size, 4])
# image, mask = tf.split(image_mask, [3, 1], axis=3)

image = tf.image.resize_bicubic(image, (process_size, process_size))
mask = tf.image.resize_bicubic(mask, (process_size, process_size))

### Everything above is graph stuff
with tf.Session(config=config) as sess:
    tf.local_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    time_start = time.time()
    image_tensor, mask_tensor = sess.run([image, mask])
    print 'Elapsed: {}'.format(time.time() - time_start)

    print image_tensor.shape
    print mask_tensor.shape

    ##
    for k in range(image_tensor.shape[0]):
        cv2.imwrite(os.path.join('./debug/img{}.jpg'.format(k)),
            cv2.cvtColor(image_tensor[k,:,:,:] * 255, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join('./debug/img{}_.jpg'.format(k)),
            mask_tensor[k,:,:,:] * 255)

    coord.request_stop()
    coord.join(threads)
