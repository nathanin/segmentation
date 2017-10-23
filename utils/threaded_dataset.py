import tensorflow as tf
import numpy as np
import os, sys, shutil, glob, time, random, re, threading
# import pandas as pd
import cPickle as pickle
import sys

sys.path.insert(0, '../data/')
import image_utils



"""
Wrapper class

Modeled on tf MNIST example

Remember it pulls the number of grades from image_utils
"""

class Dataset:
    def __init__(self, images, labels, grades):
        self.images = images
        self.labels = labels
        self.grades = grades


class M0M1:
    def __init__(self, path, test_pct=0.2):
        assert os.path.exists(path), '{} not found'.format(path)
        self.path = path
        self.test_pct = test_pct
        self._init_dataset()

    """ split up self.*.images self.*.labels self.*.grades """
    def _init_dataset(self):
        dataset = image_utils.loadDataSet(self.path, label_type='m0m1')
        train, test, self.val_cases = image_utils.split_train_val(dataset, val_pct=self.test_pct)
        dataset = None

        self.train = Dataset(
            image_utils.extract_images(train),
            image_utils.extract_labels(train),
            image_utils.extract_grades(train))

        self.test = Dataset(
            image_utils.extract_images(test),
            image_utils.extract_labels(test),
            image_utils.extract_grades(test))



"""
The Generator class takes in data, which is an instance of threaded_dataset.Dataset:

n_sample is the number of elements to sample for each datum set
coord is tf thread coordinator
max_queue_size controls how large the queue can grow, therefore how much compute
    is being used to preprocess it
"""
class Generator:
    def __init__(self, data, n_sample, coord, max_queue_size=256):
        self.data = data
        self.n_sample = n_sample
        # self.n_classes = self.data.labels.shape[1]
        # self.n_grades = self.data.grades.shape[1]

        self.max_queue_size = max_queue_size
        self._init_queue()
        self.coord = coord


    def _init_queue(self):
        """ instantiate the tf. *queue object and the enqueue op """
        print 'Instantiating data queue'
        ## TODO: generalize shape initialization
        qshapes = [(self.n_sample, 42, 42, 1),
                   (2),
                   (4)]
                #   (self.n_classes),
                #    (self.n_grades)]
        self.queue = tf.FIFOQueue(self.max_queue_size, [tf.float32, tf.float32, tf.float32],
            shapes=qshapes)
        self.queue_size = self.queue.size()
        self.threads = []

        self.sample_X = tf.placeholder(dtype=tf.float32, shape=qshapes[0], name='inX')
        self.sample_Y = tf.placeholder(dtype=tf.float32, shape=qshapes[1], name='inY')
        self.sample_GR = tf.placeholder(dtype=tf.float32, shape=qshapes[2], name='inGR')
        self.enqueue = self.queue.enqueue([self.sample_X, self.sample_Y, self.sample_GR])

    def next_set(self):
        """ Go forever """
        # flag=True
        while True:
            # if self.coord.request_stop():
            #     flag=False
            #     break
            # else:
            yield self._generate()


    """ Originally from image_utils """
    def _generate(self):
        datakeys = [key for key in self.data.images.iterkeys()]
        target = datakeys[np.random.randint(len(datakeys))]

        label = np.squeeze(self.data.labels[target])
        grade = np.squeeze(self.data.grades[target])
        imgdata = self.data.images[target]
        n_target = imgdata.shape[0]

        imgdata = imgdata[np.random.choice(n_target, self.n_sample), ...]
        imgdata = imgdata.astype(np.float32)
        imgdata /= 255.0

        return imgdata, label, grade


    def _apply_augmentation(self, imgdata):
        pass


    def dequeue(self, num_elements):
        return self.queue.dequeue_many(num_elements)



    """ Constantly runs on a thread while queue.size < max_queue_size """
    def thread_main(self, sess):
        stop = False
        while not stop:
            try:
                batch_generator = self.next_set()
                for data in batch_generator:
                    ## Pause while full
                    while self.queue_size.eval(session=sess) == self.max_queue_size:
                        if self.coord.should_stop():
                            # stop= True
                            # print 'Full queue. Stop requested'
                            break
                        time.sleep(0.1)

                    if self.coord.should_stop():
                        stop = True
                        print 'Coord stop requested'
                        break

                    sess.run(self.enqueue,
                        feed_dict={self.sample_X:data[0], self.sample_Y:data[1], self.sample_GR:data[2]})
            except:
                print 'Hanging ?'
            finally:
                stop = True
                break


    def start_threads(self, sess, n_threads=1):
        print 'starting {} threads'.format(n_threads)
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

        return self.threads
