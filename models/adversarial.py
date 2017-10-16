import tensorflow as tf
import numpy as np
import sys, time, os

slim = tf.contrib.slim

""" Adversarial loss functions:

@article{luc2016semantic,
  title={Semantic segmentation using adversarial networks},
  author={Luc, Pauline and Couprie, Camille and Chintala, Soumith and Verbeek, Jakob},
  journal={arXiv preprint arXiv:1611.08408},
  year={2016}
}

network - a class extending BaseModel that has a method network.model()

"""

class AdversarialTraining(object):
    def __init__(self, network):
        pass
