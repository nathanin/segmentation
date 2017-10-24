# Tensorflow Segmentation Networks

I've noticed that most implementations include just one network architecture.
They're usually very specialized and there are considerable code gaps between data --> model --> application.
So what do I do?
I write another solution (Fig. 1)

![https://imgs.xkcd.com/comics/standards.png](https://imgs.xkcd.com/comics/standards.png)

Figure 1. The competing standards conundrum.

## Models
- Generic convolution / deconvolution model
- FCN-{32, 16, 8}s ([Long et al, 2014](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf))
- U-Net ([Ronneberger et al, 2015](https://arxiv.org/pdf/1505.04597.pdf))
- Autoencoder, multi-scale autoencoder with latent vector (squishes signal through 1D bottleneck)
  - Adversarial autoencoder ([Makzani & Frey, 2015](https://arxiv.org/pdf/1511.05644.pdf))
  - Optional variational autoencoder mode (testing)
  - Next: conditional variational autoencoders
- Generative Adversarial Networks ([Makhzani & Frey, 2017](https://arxiv.org/pdf/1706.00531.pdf), [Goodfellow et al, 2014](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf), [Radford & Metz, 2017](https://arxiv.org/pdf/1511.06434.pdf))
  - InfoGAN ([Chen et al, 2016](http://papers.nips.cc/paper/6399-infogan-interpretable-representation-learning-by-information-maximizing-generative-adversarial-nets.pdf); [openai implementation](https://github.com/openai/InfoGAN))
  - Credit to [carpedm20's GAN code](https://github.com/carpedm20/DCGAN-tensorflow)
- Autoencoder mode for all models (separate from standalone autoencoder)
- Adversarial training for all models ([Luc et al, 2016](https://arxiv.org/pdf/1611.08408.pdf))
- Bayesian mode for all models ([Kendall & Gal, 2017](https://arxiv.org/pdf/1703.04977.pdf))

<!-- @article{kendall2017uncertainties,
  title={What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?},
  author={Kendall, Alex and Gal, Yarin},
  journal={arXiv preprint arXiv:1703.04977},
  year={2017}
} -->

## Structure
Big assumption (my use case): training data and masks are in two folders, named alike, and exist as individual images.
Support for some more advanced data structures or the TFRecords format.. forthcoming.
For now, recommend to expand `ImageMaskDataSet` and `ImageDataSet` as necessary.
But there is threading for the I/O and potential for custom on-the-fly augmentation with tf, so that's still nice.

As-is, one can implement a training / application pipeline as so:
```
import tensorflow as tf
from models.unet import UNetModel

[options] = ... ## Settings TODO: document these like with a picture or something
features_path, labels_path = ... ## Source data
paths = ... ## Some images for testing

with tf.Session() as sess:
    ## Training
    dataset = ImageMaskDataSet(features_path, labels_path, [options])
    model = UNetModel(sess, dataset, [options])

    for _ in range(iters):
      model.train_step()

    model.snapshot()

    ## Inference
    image_tensor = load_images(paths, [options])
    inference = model.infer(image_tensor)
```

Data structure example:
```
data/
  features/
    000.jpg
    001.jpg

  labels/
    000.png
    001.png
```


### Notes
- All the ugly stuff is hidden in the `BaseModel` class with functions and hyperparameters for training, saving, tensorboard and inference
- To implement a new model, copy + paste an existing `__init__` function, and implement the `model()` method
- Option to have a separate 'validation' set feed the validation ops. That part is super hacky. But I mean if it's hacky and it works it aint hacky.
<!-- - See my [other project](https://github.com/nathanin/histo-seg) for a potential use case outside of self-driving cars and benchmarking datasets. -->
- Adversarial training is pretty unstable, so I set a low learning rate for the adversarial net, and set it to update less frequently than the main net.


### Known things that don't work right
- Tensorboard only shows me the input pipeline? Where is my model?
- Input pipeline could & should & must be faster
- Logging is clunky; iterative experimentation is clunky


#### Comments, feedback, "this is just a bad version of X other repo":
Use the issues section (gotta get that sweet activity).
This research/educational implementation comes as is, with absolutely no warranty or support.
For other correspondence, please email directly (<ing.nathany@gmail.com>).
