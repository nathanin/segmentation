# Tensorflow Segmentation Networks

I've noticed that most implementations include just one network architecture.
They're usually very specialized and there are considerable code gaps between data --> model --> application.
So what do I do?
I write yet another solution (Fig. 1)

![https://imgs.xkcd.com/comics/standards.png](https://imgs.xkcd.com/comics/standards.png)

Figure 1. The competing standards conundrum.

## Models
- Generic convolution / deconvolution model
- FCN-{32, 16, 8}s
- U-Net

## Structure

Big assumption (my use case): training data and masks are in two folders, named alike, and exist as individual images.
Support for some more advanced data structures is probably plug-and-play.
But there is threading for the I/O and potential for on-the-fly augmentation, so that's still a plus.

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


## Features
- Hide all the ugly stuff in a `BaseModel` class with functions and hyperparameters for training, saving, tensorboard and inference
- To implement a new model, copy + paste an existing `__init__` function, and implement the `model()` method
- Should be robust to input sizes (untested)


#### Comments & Feedback
Please use the issues section to point where we can improve this thing.
For other correspondence, email directly <ing.nathany@gmail.com>
