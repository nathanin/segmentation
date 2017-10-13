# Tensorflow Segmentation Networks

I've noticed that most implementations include just one network architecture.
They're usually very specialized and there are considerable code gaps between data --> model --> application.
So what do I do?
I write yet another solution (Fig. 1)

![https://imgs.xkcd.com/comics/standards.png](https://imgs.xkcd.com/comics/standards.png)
Figure 1. The competing standards conundrum.


## Structure

Big assumption: your training data and masks are in two folders, named alike, and exist as individual images.
Support for some more advanced data structures could probably be plug-and-play.
As-is, one can implement a training / application pipeline as so:

```
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
- Generalizable BaseModel class holds functions and hyperparameters for training, saving, tensorboard and inference
- To implement a new model, copy + paste an existing `__init__` function, and implement the `model()` method
- Support for customizable losses & multi-loss learning
- Should be robust to input sizes (untested)


#### Comments & Feedback
Please use the issues section or email directly <ing.nathany@gmail.com>
