import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


class DataDistribution(object):
    """
    Defines the real data distribution, which is the MNIST dataset
    """

    def __init__(self, data_dir):
        self.data = input_data.read_data_sets(data_dir, one_hot=True)

    def train(self, batch_size):
        """
        :return: batch of sample images and labels of specified size from the MNIST training data
        """
        return self.data.train.next_batch(batch_size)

    def validation(self, batch_size):
        """
        :return: batch of sample images and labels of specified size from the MNIST training data
        """
        return self.data.validation.next_batch(batch_size)

    def num_training_examples(self):
        return self.data.train.num_examples


# TODO try MNIST fake data
class GeneratorDistribution(object):
    """
    Define the generator's input noise distribution using stratified sampling - the samples are first generated
    uniformly over a specified range, and then randomly perturbed.

    This better aligns the input space with the target space and makes the transformation as smooth as possible
    and easier to learn. Stratified sampling also increases the representativeness the entire training space.
    """

    def __init__(self, sample_dim):
        self.sample_dim = sample_dim

    def image_samples(self, batch_size):
        """
        :return: batch of samples of specified size, each with a noise vector
        """
        return np.random.normal(0, 1, size=[batch_size, self.sample_dim])


def linear(x_input, dim_in, dim_out, name='linear'):
    """
    Builds a fully connected layer of neurons and returns their activations as computed by a
    matrix multiplication followed by a bias offset.

    Args:
        x_input:
        dim_in:
        dim_out:
        name:

    Returns:

    """
    with tf.variable_scope(name):
        w = weight_variable([dim_in, dim_out])
        b = bias_variable([dim_out])

        return tf.matmul(x_input, w) + b


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)


def concat(x, y):
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def weight_variable(shape, name='w', mean=0.0, std_dev=0.1):
    """
    Returns a trainable weight variable that is randomly initialized from a normal distribution.

    Args:
        shape: shape of the weight variable
        name: optional name for the variable as a string
        mean: mean of the random values to generate, a python scalar or a scalar tensor
        std_dev: standard deviation of the random values to generate, a python scalar or a scalar tensor

    Returns:
        A newly created or existing variable.
    """
    return tf.get_variable(
        name=name,
        shape=shape,
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(mean=mean, stddev=std_dev),
        trainable=True)


def bias_variable(shape, name='b'):
    """
    Returns a trainable bias variable that is initialized with constant values of 0.

    Args:
        shape: shape of the bias variable
        name: optional name for the variable as a string

    Returns:
        A newly created or existing variable.
    """
    return tf.get_variable(
        name=name,
        shape=shape,
        dtype=tf.float32,
        initializer=tf.constant_initializer(0),
        trainable=True)


def deconv2d(input_, output_dim,
             name='deconv2',
             filter_height=5,
             filter_width=5,
             strides=None):
    """
    Transpose (gradient) of the tf.nn.conv2d operation.

    Args:
         input_:
         output_dim:
         name:
         filter_height:
         filter_width:
         strides:

    Returns:
        A tensor ...
    """
    if strides is None:
        strides = [1, 2, 2, 1]
    bias_shape = [output_dim[-1]]
    in_channels = output_dim[-1]
    out_channels = input_.get_shape().as_list()[-1]
    filter_shape = [filter_height, filter_width, in_channels, out_channels]

    with tf.variable_scope(name):
        f_w = weight_variable(filter_shape)
        b = bias_variable(bias_shape)

        return tf.nn.bias_add(
            tf.nn.conv2d_transpose(input_, filter=f_w, output_shape=output_dim, strides=strides), b)


def cnn_block(x_image, out_channels,
              name='cnn_block',
              filter_height=5,
              filter_width=5,
              conv_stride=None,
              ksize=None,
              pool_stride=None):
    """
    Block of three key operations that form the basic building blocks of every
    Convolutional Neural Network (CNN)
        1. Convolution (conv2d)
        2. Non Linearity (ReLU)
        3. Pooling or Sub-Sampling (avg_pool)

    Args:
        x_image: input images as a matrix of pixel values, float-32 - [batch, in_height, in_width, in_channels]
        out_channels:
        name:
        filter_height:
        filter_width:
        conv_stride:
        ksize:
        pool_stride:

    Returns:
        A Tensor with the same type as value. The convoluted-rectified-average_pooled output tensor.
    """
    if pool_stride is None:
        pool_stride = [1, 2, 2, 1]
    if ksize is None:
        ksize = [1, 2, 2, 1]
    if conv_stride is None:
        conv_stride = [1, 1, 1, 1]
    bias_shape = [out_channels]
    in_channels = x_image.get_shape().as_list()[-1]
    filter_shape = [filter_height, filter_width, in_channels, out_channels]

    with tf.variable_scope(name):
        f_w = weight_variable(filter_shape)
        b = bias_variable(bias_shape)
        # slide the filter over the image to build a feature map
        feat_map = tf.nn.bias_add(
            tf.nn.conv2d(input=x_image, filter=f_w, strides=conv_stride, padding='SAME'), b)
        # ReLU is applied on each pixel to introduce non-linearity
        rect_feat_map = lrelu(feat_map)
        # average pooling used to reduce dimensionality
        sub_sample = tf.nn.avg_pool(rect_feat_map, ksize=ksize, strides=pool_stride, padding='SAME')

        return sub_sample
