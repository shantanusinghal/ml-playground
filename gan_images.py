import argparse

import numpy as np
import tensorflow as tf
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SHAPE = [28, 28]

# Hyper-Parameters
LEAK = 0.2
KEEP_PROB = 0.5
DIM_Z = 100
DIM_Y = np.prod(IMAGE_SHAPE)


class DataDistribution(object):
    """
    Defines the real data distribution, which is the MNIST dataset
    """

    def __init__(self, data_dir):
        self.data = input_data.read_data_sets(data_dir, one_hot=True)

    def sample(self, batch_size):
        """
        :return: batch of samples of specified size from the MNIST training data
        """
        # TODO apply this tweak => 2 * (x_real - 0.5)
        batch, _ = self.data.train.next_batch(batch_size)
        return 2 * (batch - 0.5)

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

    def __init__(self):
        self.sample_dim = DIM_Z

    def sample(self, batch_size):
        """
        :return: batch of samples of specified size, each with a noise vector
        """
        return np.random.normal(0, 1, size=[batch_size, self.sample_dim])


def weight_variable(shape, mean=0.0, std_dev=0.1):
    return tf.get_variable(
        'w',
        dtype=tf.float32,
        initializer=tf.truncated_normal(shape=shape, mean=mean, stddev=std_dev),
        trainable=True)


def bias_variable(shape, ):
    return tf.get_variable(
        'b',
        dtype=tf.float32,
        initializer=tf.zeros(shape),
        trainable=True)


def linear(x, dim_in, dim_out, scope):
    with tf.variable_scope(scope or 'linear'):
        w = weight_variable([dim_in, dim_out])
        b = bias_variable([dim_out])
    return tf.matmul(x, w) + b


def relu(x, dim_in, dim_out, scope, leaky=False):
    with tf.variable_scope(scope or 'relu'):
        w = weight_variable([dim_in, dim_out])
        b = bias_variable([dim_out])
    if leaky:
        return tf.nn.relu(tf.matmul(x, w) + b)
    else:
        leak = LEAK
        x_ = tf.matmul(x, w)
        l1 = 0.5 * (1 + leak)
        l2 = 0.5 * (1 - leak)
        return l1 * x_ + l2 * tf.abs(x_)


def generator(z, dim_h1=150, dim_h2=300):
    """
    """
    # TODO add nn.dropout layer
    h1 = relu(z, DIM_Z, dim_h1, 'h1')
    h2 = relu(h1, dim_h1, dim_h2, 'h2')
    return tf.nn.tanh(linear(h2, dim_h2, DIM_Y, 'x'))


def discriminator(x, dim_h1=300, dim_h2=150):
    """
    """
    # TODO implement minibatch
    h1 = tf.nn.dropout(relu(x, DIM_Y, dim_h1, 'h1', leaky=True), KEEP_PROB)
    h2 = tf.nn.dropout(relu(h1, dim_h1, dim_h2, 'h2', leaky=True), KEEP_PROB)
    return tf.nn.sigmoid(linear(h2, dim_h2, 1, 'y'))


class GAN(object):
    def __init__(self, data, gen, batch, epoch, learning_rate, decay_rate, num_pre_train_steps, out):
        self.data = data
        self.gen = gen
        self.epoch = epoch
        self.batch = batch
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.num_pre_train_steps = num_pre_train_steps
        self.out = out
        self._create_model()

    def _create_model(self):
        """
        Build a GAN model that include two competing neural network models G and D (combination of D1 and D2)
        """
        # In order to make sure that D is providing useful gradient information to the G from the start,
        # we're going to pre-train D using a maximum likelihood objective.
        # We define the network for this pre-training step in scope D_pre.
        # with tf.variable_scope('D_pre'):
        #     self.pre_input = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
        #     self.pre_labels = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
        #     self.D_pre = discriminator(self.pre_input)
        #     self.pre_loss = tf.reduce_mean(tf.square(self.D_pre - self.pre_labels))
        #     self.pre_opt = tf.train.GradientDescentOptimizer(tf.train.exponential_decay(
        #         learning_rate=self.learning_rate,
        #         decay_steps=150,
        #         decay_rate=0.95,
        #         staircase=True))\
        #         .minimize(self.pre_loss)
        #
        # self.d_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_pre')

        with tf.variable_scope('Gen'):
            # samples from a noise distribution as input
            self.z = tf.placeholder(tf.float32, [None, DIM_Z])
            # the generator network passes the input (self.z) through the MLP
            self.G = generator(self.z)

        with tf.variable_scope('Disc') as scope:
            self.x = tf.placeholder(tf.float32, [None, DIM_Y])
            # Discriminator D1 takes in input x (sampled from the true data distribution)
            # and outputs the likelihood of this input belonging to the true data distribution
            self.D1 = discriminator(self.x)
            # because we're reusing the variable discriminator
            scope.reuse_variables()
            # Discriminator D1 takes in the fake data generated by G and outputs the
            # likelihood of this input belonging to the true data distribution
            self.D2 = discriminator(self.G)

        # When optimizing D, we want to define it's loss function such that it
        # maximizes the quantity D1 (which maps the distribution of true data) and
        # minimizes D2 (which maps the distribution of fake data)
        self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
        self.params_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')
        self.opt_d = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.decay_rate)\
            .minimize(self.loss_d, var_list=self.params_d)

        # When optimizing G, we want to define it's loss function such that it
        # maximizes the quantity D2 (in order to successfully fool D)
        self.loss_g = tf.reduce_mean(-tf.log(self.D2))
        self.params_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')
        self.opt_g = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.decay_rate)\
            .minimize(self.loss_g, var_list=self.params_g)

    def train(self):
        """
        To train the model, we draw samples from the data distribution and the noise distribution,
        and alternate between optimizing the parameters of D and G.
        """
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            # # pre-training discriminator using a mean-square error (MSE) loss function to
            # # fit D to the true data distribution
            # for step in xrange(self.num_pre_train_steps):
            #     d = (np.random.random(self.batch_size) - 0.5) * 10.0
            #     labels = norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)
            #     session.run([self.pre_loss, self.pre_opt], {
            #         self.pre_input: np.reshape(d, (self.batch_size, 1)),
            #         self.pre_labels: np.reshape(labels, (self.batch_size, 1))
            #     })
            # weights_d_pre = session.run(self.d_pre_params)
            #
            # # copy weights from pre-training over to new D network
            # for i, v in enumerate(self.d_params):
            #     session.run(v.assign(weights_d_pre[i]))

            for epoch in xrange(self.epoch):
                loss_g_sum = loss_d_sum = 0
                num_steps = self.data.num_training_examples() // self.batch

                for step in xrange(num_steps):
                    # update discriminator
                    x = self.data.sample(self.batch)
                    z = self.gen.sample(self.batch)
                    loss_d, _ = session.run([self.loss_d, self.opt_d], feed_dict={
                        self.x: x,
                        self.z: z
                    })
                    loss_d_sum += loss_d

                    # update generator
                    z = self.gen.sample(self.batch)
                    loss_g, _ = session.run([self.loss_g, self.opt_g], feed_dict={
                        self.z: z
                    })
                    loss_g_sum += loss_g

                print('{}: avg_d {}\tavg_g {}'.format(epoch, loss_d_sum / num_steps, loss_g_sum / num_steps))
                sample = session.run(self.G, feed_dict={self.z: self.gen.sample(1)})
                imsave(self.out % epoch, np.reshape(sample/2 + 0.5, IMAGE_SHAPE))


def main(args):
    model = GAN(
        DataDistribution(args.data_dir),
        GeneratorDistribution(),
        args.batch_size,
        args.epoch_size,
        args.learning_rate,
        args.decay_rate,
        args.num_pre_train_steps,
        args.out
    )
    model.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--learning-rate', type=int, default=0.0002,
                        help='the learning rate for training')
    parser.add_argument('--decay-rate', type=int, default=0.5,
                        help='the learning rate for training')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--minibatch', type=bool, default=False,
                        help='use minibatch discrimination')
    parser.add_argument('--epoch-size', type=int, default=100,
                        help='size of each epoch')
    parser.add_argument('--num-pre-train-steps', type=int, default=1000,
                        help='number of pre-training steps')
    parser.add_argument('--out', type=str,
                        default='/Users/shantanusinghal/workspace/spike/gan/out/sample_%04d.jpg',
                        help='output location for writing samples from G')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())