import argparse
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import tensorflow as tf


class DataDistribution(object):
    """
    Defines the real data distribution, a simple Gaussian with mean 4 and standard deviation of 0.5
    """
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self, N):
        """
        :return: N samples (sorted by value) from a simple Gaussian distribution with specified mean and standard deviation
        """
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples


class GeneratorDistribution(object):
    """
    Define the generator's input noise distribution using stratified sampling - the samples are first generated
    uniformly over a specified range, and then randomly perturbed.

    This better aligns the input space with the target space and makes the transformation as smooth as possible
    and easier to learn. Stratified sampling also increases the representativeness the entire training space.
    """

    def __init__(self, range):
        self.range = range

    def sample(self, N):
        """
        :return: N samples that are first generated uniformly over the specified range, and then randomly perturbed.
        """
        return np.linspace(-self.range, self.range, N) + \
               np.random.random(N) * 0.01


def linear(input, output_dim, scope=None, stddev=1.0):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b


def generator(input, h_dim):
    """
    :return: a linear transformation of the input passed through a nonlinearity (a softplus function),
    followed by another linear transformation
    """
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
    h1 = linear(h0, 1, 'g1')
    return h1


def discriminator(input, h_dim, minibatch_layer=True):
    """
    :return: a deep neural network, with h_dim number of dimensions.
    It uses tanh nonlinearities in all layers except the final one,
    which is a sigmoid (the output of which we can interpret as a probability).
    """
    h0 = tf.tanh(linear(input, h_dim * 2, 'd0'))
    h1 = tf.tanh(linear(h0, h_dim * 2, 'd1'))

    # without the minibatch layer, the discriminator needs an additional layer
    # to have enough capacity to separate the two distributions correctly
    if minibatch_layer:
        h2 = minibatch(h1)
    else:
        h2 = tf.tanh(linear(h1, h_dim * 2, scope='d2'))

    h3 = tf.sigmoid(linear(h2, 1, scope='d3'))
    return h3


def minibatch(input, num_kernels=5, kernel_dim=3):
    """
    Implements minibatch discrimination, that allows the discriminator to look at multiple samples (batch)
    at once in order to decide whether they come from the generator or the real data.

    :param input: output of some intermediate layer of the discriminator
    :param num_kernels:
    :param kernel_dim:
    :return:
    """
    # multiply input with a 3D tensor to produce a matrix (of size num_kernels x kernel_dim)
    x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)

    # Compute the L1-distance between rows in this matrix across all samples in a batch,
    # and then apply a negative exponential
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)

    # The minibatch features for the sample are then the sum of these exponential distances
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)

    # Concatenate the original input to the minibatch layer (the output of the previous discriminator layer)
    # with the newly created minibatch features, and pass this as input to the next layer of the discriminator
    return tf.concat(1, [input, minibatch_features])


def optimizer(loss, var_list, initial_learning_rate):
    """
    :param loss: A Tensor containing the value to minimize
    :param var_list: A list of Variable objects to update to minimize loss
    :param initial_learning_rate: A Python number specifying the initial learning rate
    :return: Optimizer that implements the gradient descent algorithm with exponential learning rate decay
    """
    decay = 0.95
    num_decay_steps = 150
    global_step = tf.Variable(0)
    # applies an exponential decay function to a provided initial learning rate
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        global_step,
        num_decay_steps,
        decay,
        staircase=True
    )

    # gradient descent optimizer that minimize loss by updating var_list
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=global_step,
        var_list=var_list
    )


class GAN(object):
    def __init__(self, data, gen, num_steps, batch_size, minibatch, log_every, num_pre_train_steps):
        self.data = data
        self.gen = gen
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.minibatch = minibatch
        self.log_every = log_every
        self.num_pre_train_steps = num_pre_train_steps

        self.mlp_hidden_size = 4
        self.learning_rate = 0.03

        # can use a higher learning rate when not using the minibatch layer
        if self.minibatch:
            self.learning_rate = 0.005
        else:
            self.learning_rate = 0.03

        self._create_model()

    def _create_model(self):
        """
        Build a GAN model that include two competing neural network models G and D (combination of D1 and D2)
        """
        # In order to make sure that D is providing useful gradient information to the G from the start,
        # we're going to pre-train D using a maximum likelihood objective.
        # We define the network for this pre-training step in scope D_pre.
        with tf.variable_scope('D_pre'):
            self.pre_input = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.pre_labels = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            D_pre = discriminator(self.pre_input, self.mlp_hidden_size, self.minibatch)
            self.pre_loss = tf.reduce_mean(tf.square(D_pre - self.pre_labels))
            self.pre_opt = optimizer(self.pre_loss, None, self.learning_rate)

        self.d_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_pre')

        with tf.variable_scope('Gen'):
            # samples from a noise distribution as input
            self.z = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            # the generator network passes the input (self.z) through the MLP
            self.G = generator(self.z, self.mlp_hidden_size)

        with tf.variable_scope('Disc') as scope:
            self.x = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            # Discriminator D1 takes in input x (sampled from the true data distribution)
            # and outputs the likelihood of this input belonging to the true data distribution
            self.D1 = discriminator(self.x, self.mlp_hidden_size, self.minibatch)
            # because we're reusing the variable discriminator
            scope.reuse_variables()
            # Discriminator D1 takes in the fake data generated by G and outputs the
            # likelihood of this input belonging to the true data distribution
            self.D2 = discriminator(self.G, self.mlp_hidden_size, self.minibatch)

        # When optimizing D, we want to define it's loss function such that it
        # maximizes the quantity D1 (which maps the distribution of true data) and
        # minimizes D2 (which maps the distribution of fake data)
        self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')
        self.opt_d = optimizer(self.loss_d, self.d_params, self.learning_rate)

        # When optimizing G, we want to define it's loss function such that it
        # maximizes the quantity D2 (in order to successfully fool D)
        self.loss_g = tf.reduce_mean(-tf.log(self.D2))
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')
        self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate)

    def train(self):
        """
        To train the model, we draw samples from the data distribution and the noise distribution,
        and alternate between optimizing the parameters of D and G.
        """
        with tf.Session() as session:
            tf.initialize_all_variables().run()

            # pre-training discriminator using a mean-square error (MSE) loss function to
            # fit D to the true data distribution
            for step in xrange(self.num_pre_train_steps):
                d = (np.random.random(self.batch_size) - 0.5) * 10.0
                labels = norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)
                session.run([self.pre_loss, self.pre_opt], {
                    self.pre_input: np.reshape(d, (self.batch_size, 1)),
                    self.pre_labels: np.reshape(labels, (self.batch_size, 1))
                })
            weights_d_pre = session.run(self.d_pre_params)

            # copy weights from pre-training over to new D network
            for i, v in enumerate(self.d_params):
                session.run(v.assign(weights_d_pre[i]))

            for step in xrange(self.num_steps):
                # update discriminator
                x = self.data.sample(self.batch_size)
                z = self.gen.sample(self.batch_size)
                loss_d, _ = session.run([self.loss_d, self.opt_d], {
                    self.x: np.reshape(x, (self.batch_size, 1)),
                    self.z: np.reshape(z, (self.batch_size, 1))
                })

                # update generator
                z = self.gen.sample(self.batch_size)
                loss_g, _ = session.run([self.loss_g, self.opt_g], {
                    self.z: np.reshape(z, (self.batch_size, 1))
                })

                if step % self.log_every == 0:
                    print('{}: {}\t{}'.format(step, loss_d, loss_g))

            self._plot_distributions(session)

    def _plot_distributions(self, session):
        db, pd, pg = self._samples(session)
        db_x = np.linspace(-self.gen.range, self.gen.range, len(db))
        p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))
        f, ax = plt.subplots(1)
        ax.plot(db_x, db, label='decision boundary')
        ax.set_ylim(0, 1)
        plt.plot(p_x, pd, label='real data')
        plt.plot(p_x, pg, label='generated data')
        plt.title('1D Generative Adversarial Network')
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend()
        plt.show()

    def _samples(self, session, num_points=10000, num_bins=100):
        """
        Return a tuple (db, pd, pg), where db is the current decision
        boundary, pd is a histogram of samples from the data distribution,
        and pg is a histogram of generated samples.
        """
        xs = np.linspace(-self.gen.range, self.gen.range, num_points)
        bins = np.linspace(-self.gen.range, self.gen.range, num_bins)

        # decision boundary
        db = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            db[self.batch_size * i:self.batch_size * (i + 1)] = session.run(self.D1, {
                self.x: np.reshape(
                    xs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1)
                )
            })

        # data distribution
        d = self.data.sample(num_points)
        pd, _ = np.histogram(d, bins=bins, density=True)

        # generated samples
        zs = np.linspace(-self.gen.range, self.gen.range, num_points)
        g = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            g[self.batch_size * i:self.batch_size * (i + 1)] = session.run(self.G, {
                self.z: np.reshape(
                    zs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1)
                )
            })
        pg, _ = np.histogram(g, bins=bins, density=True)

        return db, pd, pg


def main(args):
    model = GAN(
        DataDistribution(),
        GeneratorDistribution(range=8),
        args.num_steps,
        args.batch_size,
        args.minibatch,
        args.log_every,
        args.num_pre_train_steps
    )
    model.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=1200,
                        help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='the batch size')
    parser.add_argument('--minibatch', type=bool, default=False,
                        help='use minibatch discrimination')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    parser.add_argument('--num-pre-train-steps', type=int, default=1000,
                        help='number of pre-training steps')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
