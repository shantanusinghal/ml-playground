import numpy as np
import tensorflow as tf

from datetime import datetime
from util import linear, lrelu, concat, deconv2d, cnn_block, GeneratorDistribution

DIM_Z = 100
# The MNIST dataset has 10 classes, representing the digits 0 through 9.
DIM_Y = 10
# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
DIM_IMAGE = np.prod([IMAGE_SIZE, IMAGE_SIZE])

# Hyper-Parameters
LEAK = 0.2
KEEP_PROB = 0.5


class DCGAN(object):
    def __init__(self, data, batch_size, epoch_size, learning_rate, decay_rate, log_every, job_dir):
        self.data = data
        self.gen = GeneratorDistribution(DIM_Z)
        self.num_epoch = epoch_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.log_every = log_every
        self.job_dir = job_dir

        self._create_learning_model()
        self._create_training_summary()
        self._create_validation_summary()

    def _create_learning_model(self):
        """
        Build a GAN model that include two competing neural network models G and D (combination of D1 and D2)
        """
        # placeholder for classification labels
        self.y = tf.placeholder(tf.float32, [None, DIM_Y])
        # placeholder for samples from a noise distribution
        self.z = tf.placeholder(tf.float32, [None, DIM_Z])
        # placeholder for samples from the true data distribution
        self.x = tf.placeholder(tf.float32, [None, DIM_IMAGE])

        # the generator network takes noise and target label
        self.G = generator(self.z, self.y, self.batch_size)
        # the discriminator network predicting the likelihood of true data distribution
        self.D_real, self.logits_D_real = discriminator(self.x, self.y, self.batch_size)
        # the discriminator network predicting the likelihood of generated (fake) data distribution
        self.D_fake, self.logits_D_fake = discriminator(self.G, self.y, self.batch_size, reuse=True)

        # When optimizing D, we want to define it's loss function such that it
        # maximizes the quantity D1 (which maps the distribution of true data) and
        # minimizes D2 (which maps the distribution of fake data)
        self.loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits_D_real, labels=tf.ones_like(self.D_real)))
        self.loss_D_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits_D_fake, labels=tf.zeros_like(self.D_fake)))
        self.loss_d = self.loss_D_real + self.loss_D_fake
        self.params_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.opt_d = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.decay_rate) \
            .minimize(self.loss_d, var_list=self.params_d)

        # When optimizing G, we want to define it's loss function such that it
        # maximizes the quantity D2 (in order to successfully fool D)
        self.loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits_D_fake, labels=tf.ones_like(self.D_fake)))
        self.params_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.opt_g = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.decay_rate) \
            .minimize(self.loss_g, var_list=self.params_g)

        # Define accuracy score for real and fake data
        self.accuracy_real = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.D_real, 1)), tf.float32))
        self.accuracy_fake = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.D_fake, 1)), tf.float32))

        # build a sampling net for validation
        labels_all = tf.get_variable('labels_all', shape=[10, 10], initializer=tf.constant_initializer([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ]))
        noise = tf.get_variable('noise',
                                shape=[10, DIM_Z],
                                initializer=tf.constant_initializer(self.gen.image_samples(10)))
        self.sampler = generator(noise, labels_all, reuse=True)
        self.sampler_accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(labels_all, 1),
            tf.argmax(discriminator(self.sampler, labels_all, reuse=True), 1)), tf.float32))

        # Define scalar summaries for training and validation
        with tf.variable_scope('shared'):
            tf.summary.scalar('loss_discriminator', self.loss_d)
            tf.summary.scalar('loss_discriminator_real', self.loss_D_real)
            tf.summary.scalar('loss_discriminator_fake', self.loss_D_fake)
            tf.summary.scalar('accuracy_real', self.accuracy_real)
            tf.summary.scalar('accuracy_fake', self.accuracy_fake)
            tf.summary.scalar('loss_generator', self.loss_g)

        # Define sample image summary for validation
        with tf.variable_scope('validation'):
            tf.summary.scalar('sample_accuracy', self.sampler_accuracy)
            tf.summary.image('sample_images', self.sampler, max_outputs=10)

    def _create_training_summary(self):
        self.training_summaries = tf.summary.merge([
            tf.get_collection(tf.GraphKeys.SUMMARIES, scope='generator'),
            tf.get_collection(tf.GraphKeys.SUMMARIES, scope='discriminator'),
            tf.get_collection(tf.GraphKeys.SUMMARIES, scope='shared')])

    def _create_validation_summary(self):
        self.validation_summaries = tf.summary.merge([
            tf.get_collection(tf.GraphKeys.SUMMARIES, scope='shared'),
            tf.get_collection(tf.GraphKeys.SUMMARIES, scope='validation')])

    def train(self):
        """
        To train the model, we draw samples from the data distribution and the noise distribution,
        and alternate between optimizing the parameters of D and G.
        """
        with tf.Session() as session:
            global_step = 0
            saver = tf.train.Saver()
            session.run(tf.global_variables_initializer())
            writer_train = tf.summary.FileWriter(self.job_dir + '/train', session.graph)
            writer_validate = tf.summary.FileWriter(self.job_dir + '/valid', session.graph)

            for epoch in xrange(self.num_epoch):
                # count_d_real, count_d_fake, count_g = 0, 0, 0
                num_steps = self.data.num_training_examples() // self.batch_size

                for step in xrange(num_steps):
                    global_step += 1
                    x, y = self.data.train(self.batch_size)
                    # update discriminator
                    loss_d, acc_r, acc_f, _, _, _ = session.run([self.loss_d, self.accuracy_real, self.accuracy_fake,
                                                                 self.loss_D_real, self.loss_D_fake, self.opt_d],
                                                                feed_dict={
                                                                    self.x: x,
                                                                    self.y: y,
                                                                    self.z: self.gen.image_samples(self.batch_size)})
                    # update generator
                    loss_g = session.run([self.loss_g, self.opt_g], feed_dict={
                        self.y: y,
                        self.z: self.gen.image_samples(self.batch_size)
                    })

                    print('[TRAINING] Epoch: %s Step: %s at %s [acc_x: %s acc_x\': %s loss_d: %s loss_g: %s]'
                          % (epoch, step, datetime.now(), loss_d, acc_r, acc_f, loss_g))

                    # at regular intervals record summaries for training and validation data
                    if step % self.log_every == 0:
                        summary = session.run(self.training_summaries, feed_dict={
                            self.x: x,
                            self.y: y,
                            self.z: self.gen.image_samples(self.batch_size),
                        })
                        writer_train.add_summary(summary, global_step)

                        x, y = self.data.validation(self.batch_size)
                        summary = session.run(self.validation_summaries, feed_dict={
                            self.x: x,
                            self.y: y,
                            self.z: self.gen.image_samples(self.batch_size),
                        })
                        writer_validate.add_summary(summary, global_step)

                # at each epoch save a checkpoint for all variables
                save_path = saver.save(session, self.job_dir, global_step=epoch)
                print("saved checkpoint to %s" % save_path)


def generator(z_input, y_label,
              batch_size=10,
              dim_con=64,
              dim_fc=1024,
              reuse=False):
    """
    Args:
        z_input: input noise tensor, float - [batch_size, DIM_Z=100]
        y_label: input label tensor, float - [batch_size, DIM_Y=10]
        batch_size:
        dim_con:
        dim_fc:
        reuse:
    Returns:
        x': the generated image tensor, float - [batch_size, DIM_IMAGE=784]
    """
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()

        # create z as the joint representation of the input noise and the label
        z = tf.concat([z_input, y_label], 1)

        tf.summary.histogram('act_g0', z)

        # first fully-connected layer
        g1 = tf.nn.relu(tf.contrib.layers.batch_norm(linear(
            x_input=z,
            dim_in=DIM_Z + DIM_Y,
            dim_out=dim_fc,
            name='g1'), epsilon=1e-5, scope='g1_bn'))

        # join the output of the previous layer with the labels vector
        g1 = tf.concat([g1, y_label], 1)

        tf.summary.histogram('act_g1', g1)

        # second fully-connected layer
        g2 = tf.nn.relu(tf.contrib.layers.batch_norm(linear(
            x_input=g1,
            dim_in=g1.get_shape().as_list()[-1],
            dim_out=dim_con * 2 * IMAGE_SIZE / 4 * IMAGE_SIZE / 4,
            name='g2'), epsilon=1e-5, scope='g2_bn'))

        # create a joint 4-D feature representation of the output of the previous layer and the label
        # to serve as a 7x7 input image for the next de-convolution layer
        y_ = tf.reshape(y_label, [batch_size, 1, 1, DIM_Y])
        g2 = tf.reshape(g2, [batch_size, IMAGE_SIZE / 4, IMAGE_SIZE / 4, dim_con * 2])
        g2 = concat(g2, y_)

        tf.summary.histogram('act_g2', g2)

        # first layer of deconvolution produces a larger 14x14 image
        g3 = deconv2d(g2, [batch_size, IMAGE_SIZE / 2, IMAGE_SIZE / 2, dim_con * 2], 'g3')

        # apply batch normalization to ___
        # apply ReLU to stabilize the output of this layer
        g3 = tf.nn.relu(tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='g3_bn'))

        # join the output of the previous layer with the labels vector
        g3 = concat(g3, y_)

        tf.summary.histogram('act_g3', g3)

        # second layer of deconvolution produces the final sized 28x28 image
        g4 = deconv2d(g3, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1], 'x')

        # no batch normalization in the final layer but a sigmoid activation function is used to
        # generate a sharp and crisp image vector; dimension - [28, 28, 1]
        g4 = tf.nn.sigmoid(g4)

        tf.summary.histogram('act_g4', g4)

        return g4


def discriminator(x_image, y_label,
                  batch_size=10,
                  dim_con=64,
                  dim_fc=1024,
                  reuse=False):
    """
    Returns the discriminator network. It takes an image and returns a real/fake classification across each label.
    The discriminator network is structured as a Convolution Neural Net with two layers of convolution and pooling,
    followed by two fully-connected layers.

    Args:
        x_image:
        y_label:
        batch_size:
        dim_con:
        dim_fc:
        reuse:

    Returns:
        The discriminator network.
    """
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        # create x as the joint 4-D feature representation of the image and the label
        y_4d = tf.reshape(y_label, [batch_size, 1, 1, DIM_Y])
        x_4d = tf.reshape(x_image, [batch_size, 28, 28, 1])
        x = concat(x_4d, y_4d)

        tf.summary.histogram('act_d0', x)

        # first convolution-activation-pooling layer
        d1 = cnn_block(x, 1 + DIM_Y, 'd1')

        # join the output of the previous layer with the labels vector
        d1 = concat(d1, y_4d)

        tf.summary.histogram('act_d1', d1)

        # second convolution-activation-pooling layer
        d2 = cnn_block(d1, dim_con + DIM_Y, 'd2')

        # flatten the output of the second layer to a 2-D matrix with shape - [batch, ?]
        d2 = tf.reshape(d2, [batch_size, -1])

        # join the flattened output with the labels vector and apply this as input to
        # a series of fully connected layers.
        d2 = tf.concat([d2, y_label], 1)

        tf.summary.histogram('act_d2', d2)

        # first fully connected layer
        d3 = tf.nn.dropout(lrelu(linear(
            x_input=d2,
            dim_in=d2.get_shape().as_list()[-1],
            dim_out=dim_fc,
            name='d3')), KEEP_PROB)

        # join the output of the previous layer with the labels vector
        d3 = tf.concat([d3, y_label], 1)

        tf.summary.histogram('act_d3', d3)

        # second and last fully connected layer
        # calculate the un-normalized log probability of each label
        d4_logits = linear(d3, dim_fc + DIM_Y, 1, 'd4')

        # calculate the activation values, dimension - [batch, 1]
        d4 = tf.nn.sigmoid(d4_logits)

        tf.summary.histogram('act_d4', d4)

        return d4, d4_logits
