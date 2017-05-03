import argparse
import tensorflow as tf
import model

from util import DataDistribution


tf.logging.set_verbosity(tf.logging.DEBUG)


def main(args):
    dcgan = model.DCGAN(
        DataDistribution(args.data_dir),
        args.batch_size,
        args.epoch_size,
        args.learning_rate,
        args.decay_rate,
        args.log_every,
        args.job_dir
    )
    dcgan.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True,
                        # default='/tmp/tensorflow/mnist/input_data',
                        # default='https://storage.googleapis.com/uw-cs760-dcgan/mnist/input_data/',
                        help='Directory for storing input data')
    parser.add_argument('--job-dir', type=str, required=True,
                        # default='/Users/shantanusinghal/workspace/spike/gan/tensorboard/',
                        # default='gs://uw-cs760-dcgan/log/',
                        help='Directory for logging tensorboard data')
    parser.add_argument('--learning-rate', type=int, default=0.0002,
                        help='the learning rate for training')
    parser.add_argument('--decay-rate', type=int, default=0.5,
                        help='the learning rate for training')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--epoch-size', type=int, default=100,
                        help='size of each epoch')
    parser.add_argument('--log-every', type=int, default=10,
                        help='step size for saving tensorboard summaries')

    args, unknown = parser.parse_known_args()
    tf.logging.warn('Unknown arguments: {}'.format(unknown))
    return args


if __name__ == "__main__":
    main(parse_args())
