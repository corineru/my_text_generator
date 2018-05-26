import tensorflow as tf
from my_read_utils import *
from my_model import *
import os

FLAGS = tf.app.flags

tf.app.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.app.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.app.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.app.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.app.flags.DEFINE_string('converter_path', 'model/default/converter.pkl', 'model/name/converter.pkl')
tf.app.flags.DEFINE_string('checkpoint_path', '', 'checkpoint path')
tf.app.flags.DEFINE_string('start_string', '五月的风', 'use this string to start generating')
tf.app.flags.DEFINE_integer('max_length', 30, 'max length to generate')


def main(_):
    converter = TextConverter(filename=FLAGS.converter_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    model = CharRNN(converter.vocab_size, sampling=True, lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size)

    model.load(FLAGS.checkpoint_path)

    start = converter.text_to_arr(FLAGS.start_string)
    arr = model.sample(FLAGS.max_length, start, converter.vocab_size)
    print(converter.arr_to_text(arr))


if __name__ =='__main__':
    tf.app.run()
