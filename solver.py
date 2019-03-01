import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import os
import scipy.io
import scipy.misc

from tensorflow import keras
dataset = keras.datasets.imdb

class Solver(object):

    def __init__(self, model, batch_size=100, pretrain_iter=20000, train_iter=2000, sample_iter=100,
                 src_dir='src', trg_dir='trg', log_dir='logs', sample_save_path='sample',
                 model_save_path='model', pretrained_model='model/pretrain-20000',
                 test_model='model/dtn_1800'):

        self.model = model
        self.batch_size = batch_size

        self.pretrain_iter = pretrain_iter
        self.train_iter = train_iter
        self.sample_iter = sample_iter

        self.src_dir = src_dir
        self.trg_dir = trg_dir
        self.log_dir = log_dir

        self.sample_save_path = sample_save_path
        self.model_save_path = model_save_path
        self.pretrained_model = pretrained_model
        self.test_model = test_model

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

    def load_src_texts(self, split='train'):
        (train_data, train_labels), (test_data, test_labels) = dataset.load_data(num_words=10000, maxlen=self.model.max_seq_len)

        train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                           value=0,
                                                           padding='post',
                                                           maxlen=self.model.max_seq_len)
        test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=0,
                                                           padding='post',
                                                           maxlen=self.model.max_seq_len)

        if split == 'train':
            print("TRAIN_DATA_LENS:", len(train_data))
            return train_data, train_labels
        elif split == 'test':
            print("TEST_DATA_LENS:", len(test_data))
            return test_data, test_labels
        else:
            print("Unknown split %s" % split)
            return None, None

    def load_trg_texts(self, split='train'):
        pass

    def pretrain(self):
        train_texts, train_labels = self.load_src_texts(split='train')
        test_texts, test_labels = self.load_src_texts(split='test')

        model = self.model
        model.build_model()

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

            num_of_batches = train_texts.shape[0] // self.batch_size

            for step in range(self.pretrain_iter+1):
                i = step % num_of_batches
                # if i == 0: shuffle
                batch_texts = train_texts[i*self.batch_size: (i+1)*self.batch_size]
                batch_labels = train_labels[i*self.batch_size: (i+1)*self.batch_size]
                feed_dict = {model.texts: batch_texts, model.labels: batch_labels}
                sess.run(model.train_op, feed_dict)

                if (step + 1) % 10 == 0:
                    summary, l, acc = sess.run([model.summary_op, model.loss, model.accuracy], feed_dict)
                    # rand_idx
                    test_acc, _ = sess.run(fetches=[model.accuracy, model.loss],
                                           feed_dict={model.texts: test_texts,
                                                      model.labels: test_labels})
                    summary_writer.add_summary(summary, step)
                    print('Step: [%d/%d]  loss: [%.6f]  train acc: [%.2f]  test acc: [%.2f]' \
                          % (step+1, self.pretrain_iter, l, acc, test_acc))

                if (step+1) % 1000 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'src_model'), global_step=step+1)
                    print('src_model-%d saved...!' % (step+1))


    def train(self):
        pass