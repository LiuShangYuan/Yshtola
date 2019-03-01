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
                 model_save_path='model', pretrained_model='model/src_model-13000',
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
        (train_data, train_labels), (test_data, test_labels) = dataset.load_data(num_words=10000)

        train_lens = np.array([len(d) for d in train_data])
        train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                           value=0,
                                                           padding='post',
                                                           maxlen=self.model.max_seq_len)
        train_output = train_data
        train_output_lens = np.array([self.model.max_seq_len for d in train_output])

        test_data = test_data[:1000]
        test_labels = test_labels[:1000]
        test_lens = np.array([len(d) for d in test_data])

        test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=0,
                                                           padding='post',
                                                           maxlen=self.model.max_seq_len)

        test_output = test_data
        test_output_lens = np.array([self.model.max_seq_len for d in test_output])

        if split == 'train':
            print("TRAIN_DATA_LENS:", len(train_data))
            return train_data, train_lens, train_labels, train_output, train_output_lens
        elif split == 'test':
            print("TEST_DATA_LENS:", len(test_data))
            return test_data, test_lens, test_labels, test_output, test_output_lens
        else:
            print("Unknown split %s" % split)
            return None, None

    def load_trg_texts(self, split='train'):
        _, (test_data, test_labels) = dataset.load_data(num_words=10000)
        test_data = test_data[1000:]
        test_labels = test_labels[1000:]
        test_lens = np.array([len(d) for d in test_data])
        test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                               value=0,
                                                               padding='post',
                                                               maxlen=self.model.max_seq_len)
        test_output = test_data
        test_output_lens = np.array([self.model.max_seq_len for d in test_output])

        return test_data, test_lens, test_labels, test_output, test_output_lens

    def pretrain(self):
        train_texts, train_lens, train_labels, train_outs, train_out_lens = self.load_src_texts(split='train')
        test_texts, test_lens, test_labels, test_outs, test_out_lens = self.load_src_texts(split='test')

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
                batch_text_lens = train_lens[i*self.batch_size: (i+1)*self.batch_size]
                batch_labels = train_labels[i*self.batch_size: (i+1)*self.batch_size]
                batch_outputs = train_outs[i*self.batch_size: (i+1)*self.batch_size]
                batch_out_lens = train_out_lens[i*self.batch_size: (i+1)*self.batch_size]
                feed_dict = {model.texts: batch_texts,
                             model.text_lens: batch_text_lens,
                             model.labels: batch_labels,
                             model.outputs:batch_outputs,
                             model.output_lens:batch_out_lens}
                sess.run(model.train_op, feed_dict)

                if (step + 1) % 10 == 0:
                    summary, l, acc = sess.run([model.summary_op, model.loss, model.accuracy], feed_dict)     # C
                    # summary, l = sess.run([model.summary_op, model.loss], feed_dict)
                    # rand_idx
                    test_acc, _ = sess.run(fetches=[model.accuracy, model.loss],                              # C
                    # loss =sess.run(model.loss,
                                           feed_dict={model.texts: test_texts,
                                                      model.text_lens: test_lens,
                                                      model.labels: test_labels,
                                                      model.outputs: test_outs,
                                                      model.output_lens: test_out_lens})
                    summary_writer.add_summary(summary, step)
                    # print('Step: [%d/%d]  loss:[%.6f]  test_loss:[%.6f]' % (step+1, self.pretrain_iter, l, loss))
                    print('Step: [%d/%d]  loss: [%.6f]  train acc: [%.6f]  test acc: [%.6f]' \
                          % (step+1, self.pretrain_iter, l, acc, test_acc))           # C

                if (step+1) % 1000 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'src_model'), global_step=step+1)
                    print('src_model-%d saved...!' % (step+1))


    def train(self):
        # load dataset
        semt_texts, semt_lens, _, _, _ = self.load_src_texts()
        norm_texts, norm_lens, _, _, _ = self.load_trg_texts()

        # build model
        model = self.model
        model.build_model()

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)


        with tf.Session(config=self.config) as sess:

            # init G and D
            tf.global_variables_initializer().run()

            # restore variables of F
            print('Loading pretrain model F ...')
            # restorer = tf.train.Saver()
            # restorer.restore(sess, self.pretrained_model)
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()

            print('start training...!')
            f_interval=15
            for step in range(self.train_iter+1):

                i = step % int(semt_texts.shape[0] / self.batch_size)
                src_texts = semt_texts[i*self.batch_size:(i+1)*self.batch_size]
                src_text_lens = semt_lens[i*self.batch_size:(i+1)*self.batch_size]

                feed_dict = {model.texts: src_texts,
                             model.text_lens: src_text_lens,
                             model.batch_size: self.batch_size}

                sess.run(model.d_train_op_src, feed_dict)
                sess.run(model.g_train_op_src, feed_dict)
                sess.run(model.g_train_op_src, feed_dict)
                sess.run(model.g_train_op_src, feed_dict)
                sess.run(model.g_train_op_src, feed_dict)
                sess.run(model.g_train_op_src, feed_dict)
                sess.run(model.g_train_op_src, feed_dict)

                if step > 1600:
                    f_interval = 30

                if i % f_interval == 0:
                    sess.run(model.f_train_op_src, feed_dict)

                if (step+1) % 10 == 0:
                    summary, dl, gl, fl = sess.run([model.summary_op_src,
                                                    model.d_loss_src,
                                                    model.g_loss_src,
                                                    model.f_loss_src],
                                                   feed_dict)
                    summary_writer.add_summary(summary, step)
                    print('[Source] step: [%d/%d]  d_loss: [%.6f]  g_loss: [%.6f]  f_loss:[%.6f]' %
                          (step+1, self.train_iter, dl, gl, fl))

                # train the model for target domain T
                j = step % int(norm_texts.shape[0] / self.batch_size)
                trg_texts = norm_texts[j*self.batch_size:(j+1)*self.batch_size]
                trg_text_lens = norm_lens[j*self.batch_size:(j+1)*self.batch_size]

                feed_dict = {model.trg_texts: trg_texts,
                             model.trg_text_lens: trg_text_lens,
                             model.batch_size: self.batch_size}

                sess.run(model.d_train_op_trg, feed_dict)
                sess.run(model.d_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)

                if (step+1) % 10 == 0:
                    summary, dl, gl = sess.run([model.summary_op_trg,
                                                model.d_loss_trg,
                                                model.g_loss_trg],
                                               feed_dict)
                    summary_writer.add_summary(summary, step)
                    print('[Target] step: [%d/%d]  d_loss:[%.6f]  g_loss:[%.6f]' %
                          (step+1, self.train_iter, dl, gl))

                if (step+1) % 200 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'dtn'), global_step=step+1)
                    print('model/dtn%d saved...!' % (step+1))
