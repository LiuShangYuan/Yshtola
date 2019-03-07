import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import os
import scipy.io
import scipy.misc
import json
from tqdm import tqdm
from nltk import word_tokenize

from tensorflow import keras
dataset = keras.datasets.imdb

class Solver(object):

    def __init__(self, model, batch_size=100, pretrain_iter=10000, train_iter=20000, sample_iter=100,
                 src_dir='src', trg_dir='trg', log_dir='logs', sample_save_path='sample',
                 model_save_path='model', ids_save_path='ids', pretrained_model='model/src_model-9000',
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
        self.ids_save_path = ids_save_path
        self.pretrained_model = pretrained_model
        self.test_model = test_model

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.word2idx = {}
        self.idx2word = []

        self.stop_words, self.semt_words = [], []
        self._load_semt_words()
        self._load_stop_words()

        self.content_words_mask = []

    def _load_stop_words(self):
        with open("./data/stop_words.dict") as fin:
            for line in fin:
                self.stop_words.append(line.strip())

    def _load_semt_words(self):
        with open("./data/sentiment_words.dict") as fin:
            for line in fin:
                self.semt_words.append(line.strip())

    def _get_content_words(self):
        for w in self.word2idx:
            if w in self.stop_words or w in self.semt_words or w in ['<pad>','<unk>','<bos>','<eos>']:
                self.content_words_mask.append(0)
            else:
                self.content_words_mask.append(1)

    def add_special_tokens(self, ids):
        ids = [self.word2idx['<bos>']] + ids  # self.idx2word[2] == '<bos>'
        if len(ids) > self.model.max_seq_len - 1:
            ids = ids[:self.model.max_seq_len - 1]
        return ids + [self.word2idx['<eos>']]

    def load_lm_texts(self, split='train',
                      name='lm',
                      train_path='./data/train_lm.json',
                      test_path='./data/test_lm.json'):

        # Load vocabulary
        if not tf.gfile.Exists(self.ids_save_path):
            tf.gfile.MakeDirs(self.ids_save_path)

        vocab_ids_path = os.path.join(self.ids_save_path, ('amazon_vocabs%d.ids') % self.model.vocab_size)
        if not tf.gfile.Exists(vocab_ids_path):
            # Create Vocabulary
            print("Vocabulary file %s not found. Creating new vocabs.ids file..." % vocab_ids_path)
            word2freq = {}
            dlist = json.load(open(train_path))
            for text in tqdm(dlist):
                for w in word_tokenize(text):
                    word2freq[w] = word2freq.get(w, 0) + 1

            sorted_dict = sorted(word2freq.items(), key=lambda item: item[1], reverse=True)
            sorted_dict = sorted_dict[:self.model.vocab_size - 4]

            self.word2idx = {'<pad>':0, '<unk>':1, '<bos>':2, '<eos>':3}
            self.idx2word = ['<pad>', '<unk>', '<bos>', '<eos>']
            for w, _ in sorted_dict:
                self.word2idx[w] = len(self.word2idx)
                self.idx2word.append(w)

            print('Save vocabularies into file %s ...' % vocab_ids_path)
            json.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word}, open(vocab_ids_path, 'w'),
                      ensure_ascii=False)

        else:
            # Load Vocabulary
            print('Loading vocabularies from %s ...' % vocab_ids_path)
            d_vocab = json.load(open(vocab_ids_path))
            self.word2idx = d_vocab['word2idx']
            self.idx2word = d_vocab['idx2word']

        if split == 'train':
            # Load train data -- for Language Model
            train_ids_path = os.path.join(self.ids_save_path, ('%s_train%d.ids' % (name, self.model.vocab_size)))
            if not tf.gfile.Exists(train_ids_path):
                print('train.ids file %s not found. Creating new train.ids file ...' % train_ids_path)
                d_train = json.load(open(train_path))
                d_train_ids = []

                for text in tqdm(d_train):
                    text_id = []
                    for w in word_tokenize(text):
                        text_id.append(self.word2idx.get(w, 1))# self.idx2word[1] == '<unk>'
                    d_train_ids.append(text_id)

                print('Save train.ids into file %s ...' % train_ids_path)
                json.dump(d_train_ids, open(train_ids_path, 'w'))
            else:
                print("Loading from train.ids file %s ..." % train_ids_path)
                d_train_ids = json.load(open(train_ids_path))

            train_data, train_outs, train_lens = [], [], []

            for ids in d_train_ids:
                ids = self.add_special_tokens(ids)
                train_data.append(ids)
                train_outs.append(ids[1:])
                train_lens.append(len(ids) - 1)

            train_data = keras.preprocessing.sequence.pad_sequences(sequences=train_data,
                                                                    maxlen=self.model.max_seq_len,
                                                                    padding='post',
                                                                    value=0)  # self.idx2word[2] == '<pad>'
            train_outs = keras.preprocessing.sequence.pad_sequences(sequences=train_outs,
                                                                    maxlen=self.model.max_seq_len,
                                                                    padding='post',
                                                                    value=0)
            train_lens = np.array(train_lens)

            print("Train data: %d sequences have been loaded." % len(train_data))

            return train_data, train_lens, train_outs

        else:
            # Load test data -- for Language Model
            test_ids_path = os.path.join(self.ids_save_path, ('%s_test%d.ids' % (name, self.model.vocab_size)))
            if not tf.gfile.Exists(test_ids_path):
                print('test.ids file %s not found. Creating new test.ids file ...' % test_ids_path)
                d_test = json.load(open(test_path))
                d_test_ids = []

                for text in tqdm(d_test):
                    text_id = []
                    for w in word_tokenize(text):
                        text_id.append(self.word2idx.get(w, 1))
                    d_test_ids.append(text_id)

                print('Save test.ids into file %s ...' % test_ids_path)
                json.dump(d_test_ids, open(test_ids_path, 'w'))
            else:
                print('Loading from test.ids file %s ...' % test_ids_path)
                d_test_ids = json.load(open(test_ids_path))

            test_data, test_outs, test_lens =[], [], []

            for ids in d_test_ids:
                ids = self.add_special_tokens(ids)
                test_data.append(ids)
                test_outs.append(ids[1:])
                test_lens.append(len(ids) - 1)

            test_data = keras.preprocessing.sequence.pad_sequences(sequences=test_data,
                                                                   maxlen=self.model.max_seq_len,
                                                                   padding='post',
                                                                   value=0)
            test_outs = keras.preprocessing.sequence.pad_sequences(sequences=test_outs,
                                                                   maxlen=self.model.max_seq_len,
                                                                   padding='post',
                                                                   value=0)
            test_lens = np.array(test_lens)

            print('Test data: %d sequences have been loaded.' % len(test_data))

            return test_data, test_lens, test_outs

    def load_src_texts(self, split='train',
                       name='vgame',
                       train_path='./data/train_vgame.json',
                       test_path='./data/test_vgame.json'):

        # Load vocabulary
        if not tf.gfile.Exists(self.ids_save_path):
            tf.gfile.MakeDirs(self.ids_save_path)

        vocab_ids_path = os.path.join(self.ids_save_path, ('amazon_vocabs%d.ids' % self.model.vocab_size))

        if not tf.gfile.Exists(vocab_ids_path):
            raise FileNotFoundError('Vocab.ids not found. Please run `python main.py -mode=\'pretrain\'` to generate the vocabulary file.')

        else:
            print('Loading vocabularies from %s ...' % vocab_ids_path)
            d_vocab = json.load(open(vocab_ids_path))
            self.word2idx = d_vocab['word2idx']
            self.idx2word = d_vocab['idx2word']

        if split == 'train':
            # Load train data -- for Source Language
            train_ids_path = os.path.join(self.ids_save_path, ('%s_train%d.ids' % (name, self.model.vocab_size)))

            if not tf.gfile.Exists(train_ids_path):
                print("Train.ids file %s not found. Creating new train.ids file..." % train_ids_path)

                d_train = json.load(open(train_path))
                d_train_ids = []

                for text in tqdm(d_train):
                    text_id = []
                    for w in word_tokenize(text):
                        text_id.append(self.word2idx.get(w, 1))     # self.idx2word[1] == '<unk>'
                    d_train_ids.append(text_id)

                print('Save train.ids into file %s ...' % train_ids_path)
                json.dump(d_train_ids, open(train_ids_path, 'w'))
            else:
                print("Loading from train.ids file %s ..." % train_ids_path)
                d_train_ids = json.load(open(train_ids_path))

            train_data, train_lens = [], []

            for ids in d_train_ids:
                ids = self.add_special_tokens(ids)
                train_data.append(ids)
                train_lens.append(len(ids)-1)

            train_data = keras.preprocessing.sequence.pad_sequences(sequences=train_data,
                                                                    maxlen=self.model.max_seq_len,
                                                                    padding='post',
                                                                    value=0)     # self.idx2word[2] == '<pad>'

            train_lens = np.array(train_lens)

            print("Train data: %d sequences have been loaded." % len(train_data))

            return train_data, train_lens

        elif split == 'test':

            # Load test data -- for Source Language
            test_ids_path = os.path.join(self.ids_save_path, ('%s_test%d.ids' % (name, self.model.vocab_size)))

            if not tf.gfile.Exists(test_ids_path):
                print("Test.ids file %s not found. Creating new test.ids file..." % test_ids_path)

                d_test = json.load(open(test_path))
                d_test_ids = []

                for text, label in tqdm(d_test):
                    text_id = []
                    for w in word_tokenize(text):
                        text_id.append(self.word2idx.get(w, 1))  # self.idx2word[1] == '<unk>'
                    d_test_ids.append((text_id, label))

                print('Save test.ids into file %s ...' % test_ids_path)
                json.dump(d_test_ids, open(test_ids_path, 'w'))
            else:
                print("Loading from test.ids file %s ..." % test_ids_path)
                d_test_ids = json.load(open(test_ids_path))

            test_data, test_outs, test_lens = [], [], []

            for ids in d_test_ids:
                ids = self.add_special_tokens(ids)
                test_data.append(ids)
                test_lens.append(len(ids)-1)

            test_data = keras.preprocessing.sequence.pad_sequences(sequences=test_data,
                                                                   maxlen=self.model.max_seq_len,
                                                                   padding='post',
                                                                   value=0)
            test_lens = np.array(test_lens)

            print("Test data: %d sequences have been loaded." % len(test_data))
            return test_data, test_lens

    def load_trg_texts(self, split='train',
                       name='books',
                       train_path='./data/train_books.json',
                       test_path='./data/test_books.json'):
        if split == 'train':
            # Load train data -- for Target Language
            train_ids_path = os.path.join(self.ids_save_path, ('%s_train%d.ids' % (name, self.model.vocab_size)))

            if not tf.gfile.Exists(train_ids_path):
                print('Train.ids file %s not found. Create new train.ids file ...' % train_ids_path)

                d_train = json.load(open(train_path))
                d_train_ids = []

                for text in tqdm(d_train):
                    text_id = []
                    for w in word_tokenize(text):
                        text_id.append(self.word2idx.get(w, 1))     # self.idx2word[1] == '<unk>'
                    d_train_ids.append(text_id)

                print('Save train.ids into file %s ...' % train_ids_path)
                json.dump(d_train_ids, open(train_ids_path, 'w'))
            else:
                print("Loading from train.ids file %s ..." % train_ids_path)
                d_train_ids = json.load(open(train_ids_path))

            train_data, train_outs, train_lens = [], [], []

            for ids in d_train_ids:
                ids = self.add_special_tokens(ids)
                train_data.append(ids)
                train_outs.append(ids[1:])
                train_lens.append(len(ids)-1)

            train_data = keras.preprocessing.sequence.pad_sequences(sequences=train_data,
                                                                    maxlen=self.model.max_seq_len,
                                                                    padding='post',
                                                                    value=0)  # self.idx2word[2] == '<pad>'
            train_outs = keras.preprocessing.sequence.pad_sequences(sequences=train_outs,
                                                                    maxlen=self.model.max_seq_len,
                                                                    padding='post',
                                                                    value=0)

            train_lens = np.array(train_lens)

            print("Train data: %d sequences have been loaded." % len(train_data))

            return train_data, train_lens, train_outs

        elif split == 'test':
            # Load test data -- for Target Language
            test_ids_path = os.path.join(self.ids_save_path, ('%s_train%d.ids' % (name, self.model.vocab_size)))

            if not tf.gfile.Exists(test_ids_path):
                print('Test.ids file %s not found. Create new test.ids file ...' % test_ids_path)

                d_test = json.load(open(test_path))
                d_test_ids = []

                for text in tqdm(d_test):
                    text_id = []
                    for w in word_tokenize(text):
                        text_id.append(self.word2idx.get(w, 1))
                    d_test_ids.append(text_id)

                print('Save test.ids into file %s ...' % test_ids_path)
                json.dump(d_test_ids, open(test_ids_path, 'w'))
            else:
                print('Loading from test.ids file %s ...' % test_ids_path)
                d_test_ids = json.load(open(test_ids_path))

            test_data, test_outs, test_lens = [], [], []

            for ids in d_test_ids:
                ids = self.add_special_tokens(ids)
                test_data.append(ids)
                test_outs.append(ids[1:])
                test_lens.append(len(ids)-1)

            test_data = keras.preprocessing.sequence.pad_sequences(sequences=test_data,
                                                                   maxlen=self.model.max_seq_len,
                                                                   padding='post',
                                                                   value=0)
            test_outs = keras.preprocessing.sequence.pad_sequences(sequences=test_outs,
                                                                   maxlen=self.model.max_seq_len,
                                                                   padding='post',
                                                                   value=0)
            test_lens = np.array(test_lens)

            print('Test data: %d sequences have been loaded.' % len(test_data))

            return test_data, test_lens, test_outs

    def pretrain(self):
        train_texts, train_lens, train_outs = self.load_lm_texts(split='train')
        test_texts, test_lens, test_outs = self.load_lm_texts(split='test')

        model = self.model
        model.build_model()

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

            num_of_batches = train_texts.shape[0] // self.batch_size

            for step in range(self.pretrain_iter+1):
                i = step % num_of_batches
                if i == 0:      # shuffle
                    shuffle_idx = np.random.permutation(train_texts.shape[0])
                    train_texts = train_texts[shuffle_idx]
                    train_outs = train_outs[shuffle_idx]
                    train_lens = train_lens[shuffle_idx]
                batch_texts = train_texts[i*self.batch_size: (i+1)*self.batch_size]
                batch_outs = train_outs[i*self.batch_size: (i+1)*self.batch_size]
                batch_text_lens = train_lens[i*self.batch_size: (i+1)*self.batch_size]
                feed_dict = {model.texts: batch_texts,
                             model.text_outs: batch_outs,
                             model.text_lens: batch_text_lens,
                             model.batch_size: self.batch_size}
                sess.run(model.train_op, feed_dict)

                if (step + 1) % 10 == 0:
                    summary, l, acc = sess.run([model.summary_op, model.loss, model.accuracy], feed_dict)
                    rand_idx = np.random.permutation(test_texts.shape[0])[:self.batch_size]
                    test_acc, _ = sess.run(fetches=[model.accuracy, model.loss],
                                           feed_dict={model.texts: test_texts[rand_idx],
                                                      model.text_outs: test_outs[rand_idx],
                                                      model.text_lens: test_lens[rand_idx],
                                                      model.batch_size: self.batch_size})
                    summary_writer.add_summary(summary, step)
                    print('Step: [%d/%d]  loss: [%.6f]  train acc: [%.6f]  test acc: [%.6f]' \
                          % (step+1, self.pretrain_iter, l, acc, test_acc))

                if (step+1) % 1000 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'src_model'), global_step=step+1)
                    print('src_model-%d saved...!' % (step+1))

                    rand_idx = np.random.permutation(test_texts.shape[0])[:5]
                    sampled_id = sess.run(fetches=model.sampled_id,
                                          feed_dict={model.texts: test_texts[rand_idx],
                                                     model.text_outs: test_outs[rand_idx],
                                                     model.text_lens: test_lens[rand_idx],
                                                     model.batch_size: 5})
                    target_id = test_outs[rand_idx]
                    for idx in range(5):
                        target_view = [self.idx2word[id_] for id_ in target_id[idx]]
                        sampled_view = [self.idx2word[id_] for id_ in sampled_id[idx]]
                        print("[VIEW-TARGET %d] %s" % (idx, " ".join(target_view)))
                        print("[VIEW-SAMPLE %d] %s" % (idx, " ".join(sampled_view)))


    def train(self):
        # load dataset
        semt_texts, semt_lens = self.load_src_texts()
        norm_texts, norm_lens, norm_outs = self.load_trg_texts()

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
            # variables_to_restore = slim.get_model_variables(scope='content_extractor')
            variables = slim.get_variables_to_restore(include=['word_vector', 'content_extractor', 'generator'], exclude=['optimizer_op_src'])
            variables = [v for v in variables if ('adam' not in v.name and 'trg_embedding' not in v.name)]
            print("=====Attention=====\n", [var.name for var in variables],'\n=================')
            restorer = tf.train.Saver(variables)
            restorer.restore(sess, self.pretrained_model)
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()

            self._get_content_words()

            print('start training...!')
            f_interval = 5
            for step in range(self.train_iter+1):

                i = step % int(semt_texts.shape[0] / self.batch_size)
                src_texts = semt_texts[i*self.batch_size:(i+1)*self.batch_size]
                src_text_lens = semt_lens[i*self.batch_size:(i+1)*self.batch_size]
                feed_dict = {model.src_texts: src_texts,
                             model.src_text_lens: src_text_lens,
                             model.content_mask: self.content_words_mask,
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
                    summary, dl, gl, glr, fl = sess.run([model.summary_op_src,
                                                    model.d_loss_src,
                                                    model.g_loss_src,
                                                    model.g_loss_src_recon,
                                                    model.f_loss_src],
                                                   feed_dict)
                    summary_writer.add_summary(summary, step)
                    print('[Source] step: [%d/%d]  d_loss: [%.6f]  g_loss: [%.6f]([%.6f])  f_loss:[%.6f]' %
                          (step+1, self.train_iter, dl, gl, glr, fl))

                # train the model for target domain T
                j = step % int(norm_texts.shape[0] / self.batch_size)
                trg_texts = norm_texts[j * self.batch_size:(j + 1) * self.batch_size]
                trg_text_outs = norm_outs[j * self.batch_size:(j + 1) * self.batch_size]
                trg_text_lens = norm_lens[j * self.batch_size:(j + 1) * self.batch_size]

                feed_dict = {model.trg_texts: trg_texts,
                             model.trg_text_lens: trg_text_lens,
                             model.trg_text_outs: trg_text_outs,
                             # model.is_quick_pretrain: False,
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

                if (step+1) % 50 == 0:
                    # saver.save(sess, os.path.join(self.model_save_path, 'dtn'), global_step=step+1)
                    # print('model/dtn%d saved...!' % (step+1))

                    rand_idx = np.random.permutation(semt_texts.shape[0])[:5]

                    sampled_id = sess.run(fetches=model.fake_texts,
                                          feed_dict={model.src_texts: semt_texts[rand_idx],
                                                     model.src_text_lens: semt_lens[rand_idx],
                                                     model.batch_size: 5})
                    target_id = semt_texts[rand_idx]
                    for idx in range(5):
                        target_view = [self.idx2word[id_] for id_ in target_id[idx]]
                        sampled_view = [self.idx2word[id_] for id_ in sampled_id[idx]]
                        print("[SRC VIEW-TARGET %d] %s" % (idx, " ".join(target_view)))
                        print("[SRC VIEW-SAMPLE %d] %s" % (idx, " ".join(sampled_view)))

                    rand_idx = np.random.permutation(norm_texts.shape[0])[:5]

                    sampled_id = sess.run(fetches=model.reconst_texts,
                                          feed_dict={model.trg_texts: norm_texts[rand_idx],
                                                     model.trg_text_lens: norm_lens[rand_idx],
                                                     model.trg_text_outs: norm_outs[rand_idx],
                                                     model.batch_size: 5})
                    target_id = norm_texts[rand_idx]
                    for idx in range(5):
                        target_view = [self.idx2word[id_] for id_ in target_id[idx]]
                        sampled_view = [self.idx2word[id_] for id_ in sampled_id[idx]]
                        print("[TRG VIEW-TARGET %d] %s" % (idx, " ".join(target_view)))
                        print("[TRG VIEW-SAMPLE %d] %s" % (idx, " ".join(sampled_view)))


