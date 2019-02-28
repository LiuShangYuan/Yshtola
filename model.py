import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class DTN(object):

    def __init__(self, mode='train',
                 vocab_size=10000,
                 max_seq_len=50,
                 learning_rate=0.0003,
                 rho=15.0):
        self.mode = mode
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate

        self.rho = rho


    def generator(self, inputs, reuse=False):
        # TODO: Implement a decoder here
        return None

    def discriminator(self, texts, reuse=False):
        # TODO: Implement a text classifier here
        return None

    def ae_extractor(self, texts, reuse=False):
        # TODO: Implement an autoencoder here
        return None

    def build_model(self):

        if self.mode == 'pretrain':
            self.src_texts = tf.placeholder(tf.int64, [None, self.max_seq_len], 'src_texts')        # batch_size x seq_len

        elif self.mode == 'train':
            self.src_texts = tf.placeholder(tf.int64, [None, self.max_seq_len], 'src_texts')
            self.trg_texts = tf.placeholder(tf.int64, [None, self.max_seq_len], 'trg_texts')

            # source domain
            self.fx = self.ae_extractor(self.src_texts)
            self.fake_texts = self.generator(self.fx)
            self.logits = self.discriminator(self.fake_texts)
            self.fgfx = self.ae_extractor(self.fake_texts, reuse=True)

            # loss
            self.d_loss_src = slim.losses.sigmoid_cross_entropy(self.logits, tf.zeros_like(self.logits))
            self.g_loss_src = slim.losses.sigmoid_cross_entropy(self.logits, tf.ones_like(self.logits))
            self.f_loss_src = tf.reduce_mean(tf.square(self.fx - self.fgfx)) * 15.0

            # optimizer
            self.d_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            self.g_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            self.f_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)

            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'discriminator' in var.name]
            g_vars = [var for var in t_vars if 'generator' in var.name]
            f_vars = [var for var in t_vars if 'content_extractor' in var.name]

            # train op
            with tf.name_scope('source_train_op'):
                self.d_train_op_src = slim.learning.create_train_op(total_loss=self.d_loss_src,
                                                                    optimizer=self.d_optimizer_src,
                                                                    variables_to_train=d_vars)
                self.g_train_op_src = slim.learning.create_train_op(total_loss=self.g_loss_src,
                                                                    optimizer=self.g_optimizer_src,
                                                                    variables_to_train=g_vars)
                self.f_train_op_src = slim.learning.create_train_op(total_loss=self.f_loss_src,
                                                                    optimizer=self.f_optimizer_src,
                                                                    variables_to_train=f_vars)

            # summary op
            d_loss_src_summary = tf.summary.scalar('src_d_loss', self.d_loss_src)
            g_loss_src_summary = tf.summary.scalar('src_g_loss', self.g_loss_src)
            f_loss_src_summary = tf.summary.scalar('src_f_loss', self.f_loss_src)
            origin_texts_summary = tf.summary.text('src_origin_texts', self.src_texts)
            sampled_texts_summary = tf.summary.text('src_sampled_texts', self.fake_texts)
            self.summary_op_src = tf.summary.merge([d_loss_src_summary, g_loss_src_summary,
                                                    f_loss_src_summary, origin_texts_summary,
                                                    sampled_texts_summary])

            # target domain
            self.fx = self.ae_extractor(self.trg_texts, reuse=True)
            self.reconst_texts = self.generator(self.fx, reuse=True)
            self.logits_fake = self.discriminator(self.reconst_texts, reuse=True)
            self.logits_real = self.discriminator(self.trg_texts, reuse=True)

            # loss
            self.d_loss_fake_trg = slim.losses.sigmoid_cross_entropy(self.logits_fake, tf.zeros_like(self.logits_fake))
            self.d_loss_real_trg = slim.losses.sigmoid_cross_entropy(self.logits_real, tf.ones_like(self.logits_real))
            self.d_loss_trg = self.d_loss_fake_trg + self.d_loss_real_trg

            self.g_loss_fake_trg = slim.losses.sigmoid_cross_entropy(self.logits_fake, tf.ones_like(self.logits_fake))
            # self.g_loss_const_trg = tf.reduce_mean(tf.square(self.trg_texts - self.reconst_texts)) * self.rho
            self.g_loss_const_trg = - tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self._get_onehot(self.trg_texts),
                logits=self._get_onehot(self.reconst_texts)) * self.rho
            self.g_loss_trg = self.g_loss_fake_trg + self.g_loss_const_trg

            # optimizer
            self.d_optimizer_trg = tf.train.AdamOptimizer(self.learning_rate)
            self.g_optimizer_trg = tf.train.AdamOptimizer(self.learning_rate)

            # train op
            with tf.name_scope('trg_train_op'):
                self.d_train_op_trg = slim.learning.create_train_op(total_loss=self.d_loss_trg,
                                                                    optimizer=self.d_optimizer_trg,
                                                                    variables_to_train=d_vars)
                self.g_train_op_trg = slim.learning.create_train_op(total_loss=self.g_loss_trg,
                                                                    optimizer=self.g_optimizer_trg,
                                                                    variables_to_train=g_vars)

            # summary op
            d_loss_fake_trg_summary = tf.summary.scalar('trg_d_loss_fake', self.d_loss_fake_trg)
            d_loss_real_trg_summary = tf.summary.scalar('trg_d_loss_real', self.d_loss_real_trg)
            d_loss_trg_summary = tf.summary.scalar('trg_d_loss', self.d_loss_trg)
            g_loss_fake_trg_summary = tf.summary.scalar('trg_g_loss_fake', self.g_loss_fake_trg)
            g_loss_const_trg_summary = tf.summary.scalar('trg_g_loss_const', self.g_loss_const_trg)
            g_loss_trg_summary = tf.summary.scalar('trg_g_loss', self.g_loss_trg)

            origin_texts_summary = tf.summary.text('trg_origin_texts', self.trg_texts)
            sampled_texts_summary = tf.summary.text('trg_reconstructed_texts', self.reconst_texts)
            self.summary_op_src = tf.summary.merge([d_loss_trg_summary, g_loss_trg_summary,
                                                    d_loss_fake_trg_summary, d_loss_real_trg_summary,
                                                    g_loss_fake_trg_summary, g_loss_const_trg_summary,
                                                    origin_texts_summary, sampled_texts_summary])
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

    def _get_onehot(self, texts):

        assert type(texts) is np.ndarray

        return np.eye(self.vocab_size)[texts]