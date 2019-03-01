import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class DTN(object):

    def __init__(self, mode='train',
                 embedding_size = 64,
                 vocab_size=10000,
                 max_seq_len=64,
                 hidden_size=128,
                 num_classes=2,
                 num_filters=256,
                 kernel_size=5,
                 keep_prob=0.9,
                 learning_rate=0.0003,
                 extractor_type='autoencoder',
                 rho=15.0):
        self.mode = mode

        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes

        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.keep_prob = keep_prob

        self.learning_rate = learning_rate

        self.extractor_type = extractor_type

        self.rho = rho

        with tf.variable_scope('word_vector'):
            with tf.device('/cpu:0'):
                self.embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])


    def generator(self, inputs, reuse=False):
        # TODO: Implement a decoder here
        if self.mode == 'train':
            if self.extractor_type == 'autoencoder':
                with tf.device('/cpu:0'):
                    embedding_inputs = tf.nn.embedding_lookup(self.embedding, tf.ones([self.batch_size, self.max_seq_len], dtype=tf.int32))

                with tf.variable_scope('generator', reuse=reuse):

                    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)

                    helper = tf.contrib.seq2seq.TrainingHelper(
                        embedding_inputs,
                        tf.fill([self.batch_size], self.max_seq_len),
                        time_major=False)

                    projection_layer = tf.layers.Dense(self.vocab_size, use_bias=False)

                    decoder = tf.contrib.seq2seq.BasicDecoder(
                        decoder_cell, helper, inputs,
                        output_layer=projection_layer)

                    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                        decoder, maximum_iterations=self.max_seq_len)

                    logits = outputs.rnn_output
                    sample_id = outputs.sample_id

                return sample_id, logits

    def discriminator(self, texts, reuse=False):
        # with tf.device('/cpu:0'):
        #     embedding_inputs = tf.nn.embedding_lookup(self.embedding, texts)

        with tf.variable_scope('discriminator', reuse=reuse):

            with tf.variable_scope('cnn'):
                conv = tf.layers.conv1d(texts, self.num_filters, self.kernel_size, name='conv')
                gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

            with tf.variable_scope('score'):
                fc = tf.layers.dense(gmp, self.hidden_size, name='fc1')
                fc = tf.contrib.layers.dropout(fc, self.keep_prob)
                fc = tf.nn.relu(fc)
                fc = tf.layers.dense(fc, 2, name='fc2')

        return fc

    def classifier_extractor(self, texts, reuse=False):
        with tf.device('/cpu:0'):
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, texts)

        with tf.variable_scope('content_extractor', reuse=reuse):
            with tf.variable_scope('cnn'):
                conv = tf.layers.conv1d(embedding_inputs, self.num_filters, self.kernel_size, name='conv')
                gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

            with tf.variable_scope('score'):
                fc = tf.layers.dense(gmp, self.hidden_size, name='fc1')
                fc = tf.contrib.layers.dropout(fc, self.keep_prob)
                fc = tf.nn.relu(fc)

            if self.mode == 'pretrain':
                fc = tf.layers.dense(fc, self.num_classes, name='fc2')

        return fc

    def autoencoder_extractor(self, texts, text_lens, reuse=False):
        with tf.device('/cpu:0'):
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, texts)

        with tf.variable_scope('content_extractor', reuse=reuse):

            with tf.variable_scope('encoder'):
                encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)

                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    encoder_cell, embedding_inputs,
                    sequence_length=text_lens,
                    dtype=tf.float32,
                    time_major=False)
            if self.mode == 'pretrain':

                with tf.variable_scope('decoder'):
                    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)

                    helper = tf.contrib.seq2seq.TrainingHelper(
                        embedding_inputs, self.output_lens, time_major=False)

                    projection_layer = tf.layers.Dense(self.vocab_size, use_bias=False)

                    decoder = tf.contrib.seq2seq.BasicDecoder(
                        decoder_cell, helper, encoder_state,
                        output_layer=projection_layer)

                    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
                    logits = outputs.rnn_output
                    sample_id = outputs.sample_id


                return logits, sample_id

        return encoder_state

    def legacy_extractor(self, texts, reuse=False):
        # TODO: Firstly we implement a classifier here
        # with tf.device('/cpu:0'):
        #     # embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
        #     embedding_inputs = tf.nn.embedding_lookup(self.embedding, texts)
        #
        # with tf.variable_scope('content_extractor', reuse=reuse):
        #
        #     with tf.variable_scope('cnn'):
        #         conv = tf.layers.conv1d(embedding_inputs, self.num_filters, self.kernel_size, name='conv')
        #         gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
        #
        #     with tf.variable_scope('score'):
        #         fc = tf.layers.dense(gmp, self.hidden_size, name='fc1')
        #         fc = tf.contrib.layers.dropout(fc, self.keep_prob)
        #         fc = tf.nn.relu(fc)
        #
        #         if self.mode == 'pretrain':
        #             fc = tf.layers.dense(fc, self.num_classes, name='fc2')
        #
        #         return fc

        # TODO: Implement an autoencoder here
        with tf.device('/cpu:0'):
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, texts)

        with tf.variable_scope('content_extractor', reuse=reuse):

            with tf.variable_scope('encoder'):
                encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)

                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    encoder_cell, embedding_inputs,
                    sequence_length=self.text_lens,
                    dtype=tf.float32,
                    time_major=False)

            with tf.variable_scope('decoder'):
                decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)

                helper = tf.contrib.seq2seq.TrainingHelper(
                    embedding_inputs, self.output_lens, time_major=False)

                projection_layer = tf.layers.Dense(self.vocab_size, use_bias=False)


                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, helper, encoder_state,
                    output_layer=projection_layer)

                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
                logits = outputs.rnn_output

            with tf.variable_scope('classifier'):

                conv = tf.layers.conv1d(encoder_outputs, self.num_filters, self.kernel_size, name='conv')
                gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

            with tf.variable_scope('score'):
                fc = tf.layers.dense(gmp, self.hidden_size, name='fc1')
                fc = tf.contrib.layers.dropout(fc, self.keep_prob)
                fc = tf.nn.relu(fc)

                if self.mode == 'pretrain':
                    fc = tf.layers.dense(fc, self.num_classes, name='fc2')

            if self.mode == 'pretrain':
                return logits, fc

            return encoder_outputs, encoder_state

    def build_model(self):

        if self.mode == 'pretrain':
            self.texts = tf.placeholder(tf.int32, [None, self.max_seq_len], 'src_texts')        # batch_size x seq_len
            self.text_lens = tf.placeholder(tf.int32, [None,], 'src_text_lens')
            self.labels = tf.placeholder(tf.int64, [None,], 'src_labels')
            self.outputs = tf.placeholder(tf.int32, [None, self.max_seq_len], 'outputs')
            self.output_lens = tf.placeholder(tf.int32, [None,], 'output_lens')

            if self.extractor_type == 'autoencoder':
                # logits
                self.rc_logits, sampled_id = self.autoencoder_extractor(self.texts, self.text_lens)
                self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.outputs, logits=self.rc_logits)
                mask_weight = tf.sequence_mask(self.output_lens, self.max_seq_len, dtype=tf.float32)

                masked_sample_id = sampled_id * tf.cast(mask_weight, dtype=tf.int32)
                masked_outputs = self.outputs * tf.cast(mask_weight, dtype=tf.int32)

                self.correct_pred = tf.equal(masked_sample_id, masked_outputs)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

                # loss and train op
                self.loss = tf.reduce_mean(tf.reduce_sum(self.crossent * mask_weight, 1))
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)

                # summary op
                loss_summary = tf.summary.scalar('autoencode_loss', self.loss)
                accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
                self.summary_op = tf.summary.merge([loss_summary, accuracy_summary])

            elif self.extractor_type == 'classifier':
                # logits and accuracy
                self.csf_logits = self.classifier_extractor(self.texts)
                self.pred = tf.argmax(self.csf_logits, 1)
                self.correct_pred = tf.equal(self.pred, self.labels)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

                # loss and train op
                self.loss = slim.losses.sparse_softmax_cross_entropy(self.csf_logits, self.labels)
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)

                # summary op
                loss_summary = tf.summary.scalar('classifier_loss', self.loss)
                accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
                self.summary_op = tf.summary.merge([loss_summary, accuracy_summary])

        elif self.mode == 'train':
            self.texts = tf.placeholder(tf.int32, [None, self.max_seq_len], 'src_texts')
            self.text_lens = tf.placeholder(tf.int32, [None,], 'src_text_lens')

            self.trg_texts = tf.placeholder(tf.int32, [None, self.max_seq_len], 'trg_texts')
            self.trg_text_lens = tf.placeholder(tf.int32, [None,], 'trg_text_lens')

            self.batch_size = tf.placeholder(tf.int32, name='batch_size')

            # source domain
            if self.extractor_type == 'autoencoder':
                self.fx = self.autoencoder_extractor(self.texts, self.text_lens)
                self.fake_texts, self.fake_texts_logits = self.generator(self.fx)
                self.logits = self.discriminator(self.fake_texts_logits)
                self.fgfx = self.autoencoder_extractor(self.fake_texts, self.text_lens, reuse=True)
            elif self.extractor_type == 'classifier':
                self.fx = self.classifier_extractor(self.texts, self.text_lens)
                self.fake_texts, self.fake_texts_logits = self.generator(self.fx)
                self.logits = self.discriminator(self.fake_texts_logits)
                self.fgfx = self.classifier_extractor(self.fake_texts, reuse=True)


            # loss
            self.d_loss_src = slim.losses.sigmoid_cross_entropy(self.logits, tf.zeros_like(self.logits))
            self.g_loss_src = slim.losses.sigmoid_cross_entropy(self.logits, tf.ones_like(self.logits))
            self.f_loss_src = tf.reduce_mean(tf.square(self.fx.c - self.fgfx.c)) * 15.0

            # optimizer
            self.d_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            self.g_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            self.f_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)

            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'discriminator' in var.name]
            g_vars = [var for var in t_vars if 'generator' in var.name]
            f_vars = [var for var in t_vars if 'content_extractor' in var.name]

            # train op
            with tf.variable_scope('source_train_op'):
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
            origin_texts_summary = tf.summary.scalar('src_origin_texts', self.texts)
            sampled_texts_summary = tf.summary.scalar('src_sampled_texts', self.fake_texts)
            self.summary_op_src = tf.summary.merge([d_loss_src_summary, g_loss_src_summary,
                                                    f_loss_src_summary, origin_texts_summary,
                                                    sampled_texts_summary])

            # target domain
            if self.extractor_type == 'autoencoder':
                self.fx = self.autoencoder_extractor(self.trg_texts, self.trg_text_lens, reuse=True)
            elif self.extractor_type == 'classifier':
                self.fx = self.classifier_extractor(self.trg_texts, reuse=True)
            else:
                raise ValueError("Unknown extractor type.")

            self.reconst_texts, self.reconst_texts_logits = self.generator(self.fx, reuse=True)
            self.logits_fake = self.discriminator(self.reconst_texts_logits, reuse=True)
            self.logits_real = self.discriminator(tf.one_hot(self.trg_texts, self.vocab_size), reuse=True)


            # loss
            self.d_loss_fake_trg = slim.losses.sigmoid_cross_entropy(self.logits_fake, tf.zeros_like(self.logits_fake))
            self.d_loss_real_trg = slim.losses.sigmoid_cross_entropy(self.logits_real, tf.ones_like(self.logits_real))
            self.d_loss_trg = self.d_loss_fake_trg + self.d_loss_real_trg

            self.g_loss_fake_trg = slim.losses.sigmoid_cross_entropy(self.logits_fake, tf.ones_like(self.logits_fake))
            self.g_loss_const_trg = tf.reduce_mean(tf.square(tf.one_hot(self.trg_texts, self.vocab_size) - self.reconst_texts_logits)) * self.rho
            self.g_loss_trg = self.g_loss_fake_trg + self.g_loss_const_trg

            # optimizer
            self.d_optimizer_trg = tf.train.AdamOptimizer(self.learning_rate)
            self.g_optimizer_trg = tf.train.AdamOptimizer(self.learning_rate)

            # train op
            with tf.variable_scope('trg_train_op'):
                self.g_train_op_trg = slim.learning.create_train_op(total_loss=self.g_loss_trg,
                                                                    optimizer=self.g_optimizer_trg,
                                                                    variables_to_train=g_vars)
                self.d_train_op_trg = slim.learning.create_train_op(total_loss=self.d_loss_trg,
                                                                    optimizer=self.d_optimizer_trg,
                                                                    variables_to_train=d_vars)


            # summary op
            d_loss_fake_trg_summary = tf.summary.scalar('trg_d_loss_fake', self.d_loss_fake_trg)
            d_loss_real_trg_summary = tf.summary.scalar('trg_d_loss_real', self.d_loss_real_trg)
            d_loss_trg_summary = tf.summary.scalar('trg_d_loss', self.d_loss_trg)
            g_loss_fake_trg_summary = tf.summary.scalar('trg_g_loss_fake', self.g_loss_fake_trg)
            g_loss_const_trg_summary = tf.summary.scalar('trg_g_loss_const', self.g_loss_const_trg)
            g_loss_trg_summary = tf.summary.scalar('trg_g_loss', self.g_loss_trg)

            origin_texts_summary = tf.summary.scalar('trg_origin_texts', self.trg_texts)
            sampled_texts_summary = tf.summary.scalar('trg_reconstructed_texts', self.reconst_texts)
            self.summary_op_src = tf.summary.merge([d_loss_trg_summary, g_loss_trg_summary,
                                                    d_loss_fake_trg_summary, d_loss_real_trg_summary,
                                                    g_loss_fake_trg_summary, g_loss_const_trg_summary,
                                                    origin_texts_summary, sampled_texts_summary])
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

    def _get_onehot(self, texts):
        return np.sign(np.sum(np.eye(self.vocab_size)[texts], 1))
