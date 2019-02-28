import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import os
import scipy.io
import scipy.misc


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

    def load_src_texts(self):
        pass

    def load_trg_texts(self):
        pass

    def pretrain(self):
        pass

    def train(self):
        pass