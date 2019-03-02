import json
import numpy as np
import gzip

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def zip2text():
    f = open('./data/amazon_all.txt', 'w')

    counter = 0

    for d in parse('./data/reviews_Books_5.json.gz'):

        text_len = len(d['reviewText'].strip().split())

        if text_len < 32 and (d['overall'] == 1.0 or d['overall'] == 5.0):
            if d['overall'] == 1.0:
                f.write(d['reviewText'].strip())
                f.write('\t')
                f.write('0')
                f.write('\n')
            elif d['overall'] == 5.0:
                f.write(d['reviewText'].strip())
                f.write('\t')
                f.write('1')
                f.write('\n')

            counter += 1

    f.close()

    print('Saved %d lines into ./data/amazon_all.txt' % counter)

def split_dataset():

    f_train = open('./data/amazon_train.txt', 'w')
    f_test = open('./data/amazon_test.txt', 'w')

    d_true = []
    d_false = []

    with open('./data/amazon_all.txt') as fin:

        for l in fin.readlines():
            try:
                text, label = l.strip().split('\t')
            except:
                continue
            if label == '0':
                d_false.append((text, '0'))
            else:
                d_true.append((text, '1'))

    false_len = len(d_false)
    true_len = len(d_true)

    print('%d samples in Class:0 and %d samples in Class:1' % (false_len, true_len))

    false_test_len = false_len // 100
    true_test_len = true_len // 100

    d_train = d_true[:-true_test_len] + d_false[:-false_test_len]
    d_test = d_true[-true_test_len:] + d_false[-false_test_len:]

    np.random.shuffle(d_train)
    np.random.shuffle(d_test)

    print('%d samples in train_set and %d samples in test_set' % (len(d_train), len(d_test)))

    for d in d_train:
        f_train.write(d[0])
        f_train.write('\t')
        f_train.write(d[1])
        f_train.write('\n')

    for d in d_test:
        f_test.write(d[0])
        f_test.write('\t')
        f_test.write(d[1])
        f_test.write('\n')


if __name__ == '__main__':
    zip2text()
    split_dataset()
