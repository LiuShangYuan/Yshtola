import json
import numpy as np
import gzip

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def zip2text_src(zip_file_path='./data/reviews_VideoGames_5.json.gz',
                 txt_file_path='./data/all_src.txt'):
    f = open(txt_file_path, 'w')

    counter = 0

    for d in parse(zip_file_path):

        text_len = len(d['reviewText'].strip().split())

        if text_len < 32 and d['overall'] > 3.0:
            f.write(d['reviewText'].strip())
            f.write('\t')
            f.write('0')
            f.write('\n')

            counter += 1

    f.close()
    print('Saved %d lines into %s' % (counter, txt_file_path))


def split_dataset_src(txt_file_path='./data/all_src.txt',
                      train_json_path='./data/train_src.json',
                      test_json_path='./data/test_src.json'):
    f_train = open(train_json_path, 'w')
    f_test = open(test_json_path, 'w')

    d_list = []

    with open(txt_file_path) as fin:
        for l in fin.readlines():
            try:
                text, _ = l.strip().split('\t')
            except:
                continue
            d_list.append((text, '0'))

    split_len = len(d_list) // 20

    d_train = d_list[:-split_len]
    d_test = d_list[-split_len:]

    print('%d samples in train_set and %d samples in test_set ...' % (len(d_train), len(d_test)))

    json.dump(d_train, f_train)
    json.dump(d_test, f_test)

    f_train.close()
    f_test.close()


def zip2text_trg(zip_file_path='./data/reviews_Books_5.json.gz',
                 txt_file_path='./data/all_trg.txt'):

    f = open(txt_file_path, 'w')

    counter = 0

    for d in parse(zip_file_path):

        text_len = len(d['reviewText'].strip().split())

        if text_len < 32 and (d['overall'] == 1.0):
            f.write(d['reviewText'].strip())
            f.write('\t')
            f.write('1')
            f.write('\n')

            counter += 1

    f.close()
    print('Saved %d lines into %s' % (counter, txt_file_path))


def split_dataset_trg(txt_file_path='./data/all_trg.txt',
                      train_json_path='./data/train_trg.json',
                      test_json_path='./data/test_trg.json'):

    f_train = open(train_json_path, 'w')
    f_test = open(test_json_path, 'w')

    d_list = []

    with open(txt_file_path) as fin:

        for l in fin.readlines():
            try:
                text, label = l.strip().split('\t')
            except:
                continue
            d_list.append((text, '-1'))

    np.random.shuffle(d_list)

    test_len = len(d_list) // 10

    d_train = d_list[:-test_len]
    d_test = d_list[-test_len:]

    print('%d samples in train_set and %d samples in test_set' % (len(d_train), len(d_test)))

    json.dump(d_train, f_train)
    json.dump(d_test, f_test)

    f_train.close()
    f_test.close()



if __name__ == '__main__':
    zip2text_src()
    split_dataset_src()
    zip2text_trg()
    split_dataset_trg()
