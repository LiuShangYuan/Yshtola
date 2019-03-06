import json
import numpy as np
import gzip

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def prepro_amazon(maxlen=32):

    data_lm = []
    count_lm = 0

    # Source: VideoGames_Reviews ==> Positive
    with open('./data/train_vg.json','w') as f_train, \
            open('./data/test_vg.json','w') as f_test:
        data_vg = []

        for d in parse('./data/reviews_VideoGames_5.json.gz'):
            text_len = len(d['reviewText'].strip().split())
            if text_len < maxlen and text_len > 1 and d['overall'] > 3.0:
                text = d['reviewText'].strip()
                data_vg.append(text)
                data_lm.append(text)
                count_lm += 1

        split_length = len(data_vg) // 20       # Train_VG : Test_VG == 19:1
        np.random.shuffle(data_vg)

        train_vg = data_vg[:-split_length]
        test_vg = data_vg[-split_length:]

        print('Saved %d SOURCE items.[Train: %d, Test: %d]' % (len(data_vg), len(train_vg), len(test_vg)))

        json.dump(train_vg, f_train, ensure_ascii=False)
        json.dump(test_vg, f_test, ensure_ascii=False)

    # Target: Books_Reviews ==> Negative
    with open('./data/train_b.json', 'w') as f_train, \
        open('./data/test_b.json','w') as f_test:
        data_b = []

        for d in parse('./data/reviews_Books_5.json.gz'):
            text_len = len(d['reviewText'].strip().split())
            if text_len < maxlen and text_len > 1 and d['overall'] == 1.0:
                text = d['reviewText'].strip()
                data_b.append(text)
                data_lm.append(text)
                count_lm += 1

        split_length = len(data_b) // 20        # Train_Book : Test_Book == 19:1
        np.random.shuffle(data_b)

        train_b = data_b[:-split_length]
        test_b = data_b[-split_length:]

        print('Saved %d TARGET items.[Train: %d, Test: %d]' % (len(data_b), len(train_b), len(test_b)))

        json.dump(train_b, f_train, ensure_ascii=False)
        json.dump(test_b, f_test, ensure_ascii=False)

    # Language Model
    with open('./data/train_lm.json','w') as f_train, \
        open('./data/test_lm.json','w') as f_test:

        np.random.shuffle(data_lm)
        split_length = len(data_lm) // 20

        train_lm = data_lm[:-split_length]
        test_lm = data_lm[-split_length:]

        print('Saved %d LM items.[Train: %d, Test: %d]' % (len(data_lm), len(train_lm), len(test_lm)))

        json.dump(train_lm, f_train, ensure_ascii=False)
        json.dump(test_lm, f_test, ensure_ascii=False)







# def zip2text_src(zip_file_path='./data/reviews_VideoGames_5.json.gz',
#                  txt_file_path='./data/all_src.txt'):
#     f = open(txt_file_path, 'w')
#
#     counter = 0
#     for d in parse(zip_file_path):
#         text_len = len(d['reviewText'].strip().split())
#         if text_len < 32 and d['overall'] > 3.0:
#             f.write(d['reviewText'].strip())
#             f.write('\t')
#             f.write('0')
#             f.write('\n')
#
#             counter += 1
#
#     f.close()
#     print('Saved %d lines into %s' % (counter, txt_file_path))
#
#
# def split_dataset_src(txt_file_path='./data/all_src.txt',
#                       train_json_path='./data/train_src.json',
#                       test_json_path='./data/test_src.json'):
#     f_train = open(train_json_path, 'w')
#     f_test = open(test_json_path, 'w')
#
#     d_list = []
#     with open(txt_file_path) as fin:
#         for l in fin.readlines():
#             try:
#                 text, _ = l.strip().split('\t')
#             except:
#                 continue
#             d_list.append((text, '0'))
#
#     split_len = len(d_list) // 20
#
#     d_train = d_list[:-split_len]
#     d_test = d_list[-split_len:]
#
#     print('%d samples in train_set and %d samples in test_set ...' % (len(d_train), len(d_test)))
#
#     json.dump(d_train, f_train, ensure_ascii=False)
#     json.dump(d_test, f_test, ensure_ascii=False)
#
#     f_train.close()
#     f_test.close()


# def zip2text_trg(zip_file_path='./data/reviews_Books_5.json.gz',
#                  txt_file_path='./data/all_trg.txt'):
#
#     f = open(txt_file_path, 'w')
#
#     counter = 0
#     for d in parse(zip_file_path):
#         text_len = len(d['reviewText'].strip().split())
#         if text_len < 32 and (d['overall'] == 1.0):
#             f.write(d['reviewText'].strip())
#             f.write('\t')
#             f.write('1')
#             f.write('\n')
#
#             counter += 1
#
#     f.close()
#     print('Saved %d lines into %s' % (counter, txt_file_path))
#
#
# def split_dataset_trg(txt_file_path='./data/all_trg.txt',
#                       train_json_path='./data/train_trg.json',
#                       test_json_path='./data/test_trg.json'):
#
#     f_train = open(train_json_path, 'w')
#     f_test = open(test_json_path, 'w')
#
#     d_list = []
#     with open(txt_file_path) as fin:
#         for l in fin.readlines():
#             try:
#                 text, label = l.strip().split('\t')
#             except:
#                 continue
#             d_list.append((text, '-1'))
#
#     np.random.shuffle(d_list)
#
#     test_len = len(d_list) // 10
#
#     d_train = d_list[:-test_len]
#     d_test = d_list[-test_len:]
#
#     print('%d samples in train_set and %d samples in test_set' % (len(d_train), len(d_test)))
#
#     json.dump(d_train, f_train, ensure_ascii=False)
#     json.dump(d_test, f_test, ensure_ascii=False)
#
#     f_train.close()
#     f_test.close()



if __name__ == '__main__':
    prepro_amazon()
