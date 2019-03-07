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

if __name__ == '__main__':
    prepro_amazon()
