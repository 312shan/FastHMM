from FastHMM.hmm import HMMModel
import random

if __name__ == "__main__":
    import timeit

    with open('../data/199801.txt', 'r') as f:
        data = [l.split()[1:] for l in f if l.strip() != '']
        data = [[tuple(pair.split('/')) for pair in line] for line in data]
    # TODO: try to training with BMES tagging scheme
    random.shuffle(data)
    L = len(data)
    test_size = 1000
    train_data, test_data = data[:-test_size], data[-test_size:]
    print('train size {} ,test_size {}'.format(len(train_data), len(test_data)))


    def test(hmm_model):
        for d in train_data:
            hmm_model.train_one_line(d)
        print('finish training')
        corret_cnt = 0
        total_cnt = 0

        start = timeit.default_timer()
        for d in test_data:
            words = []
            pos = []
            for w, tag in d:
                words.append(w)
                pos.append(tag)
            pred = hmm_model.predict(words)
            corret_cnt += sum([word_tag[1] == pos[ind] for ind, word_tag in enumerate(pred)])
            total_cnt += len(pos)
        print('eval result: ')
        print('predict {} tags, {} correct,  accuracy {}'.format(total_cnt, corret_cnt, corret_cnt / total_cnt))
        stop = timeit.default_timer()
        print('runtime : {} seconds'.format(stop - start))


    # test model performance
    hmm_model = HMMModel()
    test(hmm_model)

    # test persist and load model
    hmm_model.save_model()
    hmm_model = HMMModel().load_model()

    res = hmm_model.predict(['我', '是', '中国', '深圳', '打工', '的', '程序猿'])
    print(res)
