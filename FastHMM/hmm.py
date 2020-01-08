#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import unicode_literals

import math
import os
import pathlib
import pickle
from collections import defaultdict
from typing import List, Tuple, Union, Set, Dict, DefaultDict

from FastHMM.non_rec_viterbi import Viterbi


class HMMModel(object):
    # tag padding for start and end for representation consistency
    START_STATE: str = '<start>'
    END_STATE: str = '<end>'

    def __init__(
            self,
            A: Union[Dict[str, Dict[str, float]], DefaultDict[str, Dict[str, float]]] = None,
            B: Union[Dict[str, Dict[str, float]], DefaultDict[str, Dict[str, float]]] = None,
            PI: Union[Dict[str, float], DefaultDict[str, float]] = None,
            STATE: Union[Set[str]] = None,
    ):
        if not A:
            self.A = defaultdict(defaultdict)
            self.B = defaultdict(defaultdict)
            self.PI = {}
            self.STATE = set()
            # tmp variable only on training process
            self.state_count = defaultdict(int)  # count of each state
            self.state_bigram = defaultdict(defaultdict)  # count of (State_{t} | State_{t-1})
            self.state_observation_pair = defaultdict(defaultdict)  # count of pair state and emission observation
        else:
            self.A = A
            self.B = B
            self.PI = PI
            self.STATE = STATE

    def train_one_line(self, list_of_word_tag_pair):
        # type: (List[(Union[List[str, str], Tuple[str, str]])]) -> None
        """
        train model from one line data
        :param list_of_word_tag_pair: list of tuple (word, tag)
        :return: None
        """
        previous_tag = self.START_STATE
        for word, tag in list_of_word_tag_pair:
            self.STATE.add(tag)
            self._state_bigram_increase_one(previous_tag, tag)
            self._tag_count_increase_one(previous_tag)
            previous_tag = tag
            self._state_observation_pair_increase_one(tag, word)
        self._state_bigram_increase_one(previous_tag, self.END_STATE)
        self._tag_count_increase_one(previous_tag)

    def _state_bigram_increase_one(self, previous_tag, tag):
        tag_state_bigram = self.state_bigram[previous_tag]
        bigram = (previous_tag, tag)
        tag_state_bigram[bigram] = tag_state_bigram.get(bigram, 0) + 1

    def _tag_count_increase_one(self, tag):
        self.state_count[tag] += 1

    def _state_observation_pair_increase_one(self, tag, word):
        self.state_observation_pair[tag][word] = self.state_observation_pair[tag].get(word, 0) + 1

    def do_train(self):
        for previous_state, previous_state_count in self.state_count.items():
            # compute transition probability

            # NOTE: using dict.get() to prevent no such dict key AKA no such bigram pair
            bigram_local_storage = self.state_bigram.get(previous_state, {})
            for bigram, bigram_count in bigram_local_storage.items():
                bigram_probability = bigram_count / previous_state_count
                state = bigram[1]
                self.A[previous_state][state] = math.log(bigram_probability)

            # compute emission probability
            # NOTE: using dict.get() to prevent start state have on emission will cause exeception
            emission_local_storage = self.state_observation_pair.get(previous_state, {})
            for word, word_count in emission_local_storage.items():
                emmit_probability = word_count / previous_state_count
                self.B[previous_state][word] = math.log(emmit_probability)

        self.PI = {k[1]: v / self.state_count['<start>'] for k, v in
                   self.state_bigram.get('<start>', {}).items()}

    def predict(self, word_list, ):
        # type: (List[str]) -> List[Tuple[str, str]]
        if not self.A:  # using self.A as an training-flag indicate if already trained.
            self.do_train()

        viterbi = Viterbi(self.A, self.B, self.PI, self.STATE)
        state_list, _ = viterbi.predict_state(word_list)
        return [(word, state) for word, state in zip(word_list, state_list)]

    @staticmethod
    def _load_data(input_file):
        with input_file.open('rb') as fd:
            obj = pickle.load(fd)
            return obj

    @staticmethod
    def _save_data(obj, output_file):
        with output_file.open('wb') as fd:
            # using protocol=2 to keep compatible with python 2
            pickle.dump(obj, fd, protocol=2)

    @classmethod
    def load_model(cls, model_dir="model"):
        assert os.path.exists(model_dir)
        model_dir_path = pathlib.Path(model_dir)

        A = cls._load_data(model_dir_path / 'A.pickle')
        B = cls._load_data(model_dir_path / 'B.pickle')
        PI = cls._load_data(model_dir_path / 'PI.pickle')
        STATE = cls._load_data(model_dir_path / 'STATE.pickle')

        return cls(A, B, PI, STATE)

    def save_model(self, model_dir="model"):
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        model_dir_path = pathlib.Path(model_dir)

        self._save_data(self.A, model_dir_path / 'A.pickle')
        self._save_data(self.B, model_dir_path / 'B.pickle')
        self._save_data(self.PI, model_dir_path / 'PI.pickle')
        self._save_data(self.STATE, model_dir_path / 'STATE.pickle')


if __name__ == "__main__":
    import timeit
    import random

    with open('../data/199801.txt', 'r', encoding='gbk') as f:
        data = [l.split()[1:] for l in f if l.strip() != '']
        data = [[tuple(pair.split('/')) for pair in line] for line in data]
    # TODO: try to training with BMES tagging scheme
    random.shuffle(data)
    L = len(data)
    test_size = 100
    train_data, test_data = data[:-test_size], data[-test_size:]
    print('train size {} ,test_size {}'.format(len(train_data), len(test_data)))


    def test(hmm_model: HMMModel):
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
        print('runtime : {}'.format(stop - start))


    hmm_model = HMMModel()
    test(hmm_model)
    hmm_model.save_model()
    hmm_model = HMMModel().load_model()

    res = hmm_model.predict(['我', '是', '中国', '深圳', '打工', '的', '程序猿'])
    print(res)
