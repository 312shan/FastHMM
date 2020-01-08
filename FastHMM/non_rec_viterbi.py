#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from collections import deque
from functools import reduce

from typing import List, Union, Set, Dict, DefaultDict, Tuple


class Viterbi(object):
    def __init__(
            self,
            A: Union[Dict[str, Dict[str, float]], DefaultDict[str, Dict[str, float]]],
            B: Union[Dict[str, Dict[str, float]], DefaultDict[str, Dict[str, float]]],
            PI: Union[Dict[str, float], DefaultDict[str, float]],
            STATE: Union[Set[str], List[str]],
            very_small_probability: float = 1e-32
    ):
        self._A = A
        self._B = B
        self._PI = PI
        self._STATE = set(STATE)
        # TODO: find out what is the best value and why?
        self._MINI_FOR_ZERO = math.log(very_small_probability)

    def predict_state(self, word_list):
        # type: (List[str]) -> Tuple[List[str], float]
        return self._viterbi(word_list)

    def p_aij(self, i, j):
        if not self._A.get(i):
            return self._MINI_FOR_ZERO
        return self._A[i].get(j, self._MINI_FOR_ZERO)

    def p_bik(self, i, k):
        if not self._B.get(i):
            return self._MINI_FOR_ZERO
        return self._B[i].get(k, self._MINI_FOR_ZERO)

    def p_pi(self, i):
        return self._PI.get(i, self._MINI_FOR_ZERO)

    def _viterbi(self, obs_init):
        # type: (List[str]) -> Tuple[List[str], float]
        """
        Viterbi decode algorithm
        Uses queues to store calculated candidate sequences and their probabilities
        :param obs_init: Observation sequence
        :return: Hidden state sequences and probability scores
        """
        q = deque()
        q.append((obs_init, {}, []))
        while q:
            obs, val_pre, qseq_pre = q.popleft()
            if len(obs) == 0:
                val_temp = [(qseq_pre[q_] + [q_], val_pre[q_]) for q_ in self._STATE]
                max_q_seq = reduce(lambda x1, x2: x2 if x2[1] > x1[1] else x1, val_temp)
                seq, val = max_q_seq
                return seq, val
            val = {}
            qseq = {}
            for cur_state in self._STATE:
                if len(val_pre) == 0:
                    val.update({cur_state: self.p_pi(cur_state) + self.p_bik(cur_state, obs[0])})
                    qseq.update({cur_state: []})
                else:
                    # transition probability of (pre_tag->cur_tag) * Output probability of (cur_tag->obs[0])
                    val_temp = [(qseq_pre[q_pre] + [q_pre],
                                 val_pre[q_pre] + self.p_aij(q_pre, cur_state) + self.p_bik(cur_state, obs[0]))
                                for q_pre in self._STATE]
                    # gain tuple with max probability
                    max_q_seq = reduce(lambda x1, x2: x2 if x2[1] > x1[1] else x1, val_temp)
                    val.update({cur_state: max_q_seq[1]})
                    qseq.update({cur_state: max_q_seq[0]})
            q.append((obs[1:], val, qseq))


if __name__ == "__main__":
    STATE = ['A', 'B', 'C']
    PI = {'A': .8}
    A = {'A': {'A': 0.1, 'B': .7, 'C': .2, }, 'B': {'A': .1, 'B': 0.1, 'C': .8}, 'C': {'A': .1, 'B': 0.1, 'C': .8}}
    B = {'A': {'你': 0.5, '我': 0.5},
         'B': {'是': 0.4, '打': 0.6},
         'C': {'人': 0.5, '中国人': 0.5}}

    viterbi = Viterbi(A, B, PI, STATE)
    trace = viterbi.predict_state(["我", "打", "中国人"])
    print(trace)
    state_sequence = viterbi.predict_state(["我", "是", "中国人"])
    print(state_sequence)
