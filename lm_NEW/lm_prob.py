# -*- coding: utf-8 -*-

import math
import torch
import pickle
from lm_model import *
words = ['其', '高', '七', '尺', '，']
lmprob = LMProb('my_model/a2c_an.pt', 'dict_an.pkl')
norm_prob = lmprob.get_prob(words, verbose=True)
print('\n  => norm_prob = {:.4f}'.format(norm_prob))
