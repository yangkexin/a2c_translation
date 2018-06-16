from __future__ import print_function

import os
import sys
import time
import argparse
from itertools import tee

import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
#上面提到的函数的功能是将一个填充后的变长序列压紧。 
#这个操作和pack_padded_sequence()是相反的。把压紧的序列再填充回来。
from nltk.translate.bleu_score import corpus_bleu

from util import read_corpus, data_iter
from vocab import Vocab, VocabEntry


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=5783287, type=int, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=True, help='use gpu')
    parser.add_argument('--mode', choices=['train', 'raml_train', 'test', 'sample', 'prob', 'interactive'],
                        default='train', help='run mode')
    parser.add_argument('--reverse', type=bool,default=False,help='reverse the input sentence')
    parser.add_argument('--residual', type=bool, default=False, help='encoder using residual')
    parser.add_argument('--vocab', type=str, help='path of the serialized vocabulary')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--beam_size', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--sample_size', default=10, type=int, help='sample size')
    parser.add_argument('--embed_size', default=256, type=int, help='size of word embeddings')
    parser.add_argument('--hidden_size', default=256, type=int, help='size of LSTM hidden states')
    parser.add_argument('--dropout', default=0., type=float, help='dropout rate')

    parser.add_argument('--train_src', default='', type=str, help='path to the training source file')
    parser.add_argument('--train_tgt', default='', type=str, help='path to the training target file')
    parser.add_argument('--dev_src', default='', type=str, help='path to the dev source file')
    parser.add_argument('--dev_tgt', default='',type=str, help='path to the dev target file')
    parser.add_argument('--test_src', default='',type=str, help='path to the test source file')
    parser.add_argument('--test_tgt', default='',type=str, help='path to the test target file')

    parser.add_argument('--decode_max_time_step', default=100, type=int, help='maximum number of time steps used '
                                                                              'in decoding and sampling')

    parser.add_argument('--valid_niter', default=500, type=int, help='every n iterations to perform validation')
    parser.add_argument('--valid_metric', default='bleu', choices=['bleu', 'ppl', 'word_acc', 'sent_acc'], help='metric used for validation')
    parser.add_argument('--log_every', default=50, type=int, help='every n iterations to log training statistics')
    parser.add_argument('--load_model', default=None, type=str, help='load a pre-trained model')
    parser.add_argument('--save_to', default='model', type=str, help='save trained model to')
    parser.add_argument('--save_model_after', default=2, type=int, help='save the model only after n validation iterations')
    parser.add_argument('--save_to_file', default=None, type=str, help='if provided, save decoding results to file')
    parser.add_argument('--save_nbest', default=False, action='store_true', help='save nbest decoding results')
    parser.add_argument('--patience', default=5, type=int, help='training patience')
    parser.add_argument('--uniform_init', default=None, type=float, help='if specified, use uniform initialization for all parameters')
    parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')
    parser.add_argument('--max_niter', default=-1, type=int, help='maximum number of training iterations')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=0.5, type=float, help='decay learning rate if the validation performance drops')

    # raml training
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--temp', default=0.85, type=float, help='temperature in reward distribution')
    parser.add_argument('--raml_sample_mode', default='pre_sample',
                        choices=['pre_sample', 'hamming_distance', 'hamming_distance_impt_sample'],
                        help='sample mode when using RAML')
    parser.add_argument('--raml_sample_file', type=str, help='path to the sampled targets')
    parser.add_argument('--raml_bias_groundtruth', action='store_true', default=False, help='make sure ground truth y* is in samples')

    parser.add_argument('--smooth_bleu', action='store_true', default=False,
                        help='smooth sentence level BLEU score.')

    #TODO: greedy sampling is still buggy!
    parser.add_argument('--sample_method', default='random', choices=['random', 'greedy'])

    args = parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed * 13 // 7)

    return args


def input_transpose(sents, pad_token):
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    masks = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])
        masks.append([1 if len(sents[k]) > i else 0 for k in range(batch_size)])

    return sents_t, masks


def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]


def tensor_transform(linear, X):
    # X is a 3D tensor
    #把Xshape变为适应linear的shape计算完成后再恢复到原来的shape
    #the size -1 is inferred from other dimensions
    return linear(X.contiguous().view(-1, X.size(2))).view(X.size(0), X.size(1), -1)

class NMT(nn.Module):
    def __init__(self, args, vocab):
        super(NMT, self).__init__()

        self.args = args

        self.vocab = vocab
        #考虑加预训练向量
        #输入： LongTensor (N, W), N = mini-batch, W = 每个mini-batch中提取的下标数
        #输出： (N, W, embedding_dim)
        self.src_embed = nn.Embedding(len(vocab.src), args.embed_size, padding_idx=vocab.src['<pad>'])
        self.tgt_embed = nn.Embedding(len(vocab.tgt), args.embed_size, padding_idx=vocab.tgt['<pad>'])
        #LSTM输入: input, (h_0, c_0) input (seq_len, batch, input_size)
        #h_ (num_layers * num_directions, batch, hidden_size)  c_ (num_layers * num_directions, batch, hidden_size)
        #LSTM输出 output, (h_n, c_n) output (seq_len, batch, hidden_size * num_directions)
          
        self.encoder_lstm = nn.LSTM(args.embed_size, args.hidden_size, bidirectional=True, dropout=args.dropout)
        self.decoder_lstm = nn.LSTMCell(args.embed_size + args.hidden_size, args.hidden_size)

        # attention: dot product attention
        # project source encoding to decoder rnn's h space
        self.att_src_linear = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)输入size存疑
        self.att_vec_linear = nn.Linear(args.hidden_size * 2 + args.hidden_size, args.hidden_size, bias=False)

        # prediction layer of the target vocabulary
        self.readout = nn.Linear(args.hidden_size, len(vocab.tgt), bias=False)

        # dropout layer
        self.dropout = nn.Dropout(args.dropout)

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(args.hidden_size * 2, args.hidden_size)

    def forward(self, src_sents, src_sents_len, tgt_words):
        src_encodings, init_ctx_vec = self.encode(src_sents, src_sents_len)
        scores = self.decode(src_encodings, init_ctx_vec, tgt_words)

        return scores

    def encode(self, src_sents, src_sents_len):
        """
        :param src_sents: (src_sent_len, batch_size), sorted by the length of the source
        :param src_sents_len: (src_sent_len)
        """
        # (src_sent_len, batch_size, embed_size)
        src_word_embed = self.src_embed(src_sents)
        #这里的pack，理解成压紧比较好。 将一个填充过的变长序列 压紧。（填充时候，会有冗余，所以压紧一下）
        packed_src_embed = pack_padded_sequence(src_word_embed, src_sents_len)
        
        # output: (src_sent_len, batch_size, hidden_size)
        output, (last_state, last_cell) = self.encoder_lstm(packed_src_embed)
        #这个操作和pack_padded_sequence()是相反的。把压紧的序列再填充回来。
        output, _ = pad_packed_sequence(output)

        dec_init_cell = self.decoder_cell_init(torch.cat([last_cell[0], last_cell[1]], 1))
        dec_init_state = F.tanh(dec_init_cell)

        if args.residual:
            residual_input = torch.cat([src_word_embed,src_word_embed],2)
            output = output + residual_input


        return output, (dec_init_state, dec_init_cell)

    def decode(self, src_encoding, dec_init_vec, tgt_sents):
        """
        :param src_encoding: (src_sent_len, batch_size, hidden_size)
        :param dec_init_vec: (batch_size, hidden_size)
        :param tgt_sents: (tgt_sent_len, batch_size)
        :return:
        """
        init_state = dec_init_vec[0]
        init_cell = dec_init_vec[1]
        hidden = (init_state, init_cell)

        new_tensor = init_cell.data.new
        batch_size = src_encoding.size(1)

        # (batch_size, src_sent_len, hidden_size * 2)
        src_encoding = src_encoding.permute(1, 0, 2)#交换0维和1维
        # (batch_size, src_sent_len, hidden_size)
        src_encoding_att_linear = tensor_transform(self.att_src_linear, src_encoding)
        # initialize attentional vector
        #shape [bacth_size,hidden_size]
        att_tm1 = Variable(new_tensor(batch_size, self.args.hidden_size).zero_(), requires_grad=False)
         #tgt_word_embed (tgt_sent_len,batch_size,embedding_dim)
        tgt_word_embed = self.tgt_embed(tgt_sents)
        scores = []

        # start from `<s>`, until y_{T-1} 将输入张量分割成相等形状的chunks 默认延dim=0分割
        #这里就是分为一句一句 y_tml_embed shape (1,batch_size, embedding_dim)
        for y_tm1_embed in tgt_word_embed.split(split_size=1):
            #squeeze将输入张量形状0维中的1去除并返回。再把attention vector拼上
            #att_tm1 shape [bacth_size,hidden_size]
            #shape 变为[bacth_size, embedding_dim+hidden_size]
            x = torch.cat([y_tm1_embed.squeeze(0), att_tm1], 1)

            # h_t: (batch_size, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)
            h_t = self.dropout(h_t)
            #求attention的context vector 和att_weight
            #ctx_t shape[batch_size,hidden_size]
            ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encoding, src_encoding_att_linear)
            #att_vec_linear input_shape[batch_size,hidden_size*2]
            att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))   # E.q. (5)
            att_t = self.dropout(att_t)
            #prediction layer of the target vocabulary
            score_t = self.readout(att_t)   # E.q. (6)
            scores.append(score_t)

            att_tm1 = att_t
            hidden = h_t, cell_t
        #沿着一个新维度对输入张量序列进行连接，dim默认0
        scores = torch.stack(scores)
        return scores

    def translate(self, src_sents, beam_size = None, to_word=True):
        if not type(src_sents[0]) == list:
            src_sents = [src_sents]
        if not beam_size:
            beam_size = args.beam_size
        src_sents_var = to_input_variable(src_sents, self.vocab.src, cuda=args.cuda, is_test=True)

        src_encoding, dec_init_vec = self.encode(src_sents_var, [len(src_sents[0])])
        src_encoding_att_linear = tensor_transform(self.att_src_linear, src_encoding)

        init_state = dec_init_vec[0]
        init_cell = dec_init_vec[1]
        hidden = (init_state, init_cell)

        att_tm1 = Variable(torch.zeros(1, self.args.hidden_size), volatile=True)
        hyp_scores = Variable(torch.zeros(1), volatile=True)
        if args.cuda:
            att_tm1 = att_tm1.cuda()
            hyp_scores = hyp_scores.cuda()

        eos_id = self.vocab.tgt['</s>']
        bos_id = self.vocab.tgt['<s>']
        tgt_vocab_size = len(self.vocab.tgt)

        hypotheses = [[bos_id]]
        completed_hypotheses = []
        completed_hypothesis_scores = []

        t = 0
        # while len(completed_hypotheses) < beam_size and t < 100:
        while t < 100:
            t += 1
            hyp_num = len(hypotheses)

            expanded_src_encoding = src_encoding.expand(src_encoding.size(0), hyp_num, src_encoding.size(2))
            expanded_src_encoding_att_linear = src_encoding_att_linear.expand(src_encoding_att_linear.size(0), hyp_num, src_encoding_att_linear.size(2))

            y_tm1 = Variable(torch.LongTensor([hyp[-1] for hyp in hypotheses]), volatile=True)
            if args.cuda:
                y_tm1 = y_tm1.cuda()

            y_tm1_embed = self.tgt_embed(y_tm1)

            x = torch.cat([y_tm1_embed, att_tm1], 1)

            # h_t: (hyp_num, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)
            h_t = self.dropout(h_t)

            ctx_t, alpha_t = self.dot_prod_attention(h_t, expanded_src_encoding.permute(1, 0, 2), expanded_src_encoding_att_linear.permute(1, 0, 2))

            att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))
            att_t = self.dropout(att_t)

            score_t = self.readout(att_t)
            p_t = F.log_softmax(score_t)

            # live_hyp_num = beam_size - len(completed_hypotheses)
            live_hyp_num = beam_size
            new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(p_t) + p_t).view(-1)
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=100)
            prev_hyp_ids = top_new_hyp_pos / tgt_vocab_size
            word_ids = top_new_hyp_pos % tgt_vocab_size
            # new_hyp_scores = new_hyp_scores[top_new_hyp_pos.data]

            new_hypotheses = []

            live_hyp_ids = []
            new_hyp_scores = []
            for prev_hyp_id, word_id, new_hyp_score in zip(prev_hyp_ids.cpu().data, word_ids.cpu().data, top_new_hyp_scores.cpu().data):
                hyp_tgt_words = hypotheses[prev_hyp_id] + [word_id]
                if len(live_hyp_ids)==beam_size:break
                if word_id == eos_id:
                    completed_hypotheses.append(hyp_tgt_words)
                    # length penalty
                    pl_var = (5 + (len(hyp_tgt_words)-2))/((5 + 1))
                    new_hyp_score = new_hyp_score / pl_var
                    completed_hypothesis_scores.append(new_hyp_score)
                else:
                    new_hypotheses.append(hyp_tgt_words)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(new_hyp_score)

            # if len(completed_hypotheses) == 100:
            #     break

            live_hyp_ids = torch.LongTensor(live_hyp_ids)
            if args.cuda:
                live_hyp_ids = live_hyp_ids.cuda()

            hidden = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hyp_scores = Variable(torch.FloatTensor(new_hyp_scores), volatile=True) # new_hyp_scores[live_hyp_ids]
            if args.cuda:
                hyp_scores = hyp_scores.cuda()
            hypotheses = new_hypotheses

        if len(completed_hypotheses) == 0:
            completed_hypotheses = [hypotheses[0]]
            completed_hypothesis_scores = [0.0]

        if to_word:
            for i, hyp in enumerate(completed_hypotheses):
                completed_hypotheses[i] = [self.vocab.tgt.id2word[w] for w in hyp]

        # print("候选集数量：",len(completed_hypothesis_scores))
        ranked_hypotheses = sorted(zip(completed_hypotheses, completed_hypothesis_scores), key=lambda x: x[1], reverse=True)

        ranked_hypotheses = ranked_hypotheses[:10]
        # for i in range(len(ranked_hypotheses)):
        #     sent = " ".join(ranked_hypotheses[i][0])
        #     print(sent,ranked_hypotheses[i][1],len(sent))
        return [hyp for hyp, score in ranked_hypotheses]

    def sample(self, src_sents, sample_size=None, to_word=False):
        if not type(src_sents[0]) == list:
            src_sents = [src_sents]
        if not sample_size:
            sample_size = args.sample_size

        src_sents_num = len(src_sents)
        batch_size = src_sents_num * sample_size

        src_sents_var = to_input_variable(src_sents, self.vocab.src, cuda=args.cuda, is_test=True)
        src_encoding, (dec_init_state, dec_init_cell) = self.encode(src_sents_var, [len(s) for s in src_sents])

        dec_init_state = dec_init_state.repeat(sample_size, 1)
        dec_init_cell = dec_init_cell.repeat(sample_size, 1)
        hidden = (dec_init_state, dec_init_cell)

        # tile everything
        # if args.sample_method == 'expand':
        #     # src_enc: (src_sent_len, sample_size, enc_size)
        #     # cat result: (src_sent_len, batch_size * sample_size, enc_size)
        #     src_encoding = torch.cat([src_enc.expand(src_enc.size(0), sample_size, src_enc.size(2)) for src_enc in src_encoding.split(1, dim=1)], 1)
        #     dec_init_state = torch.cat([x.expand(sample_size, x.size(1)) for x in dec_init_state.split(1, dim=0)], 0)
        #     dec_init_cell = torch.cat([x.expand(sample_size, x.size(1)) for x in dec_init_cell.split(1, dim=0)], 0)
        # elif args.sample_method == 'repeat':

        src_encoding = src_encoding.repeat(1, sample_size, 1)
        src_encoding_att_linear = tensor_transform(self.att_src_linear, src_encoding)
        src_encoding = src_encoding.permute(1, 0, 2)
        src_encoding_att_linear = src_encoding_att_linear.permute(1, 0, 2)

        new_tensor = dec_init_state.data.new
        att_tm1 = Variable(new_tensor(batch_size, self.args.hidden_size).zero_(), volatile=True)
        y_0 = Variable(torch.LongTensor([self.vocab.tgt['<s>'] for _ in range(batch_size)]), volatile=True)

        eos = self.vocab.tgt['</s>']
        # eos_batch = torch.LongTensor([eos] * batch_size)
        sample_ends = torch.ByteTensor([0] * batch_size)
        all_ones = torch.ByteTensor([1] * batch_size)
        if args.cuda:
            y_0 = y_0.cuda()
            sample_ends = sample_ends.cuda()
            all_ones = all_ones.cuda()

        samples = [y_0]

        t = 0
        while t < 100:
            t += 1

            # (sample_size)
            y_tm1 = samples[-1]

            y_tm1_embed = self.tgt_embed(y_tm1)

            x = torch.cat([y_tm1_embed, att_tm1], 1)

            # h_t: (batch_size, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)
            h_t = self.dropout(h_t)
            #求attention的context vector 和 att_weight
            ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encoding, src_encoding_att_linear)

            att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
            att_t = self.dropout(att_t)

            score_t = self.readout(att_t)  # E.q. (6)
            p_t = F.softmax(score_t)

            if args.sample_method == 'random':
                y_t = torch.multinomial(p_t, num_samples=1).squeeze(1)
            elif args.sample_method == 'greedy':
                _, y_t = torch.topk(p_t, k=1, dim=1)
                y_t = y_t.squeeze(1)

            samples.append(y_t)

            sample_ends |= torch.eq(y_t, eos).byte().data
            if torch.equal(sample_ends, all_ones):
                break

            # if torch.equal(y_t.data, eos_batch):
            #     break

            att_tm1 = att_t
            hidden = h_t, cell_t

        # post-processing
        completed_samples = [list([list() for _ in range(sample_size)]) for _ in range(src_sents_num)]
        for y_t in samples:
            for i, sampled_word in enumerate(y_t.cpu().data):
                src_sent_id = i % src_sents_num
                sample_id = i / src_sents_num
                if len(completed_samples[src_sent_id][sample_id]) == 0 or completed_samples[src_sent_id][sample_id][-1] != eos:
                    completed_samples[src_sent_id][sample_id].append(sampled_word)

        if to_word:
            for i, src_sent_samples in enumerate(completed_samples):
                completed_samples[i] = word2id(src_sent_samples, self.vocab.tgt.id2word)

        return completed_samples

    def attention(self, h_t, src_encoding, src_linear_for_att):
        # (1, batch_size, attention_size) + (src_sent_len, batch_size, attention_size) =>
        # (src_sent_len, batch_size, attention_size)
        att_hidden = F.tanh(self.att_h_linear(h_t).unsqueeze(0).expand_as(src_linear_for_att) + src_linear_for_att)

        # (batch_size, src_sent_len)
        att_weights = F.softmax(tensor_transform(self.att_vec_linear, att_hidden).squeeze(2).permute(1, 0))

        # (batch_size, hidden_size * 2)
        ctx_vec = torch.bmm(src_encoding.permute(1, 2, 0), att_weights.unsqueeze(2)).squeeze(2)

        return ctx_vec, att_weights

    def dot_prod_attention(self, h_t, src_encoding, src_encoding_att_linear, mask=None):
        """
        :param h_t: (batch_size, hidden_size)
        :param src_encoding: (batch_size, src_sent_len, hidden_size)
        :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
        :param mask: (batch_size, src_sent_len)
        """
        # att_weight(batch_size, src_sent_len)
        #对两个批batch1和batch2内存储的矩阵进行批矩阵乘操作
        # 如果batch1是形为b×n×m的张量，batch1是形为b×m×p的张量，则out形状都是n×p
        att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
        if mask:
            #在mask值为1的位置处用value填充。mask的元素个数需和本tensor相同，但尺寸可以不同
            att_weight.data.masked_fill_(mask, -float('inf'))
        att_weight = F.softmax(att_weight)
        #att_view (batch_size,1,src_sent_len)
        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

        return ctx_vec, att_weight

    def save(self, path):
        print('save parameters to [%s]' % path, file=sys.stderr)
        params = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)


def to_input_variable(sents, vocab, cuda=False, is_test=False):
    """
    return a tensor of shape (src_sent_len, batch_size)
    """

    word_ids = word2id(sents, vocab)
    sents_t, masks = input_transpose(word_ids, vocab['<pad>'])

    sents_var = Variable(torch.LongTensor(sents_t), volatile=is_test, requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()

    return sents_var


def evaluate_loss(model, data, crit):
    print('[INFO] evaluating loss')
    model.eval()
    cum_loss = 0.
    cum_tgt_words = 0.

    for src_sents, tgt_sents in data_iter(data, batch_size=args.batch_size, shuffle=False):
        pred_tgt_word_num = sum(len(s[1:]) for s in tgt_sents) # omitting leading `<s>`
        src_sents_len = [len(s) for s in src_sents]

        src_sents_var = to_input_variable(src_sents, model.vocab.src, cuda=args.cuda, is_test=True)
        tgt_sents_var = to_input_variable(tgt_sents, model.vocab.tgt, cuda=args.cuda, is_test=True)

        # (tgt_sent_len, batch_size, tgt_vocab_size)
        scores = model(src_sents_var, src_sents_len, tgt_sents_var[:-1])
        loss = crit(scores.view(-1, scores.size(2)), tgt_sents_var[1:].view(-1))

        cum_loss += loss.data[0]
        cum_tgt_words += pred_tgt_word_num

    cum_tgt_words = 1. if cum_tgt_words < 1. else cum_tgt_words
    loss = cum_loss / cum_tgt_words
    return loss


def init_training(args):
#从磁盘文件中读取一个通过torch.save()保存的对象。 
#torch.load() 可通过参数map_location 动态地进行内存重映射，使其能从不动设备中读取文件。
    if args.load_model:
        print('load model from [%s]' % args.load_model, file=sys.stderr)
        ## Load all tensors onto the CPU
        params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        opt = params['args']
        state_dict = params['state_dict']
        model = NMT(opt, vocab)
        model.load_state_dict(state_dict)
        model.train()
    else:
        vocab = torch.load(args.vocab)
        model = NMT(args, vocab)
        model.train()

        if args.uniform_init:
            print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init), file=sys.stderr)
            for p in model.parameters():
                p.data.uniform_(-args.uniform_init, args.uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0
    #负的log likelihood loss损失。用于训练一个n类分类器。
    #weight参数应该是一个1-Dtensor，里面的值对应类别的权重。
    #size_average=False那么将会把mini-batch中所有样本的loss累加起来
    nll_loss = nn.NLLLoss(weight=vocab_mask, size_average=False)
    cross_entropy_loss = nn.CrossEntropyLoss(weight=vocab_mask, size_average=False)

    if args.cuda:
        model = model.cuda()
        nll_loss = nll_loss.cuda()
        cross_entropy_loss = cross_entropy_loss.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    return vocab, model, optimizer, nll_loss, cross_entropy_loss


def train(args):
    #按空格分词并给tgt_source加<s></s>
    train_data_src = read_corpus(args.train_src, source='src',reverse = args.reverse)
    train_data_tgt = read_corpus(args.train_tgt, source='tgt',reverse = args.reverse)

    dev_data_src = read_corpus(args.dev_src, source='src',reverse = args.reverse)
    dev_data_tgt = read_corpus(args.dev_tgt, source='tgt',reverse = args.reverse)

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    vocab, model, optimizer, nll_loss, cross_entropy_loss = init_training(args)

    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = cum_batches = report_examples = epoch = valid_num = best_model_iter = 0

    if args.load_model:
        import re
        train_iter = int(re.search('(?<=iter)\d+', args.load_model).group(0))
        print('start from train_iter = %d' % train_iter)

        valid_num = train_iter // args.valid_niter

    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1
        print('start of epoch {:d}'.format(epoch))

        for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size):
            train_iter += 1
            #Word2id and padding size(_sent_len, batch_size, _vocab_size)
            src_sents_var = to_input_variable(src_sents, vocab.src, cuda=args.cuda)
            tgt_sents_var = to_input_variable(tgt_sents, vocab.tgt, cuda=args.cuda)

            batch_size = len(src_sents)
            src_sents_len = [len(s) for s in src_sents]
            pred_tgt_word_num = sum(len(s[1:]) for s in tgt_sents) # omitting leading `<s>`

            optimizer.zero_grad()

            # score shape(tgt_sent_len, batch_size, tgt_vocab_size)
            scores = model(src_sents_var, src_sents_len, tgt_sents_var[:-1])
            #
            word_loss = cross_entropy_loss(scores.view(-1, scores.size(2)), tgt_sents_var[1:].view(-1))
            loss = word_loss / batch_size
            word_loss_val = word_loss.data[0]
            loss_val = loss.data[0]

            loss.backward()
            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
            #所有的optimizer都实现了step()方法，这个方法会更新所有的参数。
            #一旦梯度被如backward()之类的函数计算好后，我们就可以调用这个函数。
            optimizer.step()
            
            #总的loss
            report_loss += word_loss_val
            cum_loss += word_loss_val
            #总共用的taget words 个数
            report_tgt_words += pred_tgt_word_num
            cum_tgt_words += pred_tgt_word_num
            #
            report_examples += batch_size
            cum_examples += batch_size
            cum_batches += batch_size

            if train_iter % args.log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (
                          epoch, train_iter,report_loss / report_examples, np.exp(report_loss / report_tgt_words),cum_examples,
                          report_tgt_words / (time.time() - train_time),time.time() - begin_time), file=sys.stderr)
                #sys.stderr 是用来重定向标准错误信息
                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % args.valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (
                    epoch, train_iter, cum_loss / cum_batches, np.exp(cum_loss / cum_tgt_words),cum_examples), file=sys.stderr)

                cum_loss = cum_batches = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)
                model.eval()

                # compute dev. ppl and bleu

                dev_loss = evaluate_loss(model, dev_data, cross_entropy_loss)
                dev_ppl = np.exp(dev_loss)

                if args.valid_metric in ['bleu', 'word_acc', 'sent_acc']:
                    dev_hyps = decode(model, dev_data,args)
                    dev_hyps = [hyps[0] for hyps in dev_hyps]
                    if args.valid_metric == 'bleu':
                        valid_metric = get_bleu([tgt for src, tgt in dev_data], dev_hyps)
                    else:
                        valid_metric = get_acc([tgt for src, tgt in dev_data], dev_hyps, acc_type=args.valid_metric)
                    print('validation: iter %d, dev. ppl %f, dev. %s %f' % (train_iter, dev_ppl, args.valid_metric, valid_metric),
                          file=sys.stderr)
                else:
                    valid_metric = -dev_ppl
                    print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl),
                          file=sys.stderr)

                model.train()

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                is_better_than_last = len(hist_valid_scores) == 0 or valid_metric > hist_valid_scores[-1]
                hist_valid_scores.append(valid_metric)

                if valid_num > args.save_model_after:
                    model_file = args.save_to + '.iter%d.bin' % train_iter
                    print('save model to [%s]' % model_file, file=sys.stderr)
                    model.save(model_file)

                if (not is_better_than_last) and args.lr_decay:
                    lr = optimizer.param_groups[0]['lr'] * args.lr_decay
                    print('decay learning rate to %f' % lr, file=sys.stderr)
                    optimizer.param_groups[0]['lr'] = lr

                if is_better:
                    patience = 0
                    best_model_iter = train_iter

                    if valid_num > args.save_model_after:
                        print('save currently the best model ..', file=sys.stderr)
                        model_file_abs_path = os.path.abspath(model_file)
                        symlin_file_abs_path = os.path.abspath(args.save_to + '.bin')
                        os.system('ln -sf %s %s' % (model_file_abs_path, symlin_file_abs_path))
                else:
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)
                    if patience == args.patience:
                        print('early stop!', file=sys.stderr)
                        print('the best model is from iteration [%d]' % best_model_iter, file=sys.stderr)
                        sys.exit


def get_bleu(references, hypotheses):
    # compute BLEU
    bleu_score = corpus_bleu([[ref[1:-1]] for ref in references],
                             [hyp[1:-1] for hyp in hypotheses])

    return bleu_score


def get_acc(references, hypotheses, acc_type='word'):
    assert acc_type == 'word_acc' or acc_type == 'sent_acc'
    cum_acc = 0.

    for ref, hyp in zip(references, hypotheses):
        ref = ref[1:-1]
        hyp = hyp[1:-1]
        if acc_type == 'word_acc':
            acc = len([1 for ref_w, hyp_w in zip(ref, hyp) if ref_w == hyp_w]) / float(len(hyp) + 1e-6)
        else:
            acc = 1. if all(ref_w == hyp_w for ref_w, hyp_w in zip(ref, hyp)) else 0.
        cum_acc += acc

    acc = cum_acc / len(hypotheses)
    return acc


def decode(model, data, args,verbose=True):
    """
    decode the dataset and compute sentence level acc. and BLEU.
    """
    hypotheses = []
    begin_time = time.time()

    data = list(data)
    if type(data[0]) is tuple:
        for src_sent, tgt_sent in data:
            hyps = model.translate(src_sent)
            hypotheses.append(hyps)

            if verbose:
                if args.reverse:
                    print('*' * 50)
                    print('Source: ', ' '.join(src_sent)[::-1])
                    print('Target: ', ' '.join(tgt_sent)[::-1])
                    print('Top Hypothesis: ', ' '.join(max(hyps,key=len))[::-1])
                else:
                    print('*' * 50)
                    print('Source: ', ' '.join(src_sent))
                    print('Target: ', ' '.join(tgt_sent),len(' '.join(tgt_sent)))
                    print('Top Hypothesis: ', ' '.join(max(hyps,key=len)),len(' '.join(max(hyps,key=len))))
    else:
        for src_sent in data:
            hyps = model.translate(src_sent)
            hypotheses.append(hyps)

            if verbose:
                if args.reverse:
                    print('*' * 50)
                    print('Source: ', ' '.join(src_sent)[::-1])
                    print('Top Hypothesis: ', ' '.join(max(hyps,key=len))[::-1])
                else:
                    print('*' * 50)
                    print('Target: ', ' '.join(tgt_sent),len(' '.join(tgt_sent)))
                    print('Top Hypothesis: ', ' '.join(max(hyps,key=len)),len(' '.join(max(hyps,key=len))))

    elapsed = time.time() - begin_time

    print('decoded %d examples, took %d s' % (len(data), elapsed), file=sys.stderr)

    return hypotheses


def compute_lm_prob(args):
    """
    given source-target sentence pairs, compute ppl and log-likelihood
    """
    test_data_src = read_corpus(args.test_src, source='src')
    test_data_tgt = read_corpus(args.test_tgt, source='tgt')
    test_data = zip(test_data_src, test_data_tgt)

    if args.load_model:
        print('load model from [%s]' % args.load_model, file=sys.stderr)
        params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        saved_args = params['args']
        state_dict = params['state_dict']

        model = NMT(saved_args, vocab)
        model.load_state_dict(state_dict)
    else:
        vocab = torch.load(args.vocab)
        model = NMT(args, vocab)

    model.eval()

    if args.cuda:
        model = model.cuda()

    f = open(args.save_to_file, 'w')
    for src_sent, tgt_sent in test_data:
        src_sents = [src_sent]
        tgt_sents = [tgt_sent]

        batch_size = len(src_sents)
        src_sents_len = [len(s) for s in src_sents]
        pred_tgt_word_nums = [len(s[1:]) for s in tgt_sents]  # omitting leading `<s>`

        # (sent_len, batch_size)
        src_sents_var = to_input_variable(src_sents, model.vocab.src, cuda=args.cuda, is_test=True)
        tgt_sents_var = to_input_variable(tgt_sents, model.vocab.tgt, cuda=args.cuda, is_test=True)

        # (tgt_sent_len, batch_size, tgt_vocab_size)
        scores = model(src_sents_var, src_sents_len, tgt_sents_var[:-1])
        # (tgt_sent_len * batch_size, tgt_vocab_size)
        log_scores = F.log_softmax(scores.view(-1, scores.size(2)))
        # remove leading <s> in tgt sent, which is not used as the target
        # (batch_size * tgt_sent_len)
        flattened_tgt_sents = tgt_sents_var[1:].view(-1)
        # (batch_size * tgt_sent_len)
        tgt_log_scores = torch.gather(log_scores, 1, flattened_tgt_sents.unsqueeze(1)).squeeze(1)
        # 0-index is the <pad> symbol
        tgt_log_scores = tgt_log_scores * (1. - torch.eq(flattened_tgt_sents, 0).float())
        # (tgt_sent_len, batch_size)
        tgt_log_scores = tgt_log_scores.view(-1, batch_size) # .permute(1, 0)
        # (batch_size)
        tgt_sent_scores = tgt_log_scores.sum(dim=0).squeeze()
        tgt_sent_word_scores = [tgt_sent_scores[i].data[0] / pred_tgt_word_nums[i] for i in range(batch_size)]

        for src_sent, tgt_sent, score in zip(src_sents, tgt_sents, tgt_sent_word_scores):
            f.write('%s ||| %s ||| %f\n' % (' '.join(src_sent), ' '.join(tgt_sent), score))

    f.close()


def test(args):
    test_data_src = read_corpus(args.test_src, source='src',reverse = args.reverse)
    test_data_tgt = read_corpus(args.test_tgt, source='tgt',reverse = args.reverse)
    test_data = zip(test_data_src, test_data_tgt)

    if args.load_model:
        print('load model from [%s]' % args.load_model, file=sys.stderr)
        params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        saved_args = params['args']
        state_dict = params['state_dict']

        model = NMT(saved_args, vocab)
        model.load_state_dict(state_dict)
    else:
        vocab = torch.load(args.vocab)
        model = NMT(args, vocab)

    model.eval()

    if args.cuda:
        model = model.cuda()

    hypotheses = decode(model, test_data, args,verbose=True)
    top_hypotheses = []
    for hyps in hypotheses:
        output_sent = max(hyps,key=len)
        top_hypotheses.append(output_sent)
    
#     top_hypotheses = [hyps[0] for hyps in hypotheses]

    # bleu_score = get_bleu([tgt for src, tgt in test_data], top_hypotheses)
    # word_acc = get_acc([tgt for src, tgt in test_data], top_hypotheses, 'word_acc')
    # sent_acc = get_acc([tgt for src, tgt in test_data], top_hypotheses, 'sent_acc')
    # print('Corpus Level BLEU: %f, word level acc: %f, sentence level acc: %f' % (bleu_score, word_acc, sent_acc), file=sys.stderr)

    if args.save_to_file:
        print('save decoding results to %s' % args.save_to_file, file=sys.stderr)
        with open(args.save_to_file, 'w',encoding="utf8") as f:
            for hyps in hypotheses:
                f.write(' '.join(hyps[0][1:-1]) + '\n')

        if args.save_nbest:
            nbest_file = args.save_to_file + '.nbest'
            print('save nbest decoding results to %s' % nbest_file, file=sys.stderr)
            with open(nbest_file, 'w') as f:
                for src_sent, tgt_sent, hyps in zip(test_data_src, test_data_tgt, hypotheses):
                    print('Source: %s' % ' '.join(src_sent), file=f)
                    print('Target: %s' % ' '.join(tgt_sent), file=f)
                    print('Hypotheses:', file=f)
                    for i, hyp in enumerate(hyps, 1):
                        print('[%d] %s' % (i, ' '.join(hyp)), file=f)
                    print('*' * 30, file=f)


def interactive(args):
    assert args.load_model, 'You have to specify a pre-trained model'
    print('load model from [%s]' % args.load_model, file=sys.stderr)
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    vocab = params['vocab']
    saved_args = params['args']
    state_dict = params['state_dict']

    model = NMT(saved_args, vocab)
    model.load_state_dict(state_dict)

    model.eval()

    if args.cuda:
        model = model.cuda()

    while True:
        src_sent = raw_input('Source Sentence:')
        src_sent = src_sent.strip().split(' ')
        hyps = model.translate(src_sent)
        for i, hyp in enumerate(hyps, 1):
            print('Hypothesis #%d: %s' % (i, ' '.join(hyp)))


def sample(args):
    train_data_src = read_corpus(args.train_src, source='src',reverse = args.reverse)
    train_data_tgt = read_corpus(args.train_tgt, source='tgt',reverse = args.reverse)
    train_data = zip(train_data_src, train_data_tgt)

    if args.load_model:
        print('load model from [%s]' % args.load_model, file=sys.stderr)
        params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        opt = params['args']
        state_dict = params['state_dict']

        model = NMT(opt, vocab)
        model.load_state_dict(state_dict)
    else:
        vocab = torch.load(args.vocab)
        model = NMT(args, vocab)

    model.eval()

    if args.cuda:
        model = model.cuda()

    print('begin sampling')

    check_every = 10
    train_iter = cum_samples = 0
    train_time = time.time()
    for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size):
        train_iter += 1
        samples = model.sample(src_sents, sample_size=args.sample_size, to_word=True)
        cum_samples += sum(len(sample) for sample in samples)

        if train_iter % check_every == 0:
            elapsed = time.time() - train_time
            print('sampling speed: %d/s' % (cum_samples / elapsed), file=sys.stderr)
            cum_samples = 0
            train_time = time.time()

        for i, tgt_sent in enumerate(tgt_sents):
            print('*' * 80)
            print('target:' + ' '.join(tgt_sent))
            tgt_samples = samples[i]
            print('samples:')
            for sid, sample in enumerate(tgt_samples, 1):
                print('[%d] %s' % (sid, ' '.join(sample[1:-1])))
            print('*' * 80)


if __name__ == '__main__':
    args = init_config()
    print(args, file=sys.stderr)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'raml_train':
        train_raml(args)
    elif args.mode == 'sample':
        sample(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'prob':
        compute_lm_prob(args)
    elif args.mode == 'interactive':
        interactive(args)
    else:
        raise RuntimeError('unknown mode')
