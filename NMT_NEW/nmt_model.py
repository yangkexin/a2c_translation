import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.init as weigth_init
from torch.autograd import Variable
import torch.nn.functional as F
import re
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from nmt_data_loader import load_data
from tensorboardX import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def propernouns_token(sent):
    #专有名词处理
    name_vocab = datareading("name_new.txt")
    place_vocab = datareading("place_new.txt")
    pos = {}
    line = []
    i = 0
    #find position
    for word in place_vocab:
        place_pos = [m.start() for m in re.finditer(word, sent)]
        if len(place_pos) !=0:
            for i in place_pos:
                pos[i] = i + len(word) - 1
    for word in name_vocab:
        name_pos = [m.start() for m in re.finditer(word, sent)]
        if len(name_pos) !=0:
            for i in name_pos:
                pos[i] = i + len(word) - 1
    #token
    while i <len(sent):
        if i in pos.keys():
            end = pos[i]
            line.append("".join(sent[i:end+1]))
            i = end+1
        else:
            line.append(sent[i])
            i +=1
    return line

def datareading(filepath):
    text = []
    file = open(filepath, encoding='utf-8')
    for line in file.readlines():
        text.append(line.strip())
    return text

def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=5783287, type=int, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=True, help='use gpu')
    parser.add_argument('--mode', choices=['train', 'test', 'sample', 'prob', 'interactive'],
                        default='train', help='run mode')
    parser.add_argument('--reverse', type=bool, default=False, help='reverse the input sentence')
    parser.add_argument('--residual', type=bool, default=False, help='encoder using residual')
    parser.add_argument('--epoch', default=100, type=int, help='epoch nums')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--beam_size', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--sample_size', default=10, type=int, help='sample size')
    parser.add_argument('--embedding_size', default=512, type=int, help='size of word embeddings')
    parser.add_argument('--bidirectional', default=True, type=bool,help='is bidirectional')
    parser.add_argument('--hidden_size', default=512, type=int, help='size of LSTM hidden states')
    parser.add_argument('--layer_size', default=1,type=int, help='size of layers')
    parser.add_argument('--drop_out', default=0.3, type=float, help='dropout rate')
    parser.add_argument('--rnn_cell', default='LSTM',type=str, help='LSTM cell')
    parser.add_argument('--decode_max_time_step', default=100, type=int,
                        help='maximum number of time steps used in decoding and sampling')

    parser.add_argument('--log_every', default=1000, type=int, help='every n iterations to log training statistics')
    parser.add_argument('--valid_niter', default=1, type=int, help='every n epoch to perform validation')

    parser.add_argument(
        '--load_model', default="my_train_model_cn-an/model_512_unreverse_unresidual_new_split_aug.epoch4.bin", type=str, help='load a pre-trained model')
    parser.add_argument(
        '--save_to', default='my_train_model_cn-an/model_512_unreverse_unresidual_new_split_aug2', type=str, help='save trained model to')
    parser.add_argument('--save_model_after', default=1, type=int,
                        help='save the model only after n validation iterations')
    parser.add_argument('--patience', default=5, type=int, help='training patience')
    parser.add_argument('--save_to_file', default=None, type=str, help='if provided, save decoding results to file')
    parser.add_argument('--save_nbest', default=False, action='store_true', help='save nbest decoding results')


    parser.add_argument('--uniform_init', default=None, type=float,
                        help='if specified, use uniform initialization for all parameters')
    parser.add_argument('--grad_clip', default=5., type=float, help='clip gradients')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=0.5, type=float,
                        help='decay learning rate if the validation performance drops')

    # TODO: greedy sampling is still buggy!
    parser.add_argument('--sample_method', default='random', choices=['random', 'greedy'])
    args = parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed * 13 // 7)

    return args

def id2string(ids, id2w_vocab):
    text = [id2w_vocab[id] for id in ids]
    text = "".join(text)
    return text

def savefiletxt(text,filepath):
    f = open(filepath,"w",encoding='utf-8')
    for i in text:
        f.write(str(i))
        f.write('\n')
    f.close()

def tensor_transform(linear, X):
    return linear(X.contiguous().view(-1, X.size(2))).view(X.size(0), X.size(1), -1)

def dot_prod_attention(h_t, src_encoding, src_encoding_att_linear):
    """
    :param h_t: (1,batch_size, hidden_size)
    :param src_encoding: (batch_size, src_sent_len, hidden_size*2)
    :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
    """
    # att_weight(batch_size, src_sent_len,1)
    # 如果batch1是形为b×n×m的张量，batch2是形为b×m×p的张量，则out形状都是(b×n×p)
    att_weight = torch.bmm(src_encoding_att_linear, h_t.permute(1,2,0))
    att_weight = F.softmax(att_weight,dim=1)
    # att_view (batch_size,1,src_sent_len)
    att_view = (att_weight.size(0), 1, att_weight.size(1))
    # (batch_size, hidden_size*2)
    ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)
    return ctx_vec

class Encoder(nn.Module):
    def __init__(self,args):
        """
        # LSTM输入: input, (h_0, c_0) input (seq_len, batch, input_size)
        # h_ (num_layers * num_directions, batch, hidden_size)  c_ (num_layers * num_directions, batch, hidden_size)
        # LSTM输出 output, (h_n, c_n) output (seq_len, batch, hidden_size * num_directions)
"""
        super(Encoder,self).__init__()
        self.args = args
        self.input_drop = nn.Dropout(p=self.args.drop_out)
        self.encoder_embedding = nn.Embedding(
            args.an_vocab_size, args.embedding_size,
            padding_idx=args.an_vocab.stoi['<pad>'])
        self.rnn = nn.LSTM(input_size=args.embedding_size,
                           hidden_size=args.hidden_size,
                           num_layers=1,
                           bias=True,
                           dropout=args.drop_out,
                           bidirectional=args.bidirectional
                           )
        self.layernormal = nn.LayerNorm(args.hidden_size * 2)
        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(args.hidden_size * 2, args.hidden_size)

        #init
        initrange = 0.1
        self.encoder_embedding.weight.data.uniform_(
            -initrange,initrange)
        for weight in self.rnn.parameters():
            if len(weight.size()) > 1:
                weigth_init.orthogonal_(weight.data)

    def forward(self,src_sents):
        src_embedding = self.encoder_embedding(src_sents)
        src_embedding = self.input_drop(src_embedding)
        enc_output,(last_state,last_cell) = self.rnn(src_embedding)
        # enc_output = self.layernormal(enc_output)

        if self.args.bidirectional == True and self.args.layer_size ==1:
            dec_init_cell = self.decoder_cell_init(torch.cat([last_cell[0], last_cell[1]], 1))
            dec_init_cell = dec_init_cell.unsqueeze(0)
            dec_init_state = F.tanh(dec_init_cell)
        else:
            dec_init_cell = self.decoder_cell_init(last_cell)
            dec_init_state = F.tanh(dec_init_cell)

        if self.args.residual:
            residual_var = torch.cat([src_embedding,src_embedding],2)
            enc_output = enc_output + residual_var

        return enc_output,(dec_init_state, dec_init_cell)

class Decoder(nn.Module):
    def __init__(self,args):
        super(Decoder,self).__init__()
        self.args = args
        self.rnn = nn.LSTM(input_size=args.embedding_size + args.hidden_size,
                           hidden_size=args.hidden_size,
                           num_layers=args.layer_size,
                           bias=True,
                           dropout=0.,
                           )
        self.dropout = nn.Dropout(args.drop_out)
        self.readout = nn.Linear(args.hidden_size, args.cn_vocab_size, bias=False)
        #init
        for weight in self.rnn.parameters():
            if len(weight.size()) > 1:
                weigth_init.orthogonal_(weight.data)

    def forward(self,src_encoding, dec_init_vec, tgt_sents,decoder_embedding,att_src_linear,att_vec_linear):

        init_state = dec_init_vec[0]
        init_cell = dec_init_vec[1]

        hidden = (init_state,init_cell)

        #for attention
        #[batch_size,src_len,hidden_size*2]
        src_encoding = src_encoding.permute(1, 0, 2)
        # [batch_size,src_len,hidden_size]
        src_encoding_att_linear = tensor_transform(
            att_src_linear, src_encoding)
        att_tm1 = Variable(
            init_cell.data.new(
                init_cell.data.size(0),init_cell.data.size(1),
                init_cell.data.size(2)).zero_(), requires_grad=False)
        tgt_embedding = decoder_embedding(tgt_sents)
        scores = []

        for one_line_embedding in tgt_embedding.split(split_size = 1):
            #one_line_embedding[1,batch_size,tgt_embedding_size]
            #att_tm1.size[1,batch_size,hidden_size]
            input_line = torch.cat(
                [one_line_embedding,att_tm1],2)

            _,(h_t, cell_t) = self.rnn(input_line, hidden)
            # h_t[layer_size,batch_size,hidden_size]
            h_t = self.dropout(h_t)
            #context_t[batch_size,hidden_size*2]
            context_t = dot_prod_attention(
                h_t, src_encoding, src_encoding_att_linear)
            att_t = F.tanh(
                att_vec_linear(torch.cat([h_t.squeeze(0), context_t], 1)))
            #att_t[batch_size,hidden_size]
            att_t = self.dropout(att_t)
            #score_t[batch_size,vocab_size]
            score_t = self.readout(att_t)
            scores.append(score_t)

            att_tm1 = att_t.unsqueeze(0)
            hidden = (h_t, cell_t)

        scores = torch.stack(scores)
        return  scores

class NMT(nn.Module):
    def __init__(self,args,encoder,decoder):
        super().__init__()
        self.args = args

        self.decoder_embedding = nn.Embedding(
            args.cn_vocab_size, args.embedding_size,
            padding_idx=args.cn_vocab.stoi['<pad>'])

        self.dropout = nn.Dropout(p=self.args.drop_out)

        self.encoder = encoder
        self.decoder = decoder

        self.att_src_linear = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        self.att_vec_linear = nn.Linear(
            args.hidden_size * 2 + args.hidden_size , args.hidden_size, bias=False)

        initrange = 0.1
        self.decoder_embedding.weight.data.uniform_(
            -initrange,initrange)

    def forward(self, src_sents, tgt_sents):
        src_encodings, init_context_vec = self.encoder(src_sents)
        scores = self.decoder(
            src_encodings, init_context_vec, tgt_sents,
            self.decoder_embedding,self.att_src_linear,self.att_vec_linear)
        return scores

    def translate(self, src_sent, beam_size=None, to_word=True):
        """
        perform beam search
        """
        if not beam_size:
            beam_size = self.args.beam_size
        src_sent = src_sent.unsqueeze(1)#[src_len,1]
        # src_encoding[src_len,1,hidden_size*2]
        src_encoding, dec_init_vec = self.encoder(src_sent)
        # [src_len,1,hidden_size]
        src_encoding_att_linear = tensor_transform(
            self.att_src_linear, src_encoding)

        init_state = dec_init_vec[0]#[1,1,hidden_size]
        init_cell = dec_init_vec[1]#[1,1,hidden_size]
        hidden = (init_state, init_cell)
        print(hidden[0].size())
        hyp_scores = torch.tensor(torch.zeros(1)).cuda()

        bos_id = self.args.cn_vocab.stoi['<s>']
        eos_id = self.args.cn_vocab.stoi['</s>']

        tgt_vocab_size = self.args.cn_vocab_size

        hypotheses = [[bos_id]]
        #att_tml[1,1,hidden_size]
        att_tm1 = torch.tensor(
            torch.zeros(1, len(hypotheses),self.args.hidden_size)).cuda()
        completed_hypotheses = []
        completed_hypothesis_scores = []

        time_step = 0
        # while len(completed_hypotheses) < beam_size and time_step < self.args.decode_max_time_step:
        while time_step < self.args.decode_max_time_step:
            time_step += 1
            hyp_num = len(hypotheses)
            #[src_len,hyp_num,hidden_size*2]
            expanded_src_encoding = src_encoding.expand(
                src_encoding.size(0), hyp_num, src_encoding.size(2))
            #[src_len,hyp_num,hidden_size]
            expanded_src_encoding_att_linear = src_encoding_att_linear.expand(
                src_encoding_att_linear.size(0), hyp_num,src_encoding_att_linear.size(2))

            #y_tml[1]取每个beam_list最后一个word
            y_tm1 = torch.LongTensor([hyp[-1] for hyp in hypotheses]).unsqueeze(0).cuda()
            # y_tml[1,hyp_num,embedding_dim]
            y_tm1_embed = self.decoder_embedding(y_tm1)
            # x[1,hyp_num,embedding_dim+hidden_size]
            x = torch.cat([y_tm1_embed, att_tm1], 2)
            # h_t[1,hyp_num, hidden_size]
            _,(h_t, cell_t) = self.decoder.rnn(x, hidden)
            h_t = self.dropout(h_t)
            #ctx_t[1, hidden_size*2]
            ctx_t = dot_prod_attention(
                h_t, expanded_src_encoding.permute(1, 0, 2),
                expanded_src_encoding_att_linear.permute(1, 0, 2))

            att_t = F.tanh(self.att_vec_linear(torch.cat([h_t.squeeze(0), ctx_t], 1)))
            att_t = self.dropout(att_t)
            #score_t[1,vocab_size]
            score_t = self.decoder.readout(att_t)
            p_t = F.log_softmax(score_t,dim=1)

            # live_hyp_num = beam_size - len(completed_hypotheses)
            #
            new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(p_t) + p_t).view(-1)
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=100)
            prev_hyp_ids = top_new_hyp_pos / tgt_vocab_size
            word_ids = top_new_hyp_pos % tgt_vocab_size
            # new_hyp_scores = new_hyp_scores[top_new_hyp_pos.data]

            new_hypotheses = []

            live_hyp_ids = []
            new_hyp_scores = []
            for prev_hyp_id, word_id, new_hyp_score in zip(prev_hyp_ids.cpu().data, word_ids.cpu().data,
                                                           top_new_hyp_scores.cpu().data):
                hyp_tgt_words = hypotheses[prev_hyp_id] + [word_id]
                if len(live_hyp_ids) == beam_size: break
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

            # if len(completed_hypotheses) == beam_size:
            #     break

            live_hyp_ids = torch.LongTensor(live_hyp_ids).cuda()
            hidden = (h_t[:,live_hyp_ids,:],cell_t[:,live_hyp_ids,:])
            att_tm1 = att_t[live_hyp_ids].unsqueeze(0)
            # new_hyp_scores[live_hyp_ids]
            hyp_scores = torch.FloatTensor(new_hyp_scores).cuda()
            hypotheses = new_hypotheses

        if len(completed_hypotheses) == 0:
            completed_hypotheses = [hypotheses[0]]
            completed_hypothesis_scores = [0.0]

        if to_word:
            for i, hyp in enumerate(completed_hypotheses):
                completed_hypotheses[i] = [self.args.cn_vocab.itos[w] for w in hyp]

        ranked_hypotheses = sorted(zip(completed_hypotheses, completed_hypothesis_scores), key=lambda x: x[1],
                                   reverse=True)
        ranked_hypotheses = ranked_hypotheses[:beam_size]
        return [hyp for hyp, score in ranked_hypotheses]

    def save(self, path):
        #只保存模型参数
        print('save parameters to [%s]' % path, file=sys.stderr)
        params = {
            'args': self.args,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

def evaluate_loss(model, val_iter,pad):
    model.eval()
    eval_loss = 0.
    for step, batch in enumerate(val_iter):
        src_sents = batch.src
        tgt_sents = batch.trg

        scores = model(src_sents,tgt_sents[:-1])
        loss = F.cross_entropy(
            scores.view(-1, scores.size(2)), tgt_sents[1:].view(-1),
            ignore_index=pad)
        eval_loss += loss.item()
    return eval_loss/len(val_iter)

def decode_sentence(args,model,val_iter,to_words=True,verbose=True):
    """
    decode the dataset and compute sentence level acc. and  BLEU.
    """
    src_sents_idx = []
    tgt_sents_idx = []
    src_sents_words = []
    tgt_sents_words = []
    hypotheses = []
    for _, batch in enumerate(val_iter):
        src_sents_idx.extend(batch.src.permute(1, 0))
        tgt_sents_idx.extend(batch.trg.permute(1, 0))
    # src_sents_words = [id2string(sent, args.an_vocab.itos) for sent in src_sents_idx]
    # tgt_sents_words = ["".join(id2string(sent, args.cn_vocab.itos)) for sent in tgt_sents_idx]
    print("[begin to translate...]")
    a = re.compile(r'\<.*?\>')
    for i in range(len(src_sents_idx)):
        hyps = model.translate(src_sents_idx[i])
        hypotheses.append(hyps)
        if verbose and to_words and i%1000==0:
            if args.reverse:
                print('*' * 50)
                print('Source: ', a.sub("",''.join(id2string(src_sents_idx[i], args.an_vocab.itos))[::-1]))
                print('Target: ', a.sub("",''.join(id2string(tgt_sents_idx[i], args.cn_vocab.itos))[::-1]))
                # print('Top Hypothesis: ', a.sub("",''.join(max(hypotheses[i],key=len))[::-1]))
                print('Top Hypothesis: ', a.sub("", ''.join(hypotheses[i][0])[::-1]))
            else:
                print('*' * 50)
                print('Source: ', a.sub('',''.join(id2string(src_sents_idx[i], args.an_vocab.itos))))
                print('Target: ', a.sub('',''.join(id2string(tgt_sents_idx[i], args.cn_vocab.itos))))
                print('Top Hypothesis: ',a.sub('',''.join(max(hypotheses[i],key=len))))
                # print('Top Hypothesis: ', a.sub("", ''.join(hypotheses[i][0])))
    top_hypotheses = []
    for i in range(len(hypotheses)):
        top_hypotheses.append(a.sub('',''.join(max(hypotheses[i], key=len))))
        # top_hypotheses.append(a.sub("", ''.join(hypotheses[i][0])))
    return top_hypotheses

def train(args):
    writer = SummaryWriter()
    iter_step = 0
    print('[Loading and preparing training dataset...]')
    train_iter, test_iter,val_iter, an, cn =  load_data(args.batch_size,reverse=args.reverse)
    args.an_vocab = an.vocab
    args.cn_vocab = cn.vocab
    args.an_vocab_size, args.cn_vocab_size = len(an.vocab), len(cn.vocab)
    print('[Dataset load Done...]')
    print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
          % (len(train_iter), len(train_iter.dataset),
             len(test_iter), len(test_iter.dataset)))
    print("[AN_vocab]:%d [CN_vocab]:%d" % (args.an_vocab_size, args.cn_vocab_size))

    print("[Instantiating models...]")
    encoder = Encoder(args)
    decoder = Decoder(args)

    seq2seq = NMT(args,encoder,decoder)
    seq2seq.cuda()
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr=args.lr)

    hist_valid_scores = []
    valid_num = best_model_iter = patience = 0
    for epoch in range(1,args.epoch+1):
        seq2seq.train()
        total_loss = 0.
        report_loss = 0.
        pad = args.cn_vocab.stoi['<pad>']
        begin_time = time.time()
        for step,batch in enumerate(train_iter):
            src_sents = batch.src
            tgt_sents = batch.trg
            optimizer.zero_grad()
            scores = seq2seq(src_sents,tgt_sents[:-1])
            loss = F.cross_entropy(
                scores.view(-1, scores.size(2)), tgt_sents[1:].view(-1),
                ignore_index=pad)
            loss.backward()
            clip_grad_norm_(seq2seq.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.data.item()
            report_loss += loss.data.item()

            writer.add_scalar('train_loss', loss.data.item(), iter_step)
            iter_step += 1
            if step % args.log_every == 0 and step != 0:
                print('epoch:{%d} | batch{%d} | avg_loss{%.3f} | avg_ppl{%.3f} '
                      % (epoch, step ,report_loss / args.log_every,
                          np.exp(report_loss / args.log_every)),file=sys.stderr)
                report_loss = 0.

        print("epoch:{%d} | avg_loss:{%.3f} |  avg_ppl:{%.3f} | time{%.2f sec}"%(
            epoch,total_loss/len(train_iter),np.exp(total_loss/len(train_iter)),
            time.time()-begin_time))

        if epoch % args.valid_niter == 0:
            print('[begin to validate] ...', file=sys.stderr)
            seq2seq.eval()
            valid_num += 1
            eval_loss = evaluate_loss(seq2seq,val_iter,pad)
            eval_ppl = np.exp(eval_loss)
            print("epoch:{%d} | eval_loss:{%.3f} |  eval_ppl:{%.3f}"%(
                epoch,eval_loss,eval_ppl))
        if epoch % 3 == 0:
            val_gener_sents = decode_sentence(
                args,seq2seq,val_iter)
            valid_metric = eval_loss
            # val_gener_sents = [" ".join(sent[0]) for sent in val_gener_sents]
            # val_tgt_sents = [" ".join(sent) for sent in val_gener_sents]
            # bleu(val_tgt_sents,val_gener_sents)
            # valid_metric = corpus_bleu(val_tgt_sents,val_gener_sents)
            # bleu_score, _, _, _, _, _ = bleu.compute_bleu(
            #     val_tgt_sents, val_gener_sents, max_order=4, smooth=True)
            # valid_metric = bleu_score*100
            # print(valid_metric)

        is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
        is_better_than_last = len(hist_valid_scores) == 0 or valid_metric > hist_valid_scores[-1]

        if valid_num >= args.save_model_after:
            model_file = args.save_to + '.epoch%d.bin' % epoch
            print('save model to [%s]' % model_file, file=sys.stderr)
            seq2seq.save(model_file)

        if (not is_better_than_last) and args.lr_decay:
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            print('decay learning rate to %f' % lr, file=sys.stderr)
            optimizer.param_groups[0]['lr'] = lr

        if is_better:
            best_model_iter = epoch

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

def test(args):
    print("begin to test...")
    _, test_iter, _, an, cn = load_data(args.batch_size)
    print('[Dataset load Done...]')
    print("[TEST]:%d (dataset:%d)"
          %(len(test_iter), len(test_iter.dataset)))
    if args.load_model:
        print('load model from [%s]' % args.load_model, file=sys.stderr)
        params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        saved_args = params['args']
        state_dict = params['state_dict']
        encoder = Encoder(saved_args)
        decoder = Decoder(saved_args)
        seq2seq = NMT(saved_args, encoder, decoder)
        seq2seq.load_state_dict(state_dict)
    else:
        encoder = Encoder(args)
        decoder = Decoder(args)
        seq2seq = NMT(args, encoder, decoder)

    seq2seq.eval()
    seq2seq = seq2seq.cuda()
    output_sents_words = decode_sentence(
        saved_args, seq2seq, test_iter, to_words=True, verbose=True)
    savefiletxt(output_sents_words,"output_new_split_aug3.txt")

def interactive(args):
    assert args.load_model, 'You have to specify a pre-trained model'
    print('load model from [%s]' % args.load_model, file=sys.stderr)
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    saved_args = params['args']
    state_dict = params['state_dict']


    encoder = Encoder(saved_args)
    decoder = Decoder(saved_args)
    seq2seq = NMT(saved_args, encoder, decoder)
    seq2seq.load_state_dict(state_dict)

    seq2seq.eval()
    seq2seq = seq2seq.cuda()

    src_sent = "阳光非常强烈，不适宜外出。"
    src_sent = " ".join(propernouns_token(src_sent))
    src_sent = [saved_args.an_vocab.stoi[w] for w in src_sent]
    # src_sent = [saved_args.an_vocab.itos[w] for w in src_sent]
    # print(src_sent)
    src_sent = torch.tensor(src_sent).cuda()
    hyps = seq2seq.translate(src_sent)
    output = ''.join(max(hyps, key=len)).replace(" ","").replace("<s>","").replace("</s>","")
    print(output)

if __name__ == '__main__':
    args = init_config()
    print(args, file=sys.stderr)
    if args.mode == 'train':
        train(args)
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
