import os
import sys
from lm_dataloader import load_data
import argparse
import time
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorboardX import SummaryWriter

def init_config():
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--save_dict', type=str, default='dict_an.pkl',
                        help='location of the save dict')
    parser.add_argument('--embedding_size', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='number of hidden units per layer')
    parser.add_argument('--layer_size', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--lr_decay', default=0.5, type=float,
                        help='decay learning rate if the validation performance drops')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='gradient clipping')
    parser.add_argument('--epoch', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--drop_out', type=float, default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', type=bool,default=False,
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', type=bool,default=True,
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=5000,
                        help='report interval')
    parser.add_argument('--save', type=str,  default='my_model/a2c_an.pt',
                        help='path to save the final model')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    return args

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == torch.Tensor:
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

class RNNModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.drop = nn.Dropout(args.drop_out)
        self.encoder = nn.Embedding(
            args.vocab_size, args.embedding_size,padding_idx=args.vocab.stoi['<pad>'])
        self.rnn = nn.GRU(args.embedding_size, args.hidden_size, args.layer_size, dropout=args.drop_out)
        self.decoder = nn.Linear(args.hidden_size, args.vocab_size)

        if args.tied:
            if args.hidden_size != args.embedding_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input,hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb,hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)),hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return torch.cuda.FloatTensor(
            weight.new(self.args.layer_size, bsz, self.args.hidden_size).zero_())

class LMProb():
    def __init__(self, model_path, dict_path):
        with open(model_path, 'rb') as f:
            self.model = torch.load(f)
            print("model load...")
            self.model.eval()
            self.model = self.model.cuda()

        with open(dict_path, 'rb') as f:
            print("dict load...")
            self.dictionary = pickle.load(f)

    def get_prob(self, words, verbose=False):
        pad_words = ['<sos>'] + words + ['<eos>']
        indxs = [self.dictionary.stoi[w] for w in pad_words]
        input = torch.cuda.LongTensor([int(indxs[0])]).unsqueeze(0)
        if verbose:
            print('words =', pad_words)
            print('indxs =', indxs)

        hidden = self.model.init_hidden(1)
        log_probs = []
        for i in range(1, len(pad_words)):
            output, hidden = self.model(input,hidden)
            word_weights = output.squeeze().data.exp()


            prob = word_weights[indxs[i]] / word_weights.sum()
            log_probs.append(math.log(prob))
            input.data.fill_(int(indxs[i]))

        if verbose:
            for i in range(len(log_probs)):
                print('  {} => {:d},\tlogP(w|s)={:.4f}'.format(pad_words[i+1], indxs[i+1], log_probs[i]))
            print('\n  => sum_prob = {:.4f}'.format(sum(log_probs)))

        return sum(log_probs) / math.sqrt(len(log_probs))

def train(args):
    writer = SummaryWriter()
    iter_step = 0
    train_iter, val_iter, test_iter, an = load_data(args.batch_size)
    print('Dataset load Done...')
    print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
          % (len(train_iter), len(train_iter.dataset),
             len(test_iter), len(test_iter.dataset)))
    args.vocab = an.vocab
    with open(args.save_dict,"wb") as savedict:
        pickle.dump(args.vocab,savedict)
    print("dict saved...")
    args.vocab_size = len(an.vocab)
    model = RNNModel(args).cuda()
    pad = args.vocab.stoi['<pad>']
    best_val_loss = None
    print("begin training...")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    batch_nums = len(train_iter)-1
    for epoch in range(1, args.epoch + 1):
        model.train()
        total_loss = 0.
        report_loss = 0.
        epoch_start_time = time.time()
        hidden = model.init_hidden(args.batch_size)
        for step,batch in enumerate(train_iter):
            train_data = batch.an
            input_data = train_data[:-1,:]
            target_data = train_data[1:,:].view(-1)
            hidden = repackage_hidden(hidden)
            optimizer.zero_grad()
            if step+1 == len(train_iter):
                break
            output,hidden = model(input_data,hidden)
            loss = criterion(
                output.view(-1, args.vocab_size), target_data)
            loss.backward()

            writer.add_scalar('train_loss', loss.data.item(), iter_step)
            iter_step += 1

            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            total_loss += loss.data.item()
            report_loss += loss.data.item()
            if step % args.log_interval == 0 and step > 0:
                print('-' * 89)
                print('| epoch {:2d} | batch {:2d} | train loss {:5.2f} | train ppl {:8.2f}'.format(
                    epoch,step,report_loss / args.log_interval, np.exp(report_loss / args.log_interval)))
                print('-' * 89)
                report_loss = 0.

        val_loss = evaluate(model,args,val_iter,pad)


        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | '
                'train ppl {:8.2f} | val loss {:5.2f} |val ppl {:8.2f}'.format(
                epoch, (time.time() - epoch_start_time),
                total_loss/batch_nums, np.exp(total_loss/batch_nums),
                val_loss , np.exp(val_loss)))
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
                print("model saved...")
            best_val_loss = val_loss
        else:
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            print('decay learning rate to %f' % lr, file=sys.stderr)
            optimizer.param_groups[0]['lr'] = lr

def evaluate(model,args,val_iter,pad):
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(256)
    batch_nums = len(val_iter) - 1
    for step, batch in enumerate(val_iter):
        if step+1 == len(val_iter):
            break
        train_data = batch.an
        input_data = train_data[:-1]
        target_data = train_data[1:].view(-1)
        output,hidden = model(input_data,hidden)
        loss = F.cross_entropy(
            output.view(-1, args.vocab_size), target_data, ignore_index=pad)
        total_loss += loss.data.item()
        hidden = repackage_hidden(hidden)
    return total_loss/ batch_nums



if __name__ =="__main__":
    args = init_config()
    words = ['其', '高', '七', '尺', '，']
    lmprob = LMProb('my_model/a2c_an.pt', 'dict_an.pkl')
    norm_prob = lmprob.get_prob(words, verbose=True)
    print('\n  => norm_prob = {:.4f}'.format(norm_prob))
    # print(args)
    # train(args)


