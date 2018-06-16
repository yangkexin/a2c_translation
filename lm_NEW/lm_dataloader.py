from torchtext import data

def id2string(ids, id2w_vocab):
    text = [id2w_vocab[id] for id in ids]
    text = "".join(text)
    return text

def load_data(batch_size):
    an = data.Field(
        sequential=True,include_lengths=False, init_token='<s>', eos_token='</s>')

    train, val, test = data.TabularDataset.splits(
        path='data/a2c/an', train='LM_tri_an.csv', validation='LM_val_an.csv', test='LM_tst_an.csv', format='csv',
        fields=[ ('an', an)])


    an.build_vocab(train, max_size=5000)

    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test), sort_key=lambda x: len(x.an),
        batch_sizes=(batch_size, 256, 256), device=0,
        sort_within_batch=True,shuffle=True,repeat = False)
    return train_iter, val_iter, test_iter, an

if __name__ == "__main__":
    train_iter, val_iter, test_iter, an = \
        load_data(600)
    print("[TRAIN]:%d (dataset:%d) [TEST]:%d (dataset:%d)"
          % (len(train_iter), len(train_iter.dataset),
             len(test_iter), len(test_iter.dataset)))
    for step, batch in enumerate(val_iter):
        train_data = batch.an
        input_data = train_data[:-1, :].permute(1,0)
        target_data = train_data[1:, :].permute(1,0)
        input_data = [id2string(sent,an.vocab.itos) for sent in input_data]
        target_data = [id2string(sent,an.vocab.itos) for sent in target_data]
        print(input_data[:2])
        print(target_data[:2])
        print("-"*40)


