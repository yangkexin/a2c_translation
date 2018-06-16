import torchtext
def tokenize_zh(sentence):
    word = [word for word in sentence]
    return word

def id2string(ids, id2w_vocab):
    text = [id2w_vocab[id] for id in ids]
    text = "".join(text)
    return text

def load_data(batch_size):

    AH = torchtext.data.Field(tokenize=tokenize_zh)
    CH = torchtext.data.Field(tokenize=tokenize_zh,)

    NMT_tri, NMT_val, NMT_tst = torchtext.datasets.TranslationDataset.splits(
        path='aug_data/',
        train='NMT_tri', validation='NMT_val', test='NMT_tst',
        exts=( '.an','.cn'), fields=(AH,CH)
    )

    AH.build_vocab(NMT_tri, max_size=15000)
    CH.build_vocab(NMT_tri, max_size=15000)

    NMT_tri_iter, NMT_tst_iter, NMT_val_iter = torchtext.data.BucketIterator.splits(
        datasets=(NMT_tri, NMT_tst, NMT_val), batch_sizes=(batch_size, 200, 200), repeat=False,device=0,
        sort_key=lambda x: torchtext.data.interleave_keys(len(x.src), len(x.trg))
    )

    return NMT_tri_iter,NMT_tst_iter,NMT_val_iter,AH,CH

if __name__ == "__main__":
    train_iter, test_iter,val_iter, an, cn = \
        load_data(128)
    print("[TRAIN]:%d (dataset:%d) [TEST]:%d (dataset:%d)"
          % (len(train_iter), len(train_iter.dataset),
             len(test_iter), len(test_iter.dataset)))
    sent = []

    for step, batch in enumerate(val_iter):
        encoder_input = batch.src
        decoder_input = batch.trg
        encoder_input = [id2string(sent,an.vocab.itos) for sent in encoder_input.permute(1,0)]
        decoder_input = [id2string(sent,cn.vocab.itos) for sent in decoder_input.permute(1,0)]
        print(encoder_input[:2])
        print(decoder_input[:2])
        print("-"*40)


