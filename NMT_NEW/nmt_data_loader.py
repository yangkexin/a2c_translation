import torchtext
import pickle
def tokenize_zh(sentence):
    word = [word for word in sentence]
    return word

def id2string(ids, id2w_vocab):
    text = [id2w_vocab[id] for id in ids]
    text = "".join(text)
    return text

def load_data(batch_size,reverse=False):
    AH = torchtext.data.Field(tokenize=tokenize_zh)
    CH = torchtext.data.Field(tokenize=tokenize_zh,init_token='<s>', eos_token='</s>')
    if reverse:
        print("[data have being reversed...]")
        NMT_tri, NMT_val, NMT_tst = torchtext.datasets.TranslationDataset.splits(
            path='augdata_new/',
            train='NMT_val_new_split_aug_reverse', validation='NMT_val_new_split_aug_reverse', test='NMT_val_new_split_aug_reverse',
            exts=( '.an','.cn'), fields=(AH,CH))
    else:
        NMT_tri, NMT_val, NMT_tst = torchtext.datasets.TranslationDataset.splits(
            path='augdata_new/',
            train='NMT_tri_new_split_aug1', validation='NMT_tst_new_split_aug1', test='NMT_tst_new_split_aug1',
            exts=( '.an','.cn'), fields=(AH,CH))

    AH.build_vocab(NMT_tri, max_size=15000)
    CH.build_vocab(NMT_tri, max_size=15000)

    NMT_tri_iter, NMT_val_iter = torchtext.data.BucketIterator.splits(
        datasets=(NMT_tri,NMT_val), batch_sizes=(batch_size, 64), repeat=False,device=0,
        sort_key=lambda x: torchtext.data.interleave_keys(len(x.src), len(x.trg))
    )

    NMT_tst_iter = torchtext.data.BucketIterator(
        dataset= NMT_tst, batch_size=64, shuffle=False,repeat=False,device=0
    )
    return NMT_tri_iter,NMT_tst_iter,NMT_val_iter,AH,CH

# def load_data(batch_size,reverse=False):
#     AH = torchtext.data.Field(tokenize=tokenize_zh)
#     CH = torchtext.data.Field(tokenize=tokenize_zh,init_token='<s>', eos_token='</s>')
#     if reverse:
#         print("[data have being reversed...]")
#         NMT_tri, NMT_val, NMT_tst = torchtext.datasets.TranslationDataset.splits(
#             path='augdata_new/',
#             train='NMT_tri_new_split_aug_reverse', validation='NMT_val_new_split_aug_reverse', test='NMT_tst_new_split_aug_reverse',
#             exts=( '.cn','.an'), fields=(AH,CH))
#     else:
#         NMT_tri, NMT_val, NMT_tst = torchtext.datasets.TranslationDataset.splits(
#             path='augdata_new/',
#             train='NMT_tri_new_split_aug', validation='NMT_tst_new_split_aug80000', test='NMT_tst_new_split_aug80000',
#             exts=( '.cn','.an'), fields=(AH,CH))
#
#     AH.build_vocab(NMT_tri, max_size=15000)
#     CH.build_vocab(NMT_tri, max_size=15000)
#
#     NMT_tri_iter, NMT_val_iter = torchtext.data.BucketIterator.splits(
#         datasets=(NMT_tri,NMT_val), batch_sizes=(batch_size, 64), repeat=False,device=0,
#         sort_key=lambda x: torchtext.data.interleave_keys(len(x.src), len(x.trg))
#     )
#
#     NMT_tst_iter = torchtext.data.BucketIterator(
#         dataset= NMT_tst, batch_size=64, shuffle=False,repeat=False,device=0
#     )
#     return NMT_tri_iter,NMT_tst_iter,NMT_val_iter,AH,CH
if __name__ == "__main__":
    train_iter,test_iter,val_iter, an, cn = \
        load_data(128,reverse=False)
    print("[TRAIN]:%d (dataset:%d) [TEST]:%d (dataset:%d)"
          % (len(train_iter), len(train_iter.dataset),
             len(test_iter), len(test_iter.dataset)))
    print("[AN_vocab]:%d [CN_vocab]:%d" % (len(an.vocab), len(cn.vocab)))
    # sent = []
    # for step, batch in enumerate(test_iter):
    #     encoder_input = batch.src
    #     decoder_input = batch.trg
    #     encoder_input = [id2string(sent,an.vocab.itos) for sent in encoder_input.permute(1,0)]
    #     decoder_input = [id2string(sent,cn.vocab.itos) for sent in decoder_input.permute(1,0)]
    #     print(encoder_input.permute(1,0)[:2])
    #     print(decoder_input.permute(1,0)[:2])
    #     print("-"*40)


