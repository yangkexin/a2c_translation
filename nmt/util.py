from collections import defaultdict
import numpy as np

def read_corpus(file_path, source,reverse=False):
    data = []
    for line in open(file_path,"r",encoding="utf-8"):
        if reverse:
            sent = line.strip()
            sent = sent[::-1].split(' ')
        else:
            sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)
    if reverse: print("data have being reversed...")
    print("success read")
    return data


def batch_slice(data, batch_size, sort=True):
    batched_data = []
    #ceil [siːl] 向正无穷取整 朝正无穷大方向取整
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        src_sents = [data[i * batch_size + b][0] for b in range(cur_batch_size)]
        tgt_sents = [data[i * batch_size + b][1] for b in range(cur_batch_size)]

        if sort:#根据句子长度降序排序
            src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(src_sents[src_id]), reverse=True)
            src_sents = [src_sents[src_id] for src_id in src_ids]
            tgt_sents = [tgt_sents[src_id] for src_id in src_ids]

        batched_data.append((src_sents, tgt_sents))

    return batched_data


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of source sentences in each batch is decreasing
    """
    #collections.defaultdict会返回一个类似dictionary的对象
    #defaultdict类的初始化函数接受一个类型作为参数，当所访问的键不存在的时候，可以实例化一个值作为默认值
    buckets = defaultdict(list)
    #这里相当于把相同长度的句子放在一个位置，key为长度，value为句子
    for pair in data:
        src_sent = pair[0]
        buckets[len(src_sent)].append(pair)

    batched_data = []
    for src_len in buckets:
        tuples = buckets[src_len]
        if shuffle: np.random.shuffle(tuples)
        batched_data.extend(batch_slice(tuples, batch_size))

    if shuffle:
        np.random.shuffle(batched_data)

    for batch in batched_data:
        yield batch

