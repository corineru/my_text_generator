import numpy as np
from collections import Counter
import copy
import pickle


# 生成batch，一共有num_batch个batch，一次批量处理num_seqs行，一行num_steps个字
# 这样可以保证num_steps个字的连续行以及batch间的连续性
def batch_generator(arr, num_seqs, num_steps):
    arr = copy.copy(arr)
    size = num_steps * num_seqs
    num_batch = len(arr)//size
    arr = arr[:num_batch*size]
    arr = arr.reshape((num_seqs, -1))
    while(True):
        np.random.shuffle(arr)
        for n in range(0, arr.shape[1], num_steps):
            x = arr[:, n:n+num_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y

# 将文字转化成对应数字的类
class TextConverter:
    def __init__(self, text, max_vocab=5000, filename=None):
        if filename is not None:
            with open(filename, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            words = set(text)
            count = Counter(words)
            count_paris = count.most_common(max_vocab)
            self.vocab, _ = list(zip(*count_paris))

        self.word_to_id_table = dict(zip(self.vocab, range(len(self.vocab))))
        self.id_to_word_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_id(self, word):
        if word in self.word_to_id_table:
            return self.word_to_id_table[word]
        else:
            return len(self.vocab)

    def id_to_word(self, id):
        if id==len(self.vocab):
            return "<unk>"
        elif id<len(self.vocab):
            return self.id_to_word_table[id]
        else:
            raise Exception("unknown index")

    def text_to_arr(self,text):
        arr = []
        for word in text:
            arr.append(self.word_to_id(word))
        return np.array(arr)

    def arr_to_text(self, arr):
        text = []
        for id in arr:
            text.append(self.id_to_word(id))
        return text

    # 将词典保存起来
    def save_to_file(self,filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)













