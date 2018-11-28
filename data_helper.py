# -*- coding:utf-8 -*-
"""
应用于Siamese LSTM的data util
输入文本为清洗好的文本,格式为
seq1_token1 seq1_token2 seq1_token2 ... seq1_tokenN\tseq2_token1 seq2_token2 seq2_token3 ... seq2_tokenN\tlabel
文本1与文本2以及label用"\t"隔开
文本之间的token使用空格" "隔开
label为0或1表示相似与不相似
"""

import os
import pickle
import numpy as np
import collections
from sklearn.model_selection import train_test_split
class InputHelper():

    def __init__(self, data_dir, input_file, batch_size, sequence_length, split_ratio,is_train=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.split_ratio=split_ratio

        # vocab_file = os.path.join(data_dir, 'vocab.pkl')
        vocab_file = os.path.join(data_dir, 'vocab.dict')
        input_file = os.path.join(data_dir, input_file)

        if not os.path.exists(vocab_file):
            with open(vocab_file,'w',encoding='utf-8') as fw:
                word2id, id2word,vocab_size = self.preprocess(input_file)
                self.vocab_size = vocab_size
                self.word2id_dict = word2id
                self.id2word_dict = id2word

                word2id_sorted=sorted(self.word2id_dict.items(),key=lambda x:x[1],reverse=False)
                for (word,id) in word2id_sorted:
                    fw.write(word+'\t'+str(id)+'\n')
                # pickle.dump([self.vocab_size,self.word2id_dict,self.id2word_dict],fw)
        else:
            with open(vocab_file,'r',encoding='utf-8') as fr:
                # self.vocab_size,self.word2id_dict,self.id_word_dict=pickle.load(fr)
                word2id_dict,id2word_dict,vocab_size=dict(),dict(),0
                for line in fr:
                    word,id=line.strip('\r\n ').split('\t')
                    word2id_dict[word]=int(id)
                    id2word_dict[int(id)]=word
                vocab_size=len(word2id_dict)
                self.word2id_dict,self.id2word_dict,self.vocab_size=word2id_dict,id2word_dict,vocab_size
        x1,x2,y=self.create_batches(input_file)
        self.x1_train, self.x1_test, self.x2_train, self.x2_test, self.y_train, self.y_test =self.split_corpus(x1,x2,y)
    def get_sent_length(self,sents_in):
        lengths=collections.defaultdict(int)
        for sent_iter in sents_in:
            lengths[len(sent_iter)]+=1
        lengths2counters=sorted(lengths.items(),key=lambda x:x[1],reverse=True)
        # print(lengths2counters)
    def preprocess(self, input_file, min_freq=2):
        word2id=collections.defaultdict(int)
        id2word=collections.defaultdict(int)
        word2counter=collections.defaultdict(int)
        # seqs_tmp=list()
        with open(input_file,'r',encoding='utf-8') as fr:
            for line in fr:
                seq1,seq2,label=line.split('\t')

                seq=seq1+' '+seq2
                words_line=seq.split(' ')
                for word_iter in words_line:
                    word2counter[word_iter]+=1
        word2counter['<pad>']=99999
        word2counter_truncate=[(word,counter) for word,counter in word2counter.items() if counter>=min_freq]
        word2counter_sorted=sorted(word2counter_truncate,key=lambda x:x[1],reverse=True)
        for index,word2counter in enumerate(word2counter_sorted):
            word2id[word2counter[0]]=index
            id2word[index]=word2counter[0]
        return word2id,id2word,len(word2id)
    def load_vocab(self, vocab_file):
        assert os.path.exists(vocab_file)
        with open(vocab_file,'rb') as fr:
            word2ids=pickle.load(fr)
            return word2ids,len(word2ids)
    def text_to_array(self, text, is_clip=True):
        # print('text==>',text)
        text_ids=[self.word2id_dict.get(word) for word in text]
        if is_clip:
            text_ids_clip=text_ids[:self.sequence_length]
        else:
            text_ids_clip=text_ids
        # print('text_ids_clip==>',text_ids_clip)
        return text_ids_clip
    def id2text(self,sent_in):
        words=list()
        for char in sent_in:
            word=self.id2word_dict[char]
            words.append(word)
        return words
    def split_corpus(self,x1,x2,y):
        return train_test_split(x1,x2,y,train_size=self.split_ratio)

    def create_batches(self, text_file):
        x1,x2,y=list(),list(),list()
        sents_tmp=list()
        with open(text_file,'r',encoding='utf-8') as fr:
            for line in fr:
                x1_iter,x2_iter,y_iter=line.split('\t')

                sents_tmp.append(x1_iter.split(' '))
                sents_tmp.append(x2_iter.split(' '))

                ids_x1_iter=self.text_to_array(x1_iter.split(' '),True)
                ids_x2_iter=self.text_to_array(x2_iter.split(' '),True)
                ids_x1_iter_padding=self.padding_seq(ids_x1_iter,self.word2id_dict.get('<pad>'))
                ids_x2_iter_padding=self.padding_seq(ids_x2_iter,self.word2id_dict.get('<pad>'))
                id_y_iter=int(y_iter)
                # print('ids_x1_iter_padding,ids_x2_iter_padding,y_iter==>\n',ids_x1_iter_padding,ids_x2_iter_padding,y_iter)
                x1.append(ids_x1_iter_padding)
                x2.append(ids_x2_iter_padding)
                y.append(id_y_iter)

        self.get_sent_length(sents_tmp)

        x1_array=np.array(x1)
        x2_array=np.array(x2)
        y_array=np.array(y)
        return x1_array,x2_array,y_array

    def padding_seq(self, seq_array, padding_index):
        seq_array+=[padding_index]*(self.sequence_length-len(seq_array))
        return seq_array

    def next_batch(self,train_or_test,random=True):
        if train_or_test=='train':
            x1_array=np.array(self.x1_train)
            x2_array=np.array(self.x2_train)
            y_array=np.array(self.y_train)
        elif train_or_test=='test':
            x1_array = np.array(self.x1_test)
            x2_array = np.array(self.x2_test)
            y_array = np.array(self.y_test)
        else:
            raise ValueError('train_or_test 输入不对')
        length=len(y_array)-1
        print('输入的语料长度  {}'.format(length))
        batch_number,_=np.divmod(length,self.batch_size)
        if random:
            print('random  打乱顺序 一次')
            random_series=np.random.permutation(range(length))
        else:
            random_series=range(length)
        for batch_index in range(int(batch_number)):
            x1_iter=x1_array[random_series][batch_index*self.batch_size:(batch_index+1)*self.batch_size]
            x2_iter=x2_array[random_series][batch_index*self.batch_size:(batch_index+1)*self.batch_size]
            y_iter=y_array[random_series][batch_index*self.batch_size:(batch_index+1)*self.batch_size]
            # print('x1_iter==>',x1_iter,'x2_iter==>',x2_iter,'y_iter==>',y_iter)
            yield (x1_iter,x2_iter,y_iter)


if __name__ == '__main__':
    data_loader = InputHelper('data', 'corpus.txt', 32, 20)
    # x1, x2, y = data_loader.next_batch()
    for x1_iter,x2_iter,y_iter in data_loader.next_batch():
        # print('x1_iter,x2_iter,y_iter==>',x1_iter,x2_iter,y_iter)
        pass