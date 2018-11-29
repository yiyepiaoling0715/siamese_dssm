#coding=utf-8
from siamese_lstm import SiameseBiLstm
from data_helper import InputHelper
import tensorflow as tf
import numpy as np
import pickle
import os
from sklearn.preprocessing import normalize

tf.flags.DEFINE_integer('rnn_size', 128, 'hidden units of RNN , as well as dimensionality of character embedding (default: 100)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability (default : 0.5)')
tf.flags.DEFINE_integer('layer_size', 4, 'number of layers of RNN (default: 2)')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch Size (default : 32)')
tf.flags.DEFINE_integer('sequence_length', 30, 'Sequence length (default : 32)')
tf.flags.DEFINE_float('grad_clip', 5.0, 'clip gradients at this value')
tf.flags.DEFINE_integer("num_epochs", 30, 'Number of training epochs (default: 200)')
tf.flags.DEFINE_float('learning_rate', 0.002, 'learning rate')
tf.flags.DEFINE_float('decay_rate', 0.97, 'decay rate for rmsprop')
tf.flags.DEFINE_string('train_file', 'train.txt', 'train raw file')
tf.flags.DEFINE_string('test_file', 'test.txt', 'train raw file')
tf.flags.DEFINE_string('data_dir', './data', 'data directory')
tf.flags.DEFINE_string('save_dir', '../data/model/', 'model save directory')
tf.flags.DEFINE_string('model_name', 'siamese_model', 'model save directory')
tf.flags.DEFINE_string('log_dir', 'log', 'log directory')
tf.flags.DEFINE_string('input_file', 'corpus.txt', 'log directory')
tf.flags.DEFINE_string('init_from', None, 'continue training from saved model at this path')
tf.flags.DEFINE_string('epoches', 10, 'continue training from saved model at this path')
FLAGS = tf.app.flags.FLAGS

tensors_filepath='../data/vector_and_chinese.tensor'
def infer():
    def feed_back(x1_in, x2_in, y_in, dropout_in):
        # print('batch_iter==>',batch_iter)
        # print('batch_iter==>',len(batch_iter))
        # print('-*- ' * 5)
        # print(batch_iter[0])
        # print(batch_iter[1])
        # print(batch_iter[2])
        return {sbl.input_x1: x1_in, sbl.input_x2: x2_in, sbl.input_y: y_in, sbl.dropout_keep_prob: dropout_in}
    datahelper = InputHelper(FLAGS.data_dir, FLAGS.input_file, FLAGS.batch_size, FLAGS.sequence_length, 0.99,is_train=True)
    vocab_size = datahelper.vocab_size
    save_path=FLAGS.save_dir+FLAGS.model_name

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    session_conf.gpu_options.allow_growth = True

    with tf.Session(config=session_conf) as sess:
        sbl=SiameseBiLstm(FLAGS.rnn_size, FLAGS.layer_size, vocab_size, FLAGS.sequence_length, FLAGS.grad_clip)
        saver=tf.train.Saver()
        saver.restore(sess,save_path)
        if not os.path.exists(tensors_filepath):
            print('tensors_filepath  不存在 生成')
            d1s,d2s=list(),list()
            d1s_chinese,d2s_chinese=list(),list()
            raw_x1,raw_x2,raw_y=datahelper.get_raw_data()

            dealed_list=list()

            for index in range(len(raw_y)):
                x1_iter,x2_iter,y_iter=raw_x1[index],raw_x2[index],raw_y[index]
                #print('x1_iter==>\n',x1_iter)
                if x1_iter.tolist() not in dealed_list:
                    dealed_list.append(x1_iter.tolist())
                    ids_in = np.array(x1_iter)
                    ids_2dim = np.expand_dims(ids_in, axis=0)
                    ids_batch_size = ids_2dim.repeat(FLAGS.batch_size, axis=0)
                    d1_iter,d2_iter=sess.run([sbl.d1,sbl.d2],feed_dict=feed_back(ids_batch_size,ids_batch_size,[0]*FLAGS.batch_size,1.0))
                    # if d1_iter[0] not in d1s:
                    d1s.append(d1_iter[0])
                    d2s.append(d2_iter[0])
                    x1_iter_chinese=datahelper.id2text(x1_iter)
                    d1s_chinese.append(x1_iter_chinese)
            d1s_array=np.array(d1s)
            with open(tensors_filepath,'wb') as fw:
                pickle.dump([d1s_array,d1s_chinese],fw)
        else:
            print('tensors_filepath  存在 加载')
            with open(tensors_filepath,'rb') as fr:
                d1s_array, d1s_chinese=pickle.load(fr)

        length_corpus_existed=len(d1s_array)
        d1s_array_normalize=normalize(d1s_array,norm='l2', axis=1)

        while True:
            str_in=input('输入文本句子')
            chars_in=list(str_in)
            ids_in=datahelper.to_array_and_padding(chars_in)
            ids_in=np.array(ids_in)
            ids_2dim=np.expand_dims(ids_in,axis=0)
            ids_batch_size=ids_2dim.repeat(FLAGS.batch_size,axis=0)
            y_batch_size=np.array([0] * FLAGS.batch_size)
            d1_iter, d2_iter = sess.run([sbl.d1, sbl.d2], feed_dict=feed_back(ids_batch_size,ids_batch_size,y_batch_size, 1.0))
            # d1_iter_2dim=np.expand_dims(d1_iter[0],axis=0)
            d1_iter_2dim=np.expand_dims(d1_iter[0],axis=0)
            print('d1_iter_2dim==>\n',d1_iter_2dim)
            d1_iter_2dim_normalize=normalize(d1_iter_2dim, norm='l2', axis=1)
            # d1_iter_2dim_normalize=np.expand_dims(d1_iter_2dim_normalize,axis=0)
            d1_iter_2dim_normalize_aligned=d1_iter_2dim_normalize.repeat(length_corpus_existed,axis=0)

            scores=np.sum(np.multiply(d1s_array_normalize,d1_iter_2dim_normalize_aligned),axis=1)
            print('scores==>\n',scores)
            index_max=np.argmax(scores,axis=0)
            print('index_max==>\n',index_max)
            chinese_max=np.array(d1s_chinese)[index_max]
            print('输出结果\n',chinese_max)
            print(chinese_max)
infer()






