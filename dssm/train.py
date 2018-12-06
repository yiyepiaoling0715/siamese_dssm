#coding:utf-8
from datetime import datetime
import numpy as np
import tensorflow as tf

from data_helper import InputHelper
from model import *

tf.app.flags.DEFINE_string('data_dir', "./output/data","")
tf.app.flags.DEFINE_string('train_dir', "./output/dssm","")
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.9,"dropout keep prob for docvecs")
tf.app.flags.DEFINE_integer('steps', 1000000,"how many steps run before end")
tf.app.flags.DEFINE_integer('batch_size', 1,"")
#tf.app.flags.DEFINE_integer('sentence_len', 1000,"input sentence length")
#tf.app.flags.DEFINE_integer('vocab_size', 4469,"vocab size")
tf.app.flags.DEFINE_integer('sentence_len', 30,"input sentence length")
# tf.app.flags.DEFINE_integer('vocab_size', 437403,"vocab size")
tf.app.flags.DEFINE_integer('embedding_size', 64,"input embedding size")
tf.app.flags.DEFINE_string('conv_filter_sizes', "1,2,3,4,5","conv2d sizes")
tf.app.flags.DEFINE_integer('conv_out_channels', 64,"conv2d out channels")
tf.app.flags.DEFINE_string('hidden_sizes', "300,256","hidden sizes, such as \"256,1000,128\"")
tf.app.flags.DEFINE_integer('NEG', 4,"NEG size")
tf.app.flags.DEFINE_float('learning_rate', 0.001,"")
tf.app.flags.DEFINE_float('l2_reg_lambda', 0.05,"")
tf.app.flags.DEFINE_float('dev_step',50,"")

activeFn = tf.nn.tanh
FLAGS = tf.app.flags.FLAGS


def main(_):
    data_dir='./data'
    log_dir='./output/logdir'
    inputhelper=InputHelper('./data', 'corpus.txt', FLAGS.batch_size, FLAGS.sentence_len, 0.9)
    vocab_size =inputhelper.vocab_size
    conv_filter_sizes = [int(num_iter) for num_iter in FLAGS.conv_filter_sizes.split(',')]
    hidden_sizes =[int(num_iter) for num_iter in FLAGS.hidden_sizes.split(',')]
    # DssmModel(sequence_length, embedding_size, vocab_size,conv_filter_sizes, conv_out_channels, hidden_sizes, batch_size=100, activeFn=tf.nn.tanh)
    dssm_model=DssmModel(FLAGS.sentence_len, FLAGS.embedding_size, vocab_size,conv_filter_sizes,
                         FLAGS.conv_out_channels, hidden_sizes, batch_size=FLAGS.batch_size, activeFn=tf.nn.tanh)
    saver=tf.train.Saver()
    summary_all=tf.summary.merge_all()
    filewriter=tf.summary.FileWriter(log_dir)
    # initial=tf.initialize_all_variables()
    # sv=tf.train.Supervisor(logdir=log_dir,summary_op=summary_all,saver=saver,global_step=global_step,
    #                        save_summaries_secs=120,save_model_secs=600,checkpoint_basename="model.ckpt",summary_writer=filewriter)
    # sv = tf.train.Supervisor(init_op=initial, logdir=log_dir, summary_op=summary_all, saver=saver,
    #                          global_step=global_step,
    #                          save_summaries_secs=120, save_model_secs=600, checkpoint_basename="model.ckpt",
    #                          summary_writer=filewriter)

    # with sv.managed_session() as sess:
    counter=0
    with tf.Session() as sess:
        print(str(datetime.now()) + ' init batches')
        dssm_model.projector()
        sess.run(tf.global_variables_initializer())
        for data_iter in inputhelper.next_batch('train',True):
            # print('data_iter==>',data_iter)
            # query,doc_pos,doc_neg=data_dir[0],data_dir[1],data_dir[2]
            # print(data_iter.shape)
            # def split_tensor(tensor_in):
            #     tensors_split=np.split(tensor_in,indices_or_sections=tensor_in.shape[1],axis=1)
            #     tensor_split_squeeze=list()
            #     for tensor_split in tensors_split:
            #         tmp=np.squeeze(tensor_split,axis=1)
            #         tensor_split_squeeze.append(tmp)
            #     return tensor_split_squeeze
            # query,doc=split_tensor(data_iter[0]),split_tensor(data_iter[1])
            # query_vecs,doc_vecs=list(),list()
            # for query_iter in query:
            #     query_vec=sess.run(dssm_model.input_x_vec,feed_dict={dssm_model.input_x:query_iter})
            #     query_vecs.append(query_vec)
            #     print('query_vec.shape==>',query_vec.shape)
            # for doc_iter in doc:
            #     doc_vec=sess.run(dssm_model.input_x_vec,feed_dict={dssm_model.input_x:doc_iter})
            #     doc_vecs.append(doc_vec)
            # print('doc_vec.shape==>',doc_vec.shape)
            # query_vec=dssm_model.predict(query,'query_vec')
            # doc_vec=dssm_model.predict(doc,'doc_vec')
            # loss=sess.run(loss_op)

            _, step, loss = sess.run([dssm_model.train_op,dssm_model.global_step,dssm_model.loss],feed_dict={dssm_model.raw_query:data_iter[0],dssm_model.raw_doc:data_iter[1]})
            if counter%FLAGS.dev_step==0:
                print(counter,loss)
            counter+=1
if __name__=='__main__':
    tf.app.run()