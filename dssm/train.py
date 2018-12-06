#coding:utf-8
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.python import  debug as tf_debug
from data_helper import InputHelper
from model import *

tf.app.flags.DEFINE_string('data_dir', "./output/data","")
tf.app.flags.DEFINE_string('train_dir', "./output/dssm","")
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.9,"dropout keep prob for docvecs")
tf.app.flags.DEFINE_integer('steps', 1000000,"how many steps run before end")
tf.app.flags.DEFINE_integer('batch_size', 32,"")
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
    dssm_model.projector()

    saver=tf.train.Saver()
    summary_all=tf.summary.merge_all()
    filewriter=tf.summary.FileWriter(log_dir)

    counter=0
    min_loss=99999

    session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    sess=tf.Session(config=session_conf)
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    with sess as sess:
        print(str(datetime.now()) + ' init batches')
        sess.run(tf.global_variables_initializer())

        for epoch in range(FLAGS.steps):
            for data_iter in inputhelper.next_batch('train',True):
                _, step, loss = sess.run([dssm_model.train_op,dssm_model.global_step,dssm_model.loss],feed_dict={dssm_model.raw_query:data_iter[0],dssm_model.raw_doc:data_iter[1]})
                if counter%FLAGS.dev_step==0:
                    loss_test=0
                    counter_test=0
                    for data_iter_test in inputhelper.next_batch('test',True):
                        # print('data_iter_test[1]==>',data_iter_test[1])
                        _, pos_test,step, loss_test_iter = sess.run([dssm_model.train_op, dssm_model.pos,dssm_model.global_step, dssm_model.loss],
                                                      feed_dict={dssm_model.raw_query: data_iter_test[0],dssm_model.raw_doc: data_iter_test[1]})
                        loss_test+=loss_test_iter
                        print('loss_test_iter,pos_test===>',loss_test_iter,pos_test.shape)
                        counter_test+=1
                    print('epoch\t{} ,counter\t{} ,loss_test/counter_test\t{}'.format(epoch,counter,loss_test/counter_test))
                    if loss_test<min_loss:
                        saver.save(sess,FLAGS.train_dir)
                        min_loss=loss_test
                counter+=1
if __name__=='__main__':
    tf.app.run()