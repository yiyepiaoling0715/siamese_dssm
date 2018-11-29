import os
import tensorflow as tf
from siamese_lstm import SiameseBiLstm
from data_helper import InputHelper
import random

tf.flags.DEFINE_integer('rnn_size', 16, 'hidden units of RNN , as well as dimensionality of character embedding (default: 100)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability (default : 0.5)')
tf.flags.DEFINE_integer('layer_size', 2, 'number of layers of RNN (default: 2)')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch Size (default : 32)')
tf.flags.DEFINE_integer('sequence_length', 20, 'Sequence length (default : 32)')
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
tf.flags.DEFINE_string('epoches', 1000, 'continue training from saved model at this path')
FLAGS = tf.app.flags.FLAGS

best_loss = 9999
if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)
savepath=FLAGS.save_dir+FLAGS.model_name
def main(_):
    def feed_back(x1_in, x2_in, y_in, dropout_in):
        # print('batch_iter==>',batch_iter)
        # print('batch_iter==>',len(batch_iter))
        # print('-*- ' * 5)
        # print(batch_iter[0])
        # print(batch_iter[1])
        # print(batch_iter[2])
        return {sbl.input_x1: x1_in, sbl.input_x2: x2_in, sbl.input_y: y_in, sbl.dropout_keep_prob: dropout_in}

    def train_step(batch_in):
        if random.random()<0.5:
            batches = feed_back(batch_in[0], batch_in[1], batch_in[2], FLAGS.dropout_keep_prob)
        else:
            batches = feed_back(batch_in[1], batch_in[0], batch_in[2], FLAGS.dropout_keep_prob)
        loss, merged_result, _ = sess.run([sbl.loss, merged, sbl.optimizer], feed_dict=batches)
        writer.add_summary(merged_result)

    def dev_step():
        global best_loss
        loss_counter = [0, 0]
        print('进入测试阶段')
        for batch_iter_test in datahelper.next_batch(train_or_test='test', random=True):
            batches_test = feed_back(batch_iter_test[0], batch_iter_test[1], batch_iter_test[2], 1.0)
            loss_test, merged_result_test, _ = sess.run([sbl.loss, merged, sbl.optimizer],
                                                        feed_dict=batches_test)
            loss_counter[0] += loss_test
            loss_counter[1] += 1
        loss_average = loss_counter[0] / loss_counter[1]
        if loss_average<best_loss:
            saver.save(sess, savepath)
            best_loss=loss_average
        print('测试:\tloss_average:\t{}\t'.format(loss_average))

    datahelper = InputHelper(FLAGS.data_dir, FLAGS.input_file, FLAGS.batch_size, FLAGS.sequence_length, 0.99,is_train=True)
    vocab_size =datahelper.vocab_size
    # with tf.Graph().as_default():
    counter = 0

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        session_conf.gpu_options.allow_growth = True


        sess = tf.Session(config=session_conf)
        with sess.as_default():
            writer = tf.summary.FileWriter(FLAGS.data_dir, graph=sess.graph)
            sbl = SiameseBiLstm(FLAGS.rnn_size, FLAGS.layer_size, vocab_size, FLAGS.sequence_length, FLAGS.grad_clip)
            # if os.path.exists(savepath+'.meta'):
            saver = tf.train.Saver()
            if tf.train.checkpoint_exists(savepath):
                print('加载已经保存的模型，保存路径\t{}'.format(str(savepath)))
                saver.restore(sess,savepath)

            tf.summary.scalar(tensor=sbl.loss, name='loss')
            merged = tf.summary.merge_all()

            sess.run(tf.global_variables_initializer())
            for epoch in range(FLAGS.epoches):
                print('epoch 一次')
                for batch_iter in datahelper.next_batch(train_or_test='train',random=True):
                    train_step(batch_iter)
                    # if counter%10==0:
                    #     print('epoch:{}\tcounter:{}\tloss:{}'.format(epoch,counter,loss))
                    if counter%FLAGS.num_epochs==100:
                        pass
                    # if counter%20==0:
                    counter+=1
                    if counter%50==0:
                        # print(batches)
                        dev_step()

if __name__=='__main__':
    tf.app.run()

