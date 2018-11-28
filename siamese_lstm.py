# -*- coding:utf-8 -*-

import tensorflow as tf


class SiameseBiLstm(object):
    def bi_lstm(self, rnn_size, layer_size, keep_prob):
        with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
            _cells = [tf.nn.rnn_cell.LSTMCell(rnn_size, forget_bias=1.0,state_is_tuple=True) for _ in range(layer_size)]
            multicell_fw = tf.nn.rnn_cell.MultiRNNCell(_cells, state_is_tuple=True)
            dropoutcells_fw = tf.nn.rnn_cell.DropoutWrapper(multicell_fw, output_keep_prob=keep_prob, state_keep_prob=1.0)
        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            _cells = [tf.nn.rnn_cell.LSTMCell(rnn_size,forget_bias=1.0,state_is_tuple=True) for _ in range(layer_size)]
            dropoutcells_bw = multicell_bw = tf.nn.rnn_cell.MultiRNNCell(_cells, state_is_tuple=True)
            tf.nn.rnn_cell.DropoutWrapper(multicell_bw, input_keep_prob=1.0, output_keep_prob=keep_prob, state_keep_prob=1.0)
        return dropoutcells_fw, dropoutcells_bw

    def weight_variables(self, shape, name):
        initial_value = tf.truncated_normal(shape, mean=0.0, stddev=0.1, dtype=tf.float32)
        variable = tf.Variable(initial_value=initial_value, trainable=True,name=name,dtype=tf.float32)
        return variable

    def bias_variables(self, shape, name):
        # todo bias 如何赋值
        initial_value = tf.constant(value=1.0, dtype=tf.float32, shape=shape)
        variable = tf.Variable(initial_value=initial_value, trainable=True,name=name, dtype=tf.float32)
        return variable

    def transform_inputs(self, inputs, rnn_size, sequence_length):
        # todo 存在问题
        inputs_transpose = tf.transpose(inputs, perm=[1, 0, 2])
        # inputs_transpose=inputs
        inputs_transpose = tf.reshape(inputs_transpose, shape=[-1, rnn_size])
        inputs_transpose = tf.split(inputs_transpose, sequence_length, 0)
        return inputs_transpose

    def contrastive_loss(self, Ew, y):
        # todo 平方还是其他值？
        tmp1 = y * tf.square(y- Ew)
        tmp2 = (1 - y) * tf.square(tf.maximum(Ew, 0)-y)
        return tf.reduce_sum(tmp1 + tmp2)
    def contrastive_loss01(self, Ew, y):
        # todo 平方还是其他值？

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(Ew) + (y - 1) * tf.log(1 - Ew)))
        tmp1 = y * tf.log(Ew)
        tmp2 = (1 - y) * tf.log(tf.maximum(1-Ew,0))
        # return tf.reduce_sum(tmp1 + tmp2)
        return cross_entropy
    def score01(self,d1,d2):
        with tf.variable_scope('calculation'):
            numberator = tf.sqrt(tf.reduce_sum(tf.multiply(d1, d2), axis=1))
            denominator1 = tf.sqrt(tf.reduce_sum(tf.square(d1), axis=1))
            denominator2 = tf.sqrt(tf.reduce_sum(tf.square(d2), axis=1))
            Ew = 2 * numberator / (denominator1 * denominator2)
            return Ew
    def score(self,d1,d2):
        # with tf.variable_scope('calculation'):
        numberator = tf.reduce_sum(tf.multiply(d1, d2), axis=1)
        denominator1 = tf.sqrt(tf.reduce_sum(tf.square(d1), axis=1))
        denominator2 = tf.sqrt(tf.reduce_sum(tf.square(d2), axis=1))
        Ew = numberator / (denominator1 * denominator2)
        return Ew

    def __init__(self, rnn_size, layer_size
                 , vocab_size, sequence_length, grad_clip):
        print('sequence_length==>',sequence_length)
        self.input_x1 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name='input_x1')
        self.input_x2 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name='input_x2')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        with tf.variable_scope('embedding'):
            # shape,  mean=0.0,  stddev=1.0,   dtype=dtypes.float32
            embedding_initial_value = tf.truncated_normal(shape=[vocab_size, rnn_size], mean=0.0, stddev=0.5)
            embedding_matrix = tf.Variable(initial_value=embedding_initial_value, trainable=True, collections=None,
                                           validate_shape=True,
                                           caching_device=None, name='embedding_matrix', variable_def=None,
                                           dtype=tf.float32, expected_shape=None, import_scope=None)
            embedding_x1 = tf.nn.embedding_lookup(embedding_matrix, self.input_x1, partition_strategy="mod", name=None,
                                                  validate_indices=True, max_norm=None)
            embedding_x2 = tf.nn.embedding_lookup(embedding_matrix, self.input_x2, partition_strategy="mod", name=None,
                                                  validate_indices=True, max_norm=None)
        with tf.variable_scope('output'):
            multicell_fw, multicell_bw = self.bi_lstm(rnn_size, layer_size, self.dropout_keep_prob)
            embedding_x1_transform = self.transform_inputs(embedding_x1, rnn_size, sequence_length)
            embedding_x2_transform = self.transform_inputs(embedding_x2, rnn_size, sequence_length)

            # initial_state_fw = multicell_fw.zero_state(self.input_x1.shape[0], tf.float32)
            # initial_state_bw = multicell_bw.zero_state(self.input_x1.shape[0], tf.float32)
            output1, _, _ = tf.nn.static_bidirectional_rnn(multicell_fw, multicell_bw, embedding_x1_transform,dtype=tf.float32)
            tf.get_variable_scope().reuse_variables()
            output2, _, _ = tf.nn.static_bidirectional_rnn(multicell_fw, multicell_bw, embedding_x2_transform,dtype=tf.float32)
            output1=tf.nn.relu(output1)
            output2=tf.nn.relu(output2)
            output1 = tf.reduce_mean(output1, axis=0)
            output2 = tf.reduce_mean(output2, axis=0)

        with tf.variable_scope('fc1'):
            # todo w shape 再斟酌一下
            weight_fc1 = self.weight_variables([2 * rnn_size, 128], 'weight_fc1')
            bias_fc1 = self.bias_variables([128], 'bias_fc1')
            d1 = tf.nn.xw_plus_b(output1, weight_fc1, bias_fc1, name='d1')
            d1=tf.nn.relu(d1)
        with tf.variable_scope('fc2'):
            weight_fc2 = self.weight_variables([2 * rnn_size, 128], 'weight_fc2')
            bias_fc2 = self.bias_variables([128], 'bias_fc2')
            d2 = tf.nn.xw_plus_b(output2, weight_fc2, bias_fc2, name='d2')
            d2=tf.nn.relu(d2)
        with tf.variable_scope('calculation'):
            Ew=self.score(d1,d2)
            # numberator = tf.sqrt(tf.reduce_sum(tf.multiply(d1, d2), axis=1))
            # denominator1 = tf.sqrt(tf.reduce_sum(tf.square(d1), axis=1))
            # denominator2 = tf.sqrt(tf.reduce_sum(tf.square(d2), axis=1))
            # Ew = 2 * numberator / (denominator1 * denominator2)
            # self.Ew_see=Ew
            # Ew=tf.nn.sigmoid(Ew)
        with tf.variable_scope('loss'):
            self.loss = self.contrastive_loss(Ew, self.input_y)
        with tf.variable_scope('train'):
            variables = tf.trainable_variables()
            grads = tf.gradients(self.loss, variables)
            grads_cliped, _ = tf.clip_by_global_norm(grads, grad_clip)
            # loss_on_varables=tf.gradients(self.loss,variables)
            optimizer = tf.train.AdamOptimizer(0.001)
            self.optimizer = optimizer.apply_gradients(list(zip(grads_cliped, variables)))
            # gradient_variable=optimizer.compute_gradients(self.loss, var_list=variables,aggregation_method=None,
            #             colocate_gradients_with_ops=False,grad_loss=None)
            # list_clipped,_=tf.clip_by_global_norm(gradient_variable, grad_clip, use_norm=None, name=None)
            # self.operator=optimizer.apply_gradients(list_clipped, global_step=None, name=None)


if __name__ == '__main__':
    Similarity(64, 4, 1000, 100, 0.5, 5.0)
