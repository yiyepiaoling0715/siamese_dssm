import tensorflow as tf
import random


class DssmModel(object):
    def __init__(self, sequence_length, embedding_size, vocab_size,
                 conv_filter_sizes, conv_out_channels, hidden_sizes, batch_size, activeFn=tf.nn.relu,
                 train_flag=True):
        """embedding 参数，   conv 参数，conv—bias参数，   全连接参数"""
        self.conv_out_channels = conv_out_channels
        self.activeFn = activeFn
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.l2_reg_lambda = 0.05
        self.train_flag = train_flag
        self.loss_l2 = tf.constant(0.0, dtype=tf.float32)
        with tf.name_scope('embedding'), tf.variable_scope('embedding'):
            initial_value = tf.truncated_normal([vocab_size, embedding_size], mean=0.0, stddev=0.1, dtype=tf.float32)
            self.embedding = tf.Variable(initial_value=initial_value, trainable=True, dtype=tf.float32)

        with tf.name_scope('conv_variables'), tf.variable_scope('conv_variables'):
            conv_filters, max_pool_sizes = list(), list()
            for conv_filter_size_iter in conv_filter_sizes:
                initial_value = tf.truncated_normal([conv_filter_size_iter, embedding_size, 1, conv_out_channels],
                                                    mean=0.0, stddev=0.1, dtype=tf.float32)
                filter_iter = tf.Variable(initial_value=initial_value, dtype=tf.float32)
                initial_value_b = tf.constant(0.1, dtype=tf.float32, shape=[conv_out_channels])
                bias_conv = tf.Variable(initial_value=initial_value_b, dtype=tf.float32)
                conv_filters.append((filter_iter, bias_conv))
                maxpool_size_iter = [1, sequence_length - conv_filter_size_iter + 1, 1, 1]
                max_pool_sizes.append(maxpool_size_iter)
                self.loss_l2 += tf.nn.l2_loss(filter_iter)
                self.loss_l2 += tf.nn.l2_loss(bias_conv)
            self.conv_filters, self.max_pool_sizes = conv_filters, max_pool_sizes

        with tf.name_scope('fc_variables'), tf.variable_scope('fc_variables'):
            fc_paras = list()
            fc_sizes = [len(conv_filter_sizes) * conv_out_channels] + hidden_sizes
            print('fc_sizes==>', fc_sizes)
            for index in range(len(fc_sizes) - 1):

                initial_value = tf.truncated_normal(mean=0.0, stddev=0.1, dtype=tf.float32,
                                                    shape=[fc_sizes[index], fc_sizes[index + 1]])
                weight_iter = tf.Variable(initial_value=initial_value, trainable=True, dtype=tf.float32)
                bias_iter = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[fc_sizes[index + 1]]))
                fc_paras.append((weight_iter, bias_iter))

                self.loss_l2 += tf.nn.l2_loss(weight_iter)
                self.loss_l2 += tf.nn.l2_loss(bias_iter)

            self.fc_paras = fc_paras

    def batch_normalization(self, x, training, name):
        """
        """
        print('training, name==>',training, name)
        with tf.variable_scope(name):
            if training:
                bn_train = tf.layers.batch_normalization(x, training=True, reuse=False, name=name)
            elif not training:
                bn_train = tf.layers.batch_normalization(x, training=False, reuse=True, name=name)
            # z = tf.cond(tf.equal(training,True), lambda: bn_train, lambda: bn_inference)
            else:
                raise ValueError('bn_triain error')

        return bn_train

    def predict(self, input_x_in):
        query_embedding = tf.nn.embedding_lookup(self.embedding, input_x_in)
        query_embedding = tf.expand_dims(query_embedding, axis=-1)
        # query_embedding=tf.reshape(query_embedding,shape=[self.batch_size*query_embedding.shape[1],query_embedding.shape[2],query_embedding.shape[3]])
        with tf.name_scope('conv'), tf.variable_scope('conv'):
            pools = list()
            for index in range(len(self.conv_filters)):
                # print('self.conv_filters[index]==>',self.conv_filters[index])
                # w_conv_filters = tf.Variable(initial_value=tf.truncated_normal(shape=self.conv_filters[index][0], stddev=0.1), name="W",dtype=tf.float32)
                stride_iter = [1, 1, 1, 1]
                query_embedding_conv_iter = tf.nn.conv2d(query_embedding, self.conv_filters[index][0], stride_iter,
                                                         'VALID', name=None)
                # convs.append(query_embedding_conv_iter)
                stride_iter = [1, 1, 1, 1]
                query_embedding_conv_iter_bias = tf.nn.bias_add(query_embedding_conv_iter, self.conv_filters[index][1])
                query_embedding_max_pool = tf.nn.max_pool(query_embedding_conv_iter_bias, self.max_pool_sizes[index],
                                                          stride_iter, 'VALID', name='max_pool')
                pools.append(query_embedding_max_pool)
            pools_concat = tf.concat(pools, axis=1)
            pools_concat_reshape = tf.reshape(pools_concat, [-1, self.conv_out_channels * len(self.conv_filters)])
        tensor_fc = pools_concat_reshape
        with tf.name_scope('fc'), tf.variable_scope('fc'):
            for index, (weight_iter, bias_iter) in enumerate(self.fc_paras):
                print('index==>',index)
                tensor_fc = self.activeFn(tf.nn.xw_plus_b(tensor_fc, weight_iter, bias_iter))
                # tensor_fc = self.batch_normalization(tensor_fc1, self.train_flag, 'fc_normalization_{}'.format(index))
        return tensor_fc

    def input_sent2vec(self):
        self.raw_query = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.sequence_length], name='query')
        self.raw_doc = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, 5, self.sequence_length], name='doc')
        raw_docs = tf.split(self.raw_doc, num_or_size_splits=self.raw_doc.shape[1], axis=1)
        query_vec = self.predict(self.raw_query)
        doc_vecs = list()
        for doc_iter in raw_docs:
            doc_iter = tf.squeeze(doc_iter, 1)
            doc_vec = self.predict(doc_iter)
            doc_vecs.append(doc_vec)
        doc_vecs_concat = tf.concat(doc_vecs, axis=0)
        return query_vec, doc_vecs_concat

    def loss_op(self, query_vec, doc_vecs_raw, neg=4):
        """
            query_vec:[ batch_size,channel]
            制作 doc_vec neg_length     计算 q,d 长度值，   计算 q，d 点乘积，得出cosine
            计算 softmax，   得出 loss_pos
        """
        tmp = tf.tile(query_vec, multiples=[1, 1])
        # doc_vecs = tf.concat(doc_vecs_raw, axis=0)
        # doc_vecs [batch_size*neg_length,channel]
        doc_vecs=doc_vecs_raw
        query_norm = tf.sqrt(tf.reduce_sum(tf.square(query_vec), axis=1, keep_dims=True))  # [batch_size,1]
        doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_vecs), axis=1, keep_dims=True))  # [batch_size*neg_len,1]
        norm = tf.multiply(tf.tile(query_norm, multiples=[5, 1]), doc_norm)  # [batch_size*neg_len,1]
        query_vecs = tf.tile(query_vec, multiples=[neg + 1, 1])
        # batch_size*neg_len,channel
        numberator_raw = tf.reduce_sum(tf.multiply(query_vecs, doc_vecs), axis=1)# [batch_size*neg_len,1]
        numberator=tf.expand_dims(numberator_raw,axis=-1)
        # cosine_raw = tf.truediv(numberator , tf.maximum(norm,tf.constant(0.000001,dtype=tf.float32,shape=norm.shape)))
        cosine_raw = tf.truediv(numberator, norm)
        pos_and_neg = tf.transpose(tf.reshape(tf.transpose(cosine_raw), shape=[neg + 1, -1]))  # [batch_size,neg]
        pos_and_neg_softmax = tf.nn.softmax(pos_and_neg)
        pos = tf.slice(pos_and_neg_softmax, [0, 0], [-1, 1])
        neg = tf.slice(pos_and_neg_softmax, [0, 1], [-1, -1])
        self.pos=pos
        # loss = -tf.reduce_mean(tf.log(tf.maximum(pos,tf.constant(0.000001,dtype=tf.float32,shape=pos.shape))))
        loss = -tf.reduce_mean(tf.log(pos))
        return loss

    def projector(self):
        query_vec, doc_vecs_concat = self.input_sent2vec()
        query_vec_dropout=tf.nn.dropout(query_vec,0.5)
        doc_vecs_dropout=tf.nn.dropout(doc_vecs_concat,0.5)
        self.loss = self.loss_op(query_vec_dropout, doc_vecs_dropout)
        loss_train = self.loss + self.loss_l2 * self.l2_reg_lambda
        vars = tf.trainable_variables()
        gradients = tf.gradients(loss_train, vars)
        # gradient_only=[grad_iter for grad_iter,_ in list(gradient_variable_pairs)]
        # var_only=[var_iter for _,var_iter in list(gradient_variable_pairs)]
        grads_cliped, _ = tf.clip_by_global_norm(gradients, 5)
        optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = optimizer.apply_gradients(list(zip(grads_cliped, vars)))

    def read_pre_emb(self):
        pass
