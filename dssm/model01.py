import tensorflow as tf
import random


class DssmModel(object):
    def __init__(self, sequence_length, embedding_size, vocab_size,
                 conv_filter_sizes, conv_out_channels, hidden_sizes, batch_size=100, activeFn=tf.nn.tanh):
        """embedding 参数，   conv 参数，conv—bias参数，   全连接参数"""
        self.conv_out_channels = conv_out_channels
        self.activeFn = activeFn
        self.batch_size = batch_size
        self.sequence_length=sequence_length
        with tf.name_scope('embedding'), tf.variable_scope('embedding'):
            initial_value = tf.truncated_normal([vocab_size, embedding_size], mean=0.0, stddev=0.1, dtype=tf.float32,
                                                seed=None, name=None)
            self.embedding = tf.Variable(initial_value=initial_value, trainable=True, name=None, dtype=None)

        with tf.name_scope('conv_variables'), tf.variable_scope('conv_variables'):
            conv_filters, max_pool_sizes = list(), list()
            for conv_filter_size_iter in conv_filter_sizes:
                filter_iter = tf.Variable(tf.truncated_normal([conv_filter_size_iter, embedding_size, 1, conv_out_channels],stddev=0.1),dtype=tf.float32)
                bias_conv = tf.constant(0.1, dtype=tf.float32, shape=[conv_out_channels])
                conv_filters.append((filter_iter, bias_conv))
                maxpool_size_iter = [1, sequence_length - conv_filter_size_iter + 1, 1, 1]
                max_pool_sizes.append(maxpool_size_iter)

            self.conv_filters, self.max_pool_sizes = conv_filters, max_pool_sizes

        with tf.name_scope('fc_variables'), tf.variable_scope('fc_variables'):
            fc_paras = list()
            fc_sizes =  [len(conv_filter_sizes) * conv_out_channels]+hidden_sizes
            # print('fc_sizes==>',fc_sizes)
            for index in range(len(fc_sizes) - 1):
                initial_value = tf.truncated_normal(mean=0.0,stddev=0.1,dtype=tf.float32,shape=[fc_sizes[index], fc_sizes[index + 1]])
                weight_iter = tf.Variable(initial_value=initial_value, trainable=True, name=None, dtype=tf.float32)
                bias_iter = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[fc_sizes[index + 1]]))
                fc_paras.append((weight_iter, bias_iter))
            self.fc_paras = fc_paras
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.sequence_length],name='inpu_x')
        # self.query_vec=self.predict_query()
        # self.doc_vec=self.predict_doc()
        self.input_x_vec=self.predict()
    def predict(self):
        query_embedding = tf.nn.embedding_lookup(self.embedding, self.input_x)
        query_embedding=tf.expand_dims(query_embedding,axis=-1)
        # query_embedding=tf.reshape(query_embedding,shape=[self.batch_size*query_embedding.shape[1],query_embedding.shape[2],query_embedding.shape[3]])
        with tf.name_scope('conv'), tf.variable_scope('conv'):
            pools = list()
            for index in range(len(self.conv_filters)):
                # print('self.conv_filters[index]==>',self.conv_filters[index])
                # w_conv_filters = tf.Variable(initial_value=tf.truncated_normal(shape=self.conv_filters[index][0], stddev=0.1), name="W",dtype=tf.float32)
                stride_iter = [1, 1, 1, 1]
                query_embedding_conv_iter = tf.nn.conv2d(query_embedding,self.conv_filters[index][0] , stride_iter,'VALID', name=None)
                # convs.append(query_embedding_conv_iter)
                stride_iter = [1, 1, 1, 1]
                query_embedding_max_pool = tf.nn.max_pool(query_embedding_conv_iter, self.max_pool_sizes[index],
                                                          stride_iter, 'VALID', name=None)
                pools.append(query_embedding_max_pool)
            pools_concat = tf.concat(pools, axis=1)
            pools_concat_reshape = tf.reshape(pools_concat, [-1, self.conv_out_channels * len(self.conv_filters)])
        tensor_fc = pools_concat_reshape
        with tf.name_scope('fc'), tf.variable_scope('fc'):
            for (weight_iter, bias_iter) in self.fc_paras:
                tensor_fc = self.activeFn(tf.nn.xw_plus_b(tensor_fc, weight_iter, bias_iter))
        return tensor_fc
    def predict_query(self):
        self.raw_query=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.sequence_length],name='query')
        self.raw_doc = tf.placeholder(dtype=tf.int32, shape=[self.batch_size,5,self.sequence_length], name='doc')
        query_vec=self.predict(self.raw_query)
        doc_vec=self.predict(self.raw_doc)
        return query_vec
    def predict_doc(self):
        doc_vec = self.predict(self.raw_doc, 'doc')
        return doc_vec
    def loss_op(self, query_vec, doc_vecs_raw, neg=4, l2_reg_lambda=0.05):
        """
            query_vec:[ batch_size,channel]
            制作 doc_vec neg_length     计算 q,d 长度值，   计算 q，d 点乘积，得出cosine
            计算 softmax，   得出 loss_pos
        """
        tmp = tf.tile(query_vec, multiples=[1,1])
        # doc_vecs = tf.tile(doc_vecs_raw, multiples=[1,1])
        # for index in range(neg):
        #     rand_num = random.randint(0, (self.batch_size + index) % self.batch_size)
            # tmp1 = tf.slice(doc_vecs, begin=[rand_num, 0], size=[self.batch_size - rand_num, -1])
            # tmp2 = tf.slice(doc_vecs, begin=[0, 0], size=[rand_num, -1])
        doc_vecs = tf.concat(doc_vecs_raw, axis=0)
        # doc_vecs [batch_size*neg_length,channel]
        query_norm = tf.sqrt(tf.reduce_sum(tf.square(query_vec), axis=1, keep_dims=True))  # [batch_size,1]
        doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_vecs), axis=1, keep_dims=True))  # [batch_size*neg_len,1]
        norm = tf.multiply(query_norm, doc_norm)  # [batch_size*neg_len,1]
        query_vecs = tf.tile(query_vec, multiples=[neg + 1, 1])
        # batch_size*neg_len,channel
        numberator = tf.reduce_sum(tf.multiply(query_vecs, doc_vecs), axis=1)  # [batch_size*neg_len,1]
        cosine_raw = numberator / norm
        pos_and_neg = tf.transpose(tf.reshape(tf.transpose(cosine_raw), shape=[neg + 1, -1]))  # [batch_size,neg]
        pos_and_neg_softmax = tf.nn.softmax(pos_and_neg)
        pos = tf.slice(pos_and_neg_softmax, [0, 0], [-1, 1])
        neg = tf.slice(pos_and_neg_softmax, [0, 1], [-1, -1])
        loss = -tf.reduce_mean(tf.log(pos))
        self.loss = loss
        return loss

    def read_pre_emb(self):
        pass
