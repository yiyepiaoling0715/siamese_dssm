import tensorflow as tf

n_layers=3

class SiameseW2v(object):
    def __init__(self, sequence_length, vocab_size, embedding_size, hidden_units, l2_reg_lambda, batch_size,trainableEmbeddings):

        # Keeping track of l2 regularization loss (optional)
        #todo 添加上  l1,l2 正则化
        l2_loss = tf.constant(0.0, name="l2_loss")

        self.input_x1=tf.placeholder(dtype=tf.float32, shape=[batch_size,sequence_length], name='input_x1')
        self.input_x2=tf.placeholder(dtype=tf.float32, shape=[batch_size,sequence_length], name='input_x2')
        self.input_y=tf.placeholder(dtype=tf.float32,shape=[batch_size],name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.variable_scope('embedding'):
            initial_value = tf.random_uniform([vocab_size, embedding_size],minval=-1.0,maxval=1.0,dtype=tf.float32)
            Embeddings=tf.Variable(initial_value=initial_value,trainable=True,collections=None,validate_shape=True,
                                   caching_device=None,name=None,variable_def=None,dtype=None,expected_shape=None,import_scope=None)
            embedding_x1=tf.nn.embedding_lookup(Embeddings,self.input_x1,partition_strategy="mod",name='embedding_x1',validate_indices=True,max_norm=None)
            embedding_x2=tf.nn.embedding_lookup(Embeddings,self.input_x2,partition_strategy="mod",name='embedding_x2',validate_indices=True,max_norm=None)
        with tf.variable_scope('output'):
            state1=self.stackedRNN(embedding_x1, self.dropout_keep_prob, 'side1', hidden_units)
            state2=self.stackedRNN(embedding_x2, self.dropout_keep_prob, 'side2', hidden_units)
            distance1=tf.sqrt(tf.reduce_sum(tf.square(state1)))
            distance2=tf.sqrt(tf.reduce_sum(tf.square(state2)))
            numerator=tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(state1,state2))))
            d_raw=tf.div(numerator,(distance1+distance2))
            distances=tf.reshape(d_raw,shape=[-1],name='distance')
        with tf.variable_scope('loss'):
            self.loss=self.contrastive_loss(self.input_y,distances,batch_size)
        with tf.variable_scope('accuracy'):
            self.accr=self.accuracy(distances,self.loss)

    def stackedRNN(self, x, dropout, scope, hidden_units):
        """
            设定层数，设置cells   dorpoutcell,变形cell    输入dynamicrnn，获取state。
        """
        cells=list()
        with tf.name_scope(scope),tf.variable_scope(scope):
            for _ in range(n_layers):
                basictcell=tf.nn.rnn_cell.BasicLSTMCell(hidden_units, forget_bias=1.0,state_is_tuple=True, activation=None, reuse=None)
                dropoutcell=tf.nn.rnn_cell.DropoutWrapper(basictcell, input_keep_prob=1.0, output_keep_prob=dropout,state_keep_prob=1.0,
                                                          variational_recurrent=False,input_size=None, dtype=None, seed=None)
                cells.append(dropoutcell)
            multicell=tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
            outputs, state=tf.nn.static_rnn(multicell,x,initial_state=None,dtype=tf.float32,sequence_length=None,scope=None)
        return state
    def contrastive_loss(self, y, d, batch_size):
        tmp1=y*tf.square(d)
        tmp2=(1-y)*tf.square(tf.maximum(1-d,0))
        return (tmp1+tmp2)/batch_size/2
    def accuracy(self,disatnces_in,y):
        accr=tf.reduce_mean(tf.cast(tf.equal(tf.rint(disatnces_in),y),dtype=tf.float32))
        return accr


