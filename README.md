siamese_dssm
 
v1.0

   simaese 判断句子相似度。

v2.0

   添加 基于siamese的句子相似度排序，类似于 搜索召回

v3.0
    
    添加 dssm，判断句子相似度

v4.0
    
    dssm和 siamese融合，强化句子相似度排序
 
 
 目前处于v1.0阶段
 
 入口文件：train.py     执行方式：python train.py
优化

 语料：corpus.txt

优化方式：
    目前已做优化：
        
        1.余弦距离计算方式完善
        
        2.添加激活函数
        
    尚待优化：
	    1.更改相似度计算方式及损失函数，余弦距离+方差 改为其他诸如 交叉熵等等；

        2.更改句子向量获取方式，rnn改为cnn；
