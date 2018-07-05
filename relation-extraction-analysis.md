# Relation Extraction

## Position-aware Attention and Supervised Data Improve Slot Filling

EMNLP 17

用向量表示不同位置，加attention，将position embedding作为attention的query



## Neural Relation Extraction with Selective Attention over Instances

PCNN+ATT

ACL16

用多个instance之间的sentence-level attention，减少有噪声的样例的权重，从而减少训练数据中的噪声的影响。

对于每个entity pair：(head, tail)，集合S包括了n个句子：x1, x2, ..., xn。为了预测关系r，用一个向量s表示整个集合S：s为表示x1, x2, ..., xn的向量的加权和，权重计算：xi的向量表示 * 对角权重矩阵A * 要预测的关系r的向量表示，再过softmax。得到的s过线性层，再过softmax层，得到预测关系的概率分布。

对于每对实体，把他们出现的所有instance放到一起预测。

（一对实体只有一个标签吗？公式11中的ri指的是哪个？）

## Improving Hypernymy Detection with an Integrated Path-based and Distributional Method

ACL16

HypeNet

hypernymy relation：上位关系（从属关系？）



## CoType

#### 候选生成（生成训练数据）

######  Domain-agnostic实体识别：Distantly-supervised文本切分

- DIstant supervision只标注了一小部分entity mention，如果用序列标注模型训练，会产生大量false negative
- 对切分质量建模的方法：短语质量+POS模式质量，用数据集中的正样例估计切分质量
- 1.从数据集中找到高频的连续的pattern（词序列和POS序列）
- 2.提取corpus-level concordance和句子级句法信号特征，训练两个随机森林分类器
- 3.用估计出来的质量分数寻找最好的切分：最大化joint segmentation quality
- 4.用切分的语料计算rectified特征，重复上述两步直到收敛
- 实际代码里用的是stanford NER

###### 生成relation mention

- 对每一对句子s中的实体mention $(m_a, m_b)$ ，生成两个relation mention $z_1 = (m_a, m_b, s)$和$z_2 = (m_b, m_a, s)$
- 抽样30%的unlinkable entity mention作为负样例

#### Joint entity and relation embedding

- 关系向量空间
- 实体向量空间

##### 1.对relation mention的type建模：

- 假设1：mention-feature co-occurrence：在语料中共享多个文本特征的两个relation mention倾向于有相似的relation type（在embedding space中接近）
- Second-order proximity：有相似的neighbor的object之间也相似
- 用LINE中的二阶近似度对relation mention和feature之间的关系建模 $p(f_j|z_i)$ ，$f_j$是特征，$z_j$是relation mention
- 最小化目标：$L_{ZF}=-\sum_{z_i} \sum_{f_j} w_{ij}\log p(f_j|z_i)$，其中$w_{ij}$是co-occurrence frequency
  - 为了更高效的计算，避免在所有的特征上做加法，使用负采样：
  - 对每个$(z_i,f_j)​$，根据噪声分布（？）sample多个false feature
  - 将上式中的$\log p(f_j|z_i)$替换为：$\log\sigma(z_i^T c_j)$ + 负样例
  - 已有的embedding方法往往基于local consistent assumption，但relation mention和relation label之间的关系可能是"false" association
- 假设2：partial-label association：relation mention的embedding向量应该与和它最相关的candidate type相似
- 公式4：最大化$z_i$与相关的r的相似度（点乘），同时最小化$z_i$与不相关的r'的相似度。将公式4加入最终的

##### 2.对entity mention的type建模



#### 问题

- 公式1？
- rectified？
- random forest在哪实现的？



## LINE 

- 一阶近似度：边权重
- 二阶近似度：结点与其他节点之间的一阶近似度构成的向量，向量之间的相似度

#### 一阶

- 概率分布$$p_1(i,j) = \frac{1}{1+\exp(-u_i^T u_j)}$$,  $\hat{p}_1(i,j)=\frac{w_ij}{W}$, $W=\sum_{(i,j)\in E} w_{ij}$
- 最小化两个概率分布之间的距离（KL散度）：$O_1=-\sum_{(i,j)\in E} w_{ij}\log p_1(i,j)$

#### 二阶

- $p_2(v_j|v_i)=\frac{\exp(u_j^T u_i)}{\sum_{k=1}^{|V|}\exp(u_k^T u_i)}$
- $\hat{p}_2(v_j|v_i)=\frac{w_ij}{d_i}$, $d_i=\sum_{k\in N(i)} w_{ik}$
- $O_2=-\sum_{(i,j)\in E} w_{ij}\log p_2(v_j|v_i)$

### HypeNet

Improving Hypernymy Detection with an Integrated Path-based and Distributional Method, Vered Shwartz, Yoav Goldberg, Ido Dagan. (<https://arxiv.org/pdf/1603.06076>)

### SDP-LSTM

Classifying relations via long short term memory networks along shortest dependency path, Xu Yan, Lili Mou, Ge Li, Yunchuan Chen, Hao Peng, and Zhi Jin. 2015. (<http://www.aclweb.org/anthology/D15-1206>)

- 先用stanford parser得到依存树，从依存树中提取最短依存路径（SDP）作为网络的输入。
- SDP中的四类信息（words, POS tags, grammatical relations, WordNet hypernyms ）构成四个channel。每个channel输入embedding向量
- SDP分位左半段和右半段，由公共祖先结点分割，左右半段分别输入两个LSTM，输出过max-pooling，然后连接
- 四个channel的输出向量连接到一起，在过一个hidden layer+softmax，用于分类

#### sprnn_model.py

max_over_time(inputs, index, seq_lens)：返回batch中第index个instance的max-pooling

原论文中用的max-pooling，但此代码可以在pooling和attention中二选一

#### train.py

第40行import sprnn_model



### CNN-PE

CNN with input word embedding + position embedding

### BGRU+2ATT

Bidirectional GRU +word-level attention + sentence-level attention from Neural Relation Extraction with Selective Attention over Instances (<http://www.aclweb.org/anthology/P16-1200>)

#### data_preprocess.py

将TACRED数据集从json格式转化成行格式

#### initial.py：

- Init: 将数据集中的样例以entity pair作为key构建dict；然后按顺序，将entity pair相同的样例放到一起，entity pair相同的集合内部把label相同的放到一起。train_sen中每个元素为给定entity pair和label的所有句子，train_ans为相应的label。（entity pair和label都相同的记为一组）

```python
train_sen = {}  # {entity pair:[[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...]}
train_ans = {}  # {entity pair:[label1,label2,...]} the label is one-hot vector
test_sen = {}  # {entity pair:[[sentence 1],[sentence 2]...]}
test_ans = {}  # {entity pair:[labels,...]} the labels is N-hot vector (N is the number of multi-label)
```

- organizing train data: 将train_sen中每个组（entity pair和label都相同）中的所有instance的集合作为一个train_x和train_y中的一个元素（**也就是train_x[idx]是一个组中的所有样例**）

- Separate: 将word embedding和position embedding分开，便于placeholder输入

#### train_GRU.py

- 131-150行：将训练数据分为若干batch，每个batch的size为big_num（设定的参数），将batch输入train_step
- train_step中的输入是一个batch，batch中的每个单元都是一个组中的所有样例
- Train_step：89-94行：将相同entity pair和label的样例（组）拆分为单个的样例
- 用total_shape记录每组的句子数（按顺序）
- 输入到sess.run中的total_word、total_pos是拆散后的（多个）组（**total_word[idx]是一个单独的样例**）

#### network.py

在计算sentence-level attention时，把同组（entity pair和label都相同）的样例放在一起计算，（batch_size是组size），每个组给出一个预测。

- sentence-level attention之前都是单个样例的计算
- 115行：sen_repre中的每个元素是一个组中的全部样例