# 关系提取综述

整理与翻译：张茂森

## 摘要
本文内容整理并翻译自Nguyen Bach与Sameer Badaskar所写的A Review of Relation Extraction一文，与Shantanu Kumar所写的A Survey of Deep Learning Methods for Relation Extraction一文。

## 1 导论
关系提取是指从无结构文本中提取实体之间的关系，从而将其转化为结构化信息的过程。关系定义在实体的元组$ t = (e_1, e_2, ..., e_n) $ 之上。大多数关系提取系统着重于提取二元关系，例如*坐落于（卡耐基梅隆大学，匹兹堡）*，*父亲（Manuel Blum, Avrim Blum）*。高阶关系与之类似。

有监督方法：将关系提取问题公式化为二分类问题：

- 基于特征的监督方法(Kambhatla, 2004)(Zhao & Grishman, 2005)
- 基于核的监督方法：能够高效地在多项式时间内搜索足够大的特征空间（通常是指数级的，在某些例子中是无穷的），并且不需要显式地表示特征。
  - 树形核、子序列核、依存树核。

半监督方法和bootstrapping方法：

- DIPRE(Brin, 1998)和Snowball(Agichtein & Gravano, 2000)，这类方法只需要少量的人工标记的种子实例或少量的人工提取模式来启动训练，使用了类似于Yarowsky的词语消歧义算法的半监督方法；
- KnowItAll(Etzioni, 2005)和TextRunner(Banko, 2007)提出了使用自训练关系二分类器的大规模关系提取系统。

## 2 有监督方法
出于简便，我们仅讨论两个实体之间的二元关系。多元关系在之后讨论。

给定一个句子$S = w_1, w_2, ..., e_1, ..., w_j, ..., e_2, ..., w_n$，其中$e_1, e_2$是实体，我们给出一个映射函数$f$：
$$
f_R(T(S)) = 
\left\{  
             \begin{array}{lr}  
             +1, & If\ e_1\ and\ e_2\ are\ related\ according\ to\ relation\ R \\
             -1, & otherwise   
             \end{array}  
\right.
$$

其中$T(S)$是从S中提取的特征。映射函数$f$为+1表示实体之间存在关系，$f$为-1表示实体之间不存在关系。如果标记的正样例和负样例数据可以用来训练，则函数$f$可以被构建成一个判别分类器（例如感知器、支持向量机等），经过预处理（词性标注、依存分析）之后用文本级特征来训练。分类器的输入也可以是结构化表示，如解析树。由于分类器训练的输入的不同，监督方法也可以分为基于特征的方法和核方法。

### 2.1 基于特征的方法
从文本中提取的句法和语义的特征，可以作为判定句子中的实体是否有某种关系的线索。

- 句法特征：实体本身，实体类别，实体之间的词序列，实体之间的词的数量，解析树中实体之间的路径。
- 语义特征：依存分析树中实体之间的路径。

这些特征都可以在特征向量中表示并输入到分类器中。

自然语言处理的应用往往涉及输入数据的结构化表示，因此仅使用相关特征的子集很难达到最优。作为补充，人们设计了专用于关系提取的核，用于更充分地利用输入数据的表示。

### 2.2 核方法
关系提取的核方法基于字符串核（string-kernels），这一方法源于文本分类。给出字符串x和y，字符串核基于它们的公共子串数量计算它们的相似度。每个字符串被映射到高维空间中的向量，每个维度对应一个特定的子串是否出现。例如字符串cat可做如下表示：

$$
\phi(x=cat) = [\phi_a(x)\ ..\ \phi_c(x)\ ..\ \phi_t(x)\ ..\ \phi_{at}(x)\ ..\ \phi_{ca}(x)\ ..\ \phi_{ct}(x)\ ..\ \phi_{cat}(x)] 
$$
其中

$$
\phi_a(x)=\phi_c(x)=\phi_t(x) = \lambda
$$
$$
\phi_{at}(x)=\phi_{ca}(x)=\phi_{ct}(x) = \lambda^2 
$$
$$
\phi_{cat}(x) = \lambda^3
$$

$\lambda$被称为衰减因子，用于惩罚较长的、不相邻的子序列。事实上$\phi_{ct}(cat)$应该比$\phi_{at}(cat)$和$\phi_{ca}(cat)$受到更多惩罚（$\lambda^3$），因为不相邻。对于索引为$i = i_1, i_2, ..., i_{|u|}$的子串$u$，令$u$的长度为$l(i)=i_{|u|}-i_1+1$，$u$在字符串x的高维空间中对应的坐标值为

$$
\phi_u(x) = \sum_{i:u=x[i]} \lambda^{l(i)}
$$
（我的理解：这里的求和的意思是，如果子序列u在x中出现多次，则把多次的$\phi$相加。上述例子中ct的衰减因子应为3次方。）

若U是字符串x和y包括的所有可能的子串的集合，则x和y的核相似度为
$$
K(x,y)=\phi(x)^T \phi(y) = \sum_{u \in U}\phi_u(x)^T \phi_u(y)
$$
这一相似度的计算可以使用动态规划，复杂度为$O(|x|||y|^2)$。

如果x和y是两个对象，$K(x,y)$可以理解为：通过计算他们之间结构的共同点，从而计算他们的相似度。

在关系提取中，如果$x^+$和$x^-$分别表示训练数据中的正样例和负样例，而y表示测试样例，则$K(x^+,y)>K(x^-,y)$意味着y包含一个关系（y更接近正样例）。实际上$K(x,y)$就是支持向量机和感知机中使用的相似性函数。在关系提取中，$x^+$、$x^-$和y可以用实体之间的词序列和包含实体的解析树表示。

#### 2.2.1 特征包核(Bag of features)
以句子"*The headquarters of* **Google** *are situated in* **Mountain View**"为例。已知Google和Mountain View是实体，词situated和headquarters表明*机构-位置*关系，则我们可以得出：实体的上下文可以用来判定它们之间是否有关系。句子s中包含实体$e_1$和$e_2$，可以被表示为$s = sb\ e_1\ sm\ e_2\ sa$，其中sb、sm、sa分别表示在实体之前、之间、之后的上下文词。给定测试样例句子t，t包含实体实体$e'_1$和$e'_2$，使用子序列和计算两个句子的三组上下文之间的相似度（这里使用词语级序列，而不是字符级序列），得到三个子核，三个子核结合就得到了最终的核。用子序列核结合支持向量机，显著提高了关系提取的准确率。

#### 2.2.2 树形核

树形核用于计算两个entity-augmented shallow parse tree结构之间的相似度。（之所以使用shallow parse tree而不用完全的解析树，是因为shallow parse tree的鲁棒性和稳定性更好。）Shallow parser用实体信息和短语信息来使树增强。给定一个shallow parse：

- 正样例：枚举最低的覆盖相关的实体的子树；

- 负样例：如果子树不覆盖两个相关的实体，或这个子树不是覆盖两个相关实体的最低的子树。

样例树中的每个节点有三个属性：
- entity role(人，机构，非实体) 

- chunk type(如NP, VP)

- 节点覆盖的文本

树形核的核函数是在字符串核的基础上改进的，计算两个shallow parse tree之间的结构共同点：计算两个shallow parse tree的公共子树个数的加权和。如下递归计算：
给定两个子树，根节点分别为$T_1$和$T_2$：
- 1、比较$T_1$和$T_2$的三种属性，如果不匹配，则返回相似度0；
- 2、如果三种属性匹配，则在总分上加1，并比较$T_1$和$T_2$的子节点序列。记$children(T_1)$和$children(T_2)$分别为$T_1$和$T_2$的子节点序列，他们的相似度由公共子节点子序列的数量计算，计算的方法与字符串核相同。

如果m和n是shallow parse tree的节点数，则核的计算复杂度为$O(mn^3)$。

##### 2.2.2.1 依存树核

2005年Bunescu和Mooney观察到：依存树中两个实体之间的最短路径已经包含了充足的信息：若$e_1$和$e_2$是句子中的两个实体而p是它们之间的谓词，则$e_1$和$e_2$之间的最短路径经过p，因为$e_1$和$e_2$是p的参数。

给定一个句子和它的依存树。设两个实体$e_1$和$e_2$之间的最短路径为$P=e_1->w_1->…->w_k<-…<-w_n<-e_2$。$w_i$为P中的词，箭头表示依存关系的方向。直接把P作为特征的效果并不好（稀疏），因此我们把P中的词替换为“词标签”（word classes，如POS）。特征向量用笛卡尔积表示：
$$
x=[e_1,C(e_1)]\times[\rightarrow]\times[w_1,C(w_1)]...\times[\rightarrow]\times[w_k,C(w_k)]...\times[\leftarrow]\times[w_n,C(w_n)].\times[\leftarrow]\times[e_2,C(e_2)]
$$
其中$C(w_i)$表示词标签（如POS）。在这个特征空间的核表示为：
$$
K(x,y)=\left\{  
             \begin{array}{lr}  
             0, & If|x|\neq|y|\\
             \prod_{i=1}^{|x|}f(x_i,y_i), & otherwise   
             \end{array}  
\right.
$$
其中$f(x_i,y_i)$为$x_i$和$y_i$的公共“词标签”数。

这种核的好处是：计算只需要线性时间，特征空间更简洁，同时性能更好。



## 3 半监督方法







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

#### 1. 候选生成（生成训练数据）

###### Domain-agnostic实体识别：Distantly-supervised文本切分

- DIstant supervision只标注了一小部分entity mention，如果用序列标注模型训练，会产生大量false negative
- 对切分质量建模的方法：短语质量+POS模式质量，用数据集中的正样例估计切分质量
- 1.从数据集中找到高频的连续的pattern（词序列和POS序列）
- 2.提取corpus-level concordance和句子级句法信号特征，训练两个随机森林分类器
- 3.用估计出来的质量分数寻找最好的切分：最大化joint segmentation quality
- 4.用切分的语料计算rectified特征，重复上述两步直到收敛



#### 问题

- 公式1？
- rectified？
- random forest在哪实现的？



### HypeNet

Improving Hypernymy Detection with an Integrated Path-based and Distributional Method, Vered Shwartz, Yoav Goldberg, Ido Dagan. (<https://arxiv.org/pdf/1603.06076>)

### SDP-LSTM

Classifying relations via long short term memory networks along shortest dependency path, Xu Yan, Lili Mou, Ge Li, Yunchuan Chen, Hao Peng, and Zhi Jin. 2015. (<http://www.aclweb.org/anthology/D15-1206>)

- 先用stanford parser得到依存树，从依存树中提取最短依存路径（SDP）作为网络的输入。
- SDP中的四类信息（words, POS tags, grammatical relations, WordNet hypernyms ）构成四个channel。每个channel输入embedding向量
- SDP分位左半段和右半段，由公共祖先结点分割，左右半段分别输入两个LSTM，输出过max-pooling，然后连接
- 四个channel的输出向量连接到一起，在过一个hidden layer+softmax，用于分类

### CNN-PE

CNN with input word embedding + position embedding

### BGRU+2ATT

Bidirectional GRU +word-level attention + sentence-level attention from Neural Relation Extraction with Selective Attention over Instances (<http://www.aclweb.org/anthology/P16-1200>)

#### data_preprocess.py

将TACRED数据集从json格式转化成行格式

#### initial.py：

- Init: 将数据集中的样例以entity pair作为key构建dict；然后按顺序，将entity pair相同的样例放到一起，entity pair相同的集合内部把label相同的放到一起。train_sen中每个元素为给定entity pair和label的所有句子，train_ans为相应的label。（entity pair和label都相同的记为一组）
- Separate: 将word embedding和position embedding分开，便于placeholder输入

#### train_GRU.py

- 将训练数据分为若干batch，每个batch的size为big_num（设定的参数），将batch输入train_step
- Train_step：将相同entity pair和label的样例（组）拆分为单个的样例，用total_shape记录每组的句子数（按顺序）

#### network.py

在计算sentence-level attention时，把同组（entity pair和label都相同）的样例放在一起计算，（batch_size是组size），每个组给出一个预测。



