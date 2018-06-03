# Notes about QA

## Answering Natural Language Questions via Phrasal Semantic Parsing 

Kun Xu 2014

- 将问题切割成四种类别的短语（变量，实体，关系，类别），序列标注模型
- 短语级别的依存分析 shift-reduce
- 用规则将依存结构转化为SPARQL查询
- 实例化：将句子中的mention映射到freebase中的实体或关系
  - 实体：Freebase Search API
  - 关系：Co-occurrence Matrix contributed by Yao 

## Question Answering on Freebase via Relation Extraction and Textual Evidence 

ACL 2016 Kun Xu

- 基于关系提取的方法，从Wikipedia文本中寻找额外的证据
- 从Freebase提取候选答案，在Wikipedia上做验证
- 多关系问题的处理：基于依存分析树，用规则将多关系问题分解，得到若干子问题，取子问题的交集得到答案
- 实体链接：S-MART
- 关系提取：MCCNN，输入syntactic和sentential信息：
  - syntactic：依存树中实体和question word之间的最短路径，连接词向量、依存树的边的方向、依存关系标签
  - sentential：句子中的所有词（除了question word和实体）
- 联合推理（joint inference）：使用SVM rank classifier对（实体，关系）对进行打分
  - 实体clue：实体链接系统返回的分数；词语的overlap；
  - 关系clue：MCCNN的输出；通过问题检索freebase关系的TF-IDF分数；
  - 答案clue：question word和答案类别的co-occurrence

## Information Extraction over Structured Data: Question Answering with Freebase 

ACL 2014

- Question Graph
  - 基于依存关系：question word（疑问词）, question focus（问题类别）, question verb（句子核心动词）, question topic；
  - 将依存关系转化为genereic question graph：标注问题特征和命名实体，将介词、限定词和标点符号去掉——问题特征图；
  - 提取特征：对每个边e(s,t)，提取：s, t, s|t, s|e|t
- Freebase Topic Graph
  - Topic Graph: 选择topic结点K阶邻域的节点
  - 用结点的关系和property作为特征
  - Alignment model：将自然语言问题映射到知识库的关系，P(relation|question)
  - 将问题特征和topic结点特征组合
- 关系映射：将自然语言问题映射到知识库的关系
  - 朴素贝叶斯：$P(R|Q)=\prod_w P(w|R)P(R)$
  - 将关系分解为“子关系”（Freebase中用点分割）:$P_{backoff}(R|Q)=\prod_r\prod_w P(w|r)P(r)$
- Co-occurrence Matrix：用于计算概率



## Improved Neural Relation Detection for Knowledge Base Question Answering 

ACL 2017

通常的关系识别（general relaiton detection，在IE中常被叫做关系提取/关系分类）和用于KBQA的关系识别之间存在明显的gap：

- 通常的关系识别只有有限的关系数（一般小于100），而KB中的关系数上千；
- 在KBQA中，测试数据中经常有训练集中从未出现过的关系；
- 在KBQA中经常需要识别一系列的关系，而不是一个关系。

KB关系的表示（表示不同级别的抽象）：

- Relation-level：表示为单个token；
- Word-level：拆成一系列单词，看做单词序列：用于应对测试数据中出现训练数据中未出现的关系这种情况；
- 用两个BiLSTM（共享参数的）分别得到Relation-level和Word-level的hidden representation，加max-pooling得到关系的向量表示
- 用word sequence的LSTM的final state作为relation sequence的initial state，作为对未见过的关系的弥补

用深度BiLSTM做问题表示：

- relation name与长短语匹配，relation word与较短的短语匹配，因此需要问题的向量表示能够总结不同长度的短语信息

残差网络：把前面的层和后面的层连起来，防止梯度消失



## Large-Scale QA-SRL Parsing 

QA-SRL：给定一个句子，对于句子中每个动词，提出若干个问题，每个问题的答案对应了一个semantic role。

数据标注分为两步：生成和验证。验证阶段标注者回答问题，若回答不了则标为无效

预处理：用CoreNLP标POS；用POS识别动词，用启发法过滤掉辅助动词，保留实义动词；

#### 模型：

- Span detection：给定动词，从句子中选出一些span作为动词的参数；
- Question generation：对每个span预测出一个问题。
- 两部分都基于LSTM对句子编码: $H_v$；
- LSTM的输入是词向量+二元特征（表示这个词是不是当前要考虑的动词）；
- 两部分的LSTM参数互相独立。

##### Span Detection

- BIO
- Span-based：对句子中所有$n^2$个可能出现的span都预测它是不是动词的参数：对span(i,j)，将两端点位置的LSTM的输出向量连接$s_{vij}=[h_{vi}, h_{vj}]$得到的向量过MLP+全连接层+激活函数得到这个span是否为参数的概率
- Span-based效果更好

##### Question Generation

- 将问题划分为若干slot：Wh, Aux, Subj, Verb, Obj, Prep, Misc

- Local model：将span对应的$s_{vij}$向量过MLP+全连接层+softmax输出位置k的slot的概率分布（不同的k对应的权重参数不同，不同的slot之间互相独立）
- Sequence model：以slot为单位的LSTM，每个cell的输入是$s_{vij}$和前一个cell输出向量相连，输出过向量MLP+全连接层+激活函数+softmax得到slot的概率分布
- 结果：Sequence的exact match和partial match更高，local的Slot-level accuracy更高
- joint：span-based+seq效果好

##### 数据扩展：

- 用模型对已有标注的句子生成问题，过滤掉重复的（答案和已标注答案重叠的，或问题和已标注问题一样的），剩下的问题就是标注时可能漏掉得的问题
- 在训练集上，把span detection的阈值调低再生成问题（为了得到更多潜在的问题）
- 用标数据的流水线来人工评测这些的问题，46017个（50%）被标注为有效
- 数据总量扩充20%
- 过滤掉扩展数据集中 答案与一个原始问题的答案有两处重叠的问题，过滤后的数据总量比原始扩充11.5%