#  Notes about QA

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
- 第一层输入word embedding，第二层输入第一层的输出，每一层网络都可能与不同级别的关系表示相匹配，两层网络的表示互补
- 残差网络：把前面的层和后面的层连起来，防止梯度消失
- 将两层的输出向量相加，过max-pooling；或先分别过max-pooling再相加
- hierarchical training使得第二层fit第一次的残差
- ranking loss to maximizing the margin 

KBQA

- entity re-ranking：用LSTM关系提取模型选出top-l个关系，用这top-l的关系的得分与entity linker的得分结合
- 数据：https://github.com/Gorov/KBQA_RE_data



## CFO: Conditional Focused Neural Question Answering with Large-scale Knowledge Bases 

ACL 2016 (Lei Li)

Conditional Factoid Factorization: $p(s,r|q)=p(r|q)\cdot p(s|q,r)$

知识库中的关系数量比实体少，所以先找关系后找实体

- $p_{\theta_r}(r|q) = softmax(f(q)^T E(r))$
  - f(q)：question embedding: 2-layer BiGRU final hidden state + linear lear
  - E(r)：embedding（随机初始化）
- $p_{\theta_s}(s|q,r) = softmax(g(q)^T E(s) + \alpha h(r,s))$
  - g(q)：question embedding
  - h(r,s)：关系-实体 得分（相连为1，否则为0）
  - E(s)：TransE，或type vector（固定的，使用类别信息，1-hot，这种情况下要给g(q)加sigmoid）
  - 实验结果表明TransE pretrain效果不如type vector
- Focused Pruning 
  - 给定知识库K和问题q，输出一个较小的subject/relation子集
  - N-gram pruning：问题中一定会出现实体的子串（问题：non-subject-mention noise）
  - 用sequence labeling network（双层BiGRU+GRU）：p(w|q) 表示n-gram w是q的subject mention的概率



## Semantic Parsing on Freebase from Question-Answer Pairs 

EMNLP 2013 Jonathan Berant , Andrew Chou, Roy Frostig, Percy Liang 

挑战：将自然语言短语映射到logical predicate

Aligning：根据句子中的短语找到对应的谓词（coarse）

Bridging：根据相邻的谓词生成谓词，而不是根据words