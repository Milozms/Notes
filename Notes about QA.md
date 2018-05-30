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

