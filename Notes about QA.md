# Notes about QA

## Answering Natural Language Questions via Phrasal Semantic Parsing 

Kun Xu 2014

- 依存分析 shift-reduce
- 用规则将依存结构转化为SPARQL查询
- 实例化：
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
- 联合推理（joint inference）：

## Information Extraction over Structured Data: Question Answering with Freebase 

ACL 2014

