# 一些调优的方法：
## 基于文本数据和分片策略：
### Chunk_size的选择（文档处理）
[choose_chunk_size](https://github.com/HeavyCrown/RAG_Tips/blob/main/choose_chunk_size.py)
1. 构造出可选的chunk_size的列表[128，256，...]
2. 构造基于文档的questions
		'data_generator = DatasetGenerator.from_documents(eval_documents)
        eval_questions = data_generator.generate_questions_from_nodes()'
3. 使用大模型作为评估模型，记录每个chunk_size的指标（平均检索时间，平均忠诚度和平均相关性）
4. 选择指标最优的chunk_size

### 命题分片（文档处理）
1. 原则：将文章分解成简洁、完整、有意义的句子以便更好地控制和处理特定查询(特别是提取知识)。事实性语句，也叫命题（xxx is xxx）
2. 实现：LLM+Prompt形成文本内容的事实生成（few-shot prompting）
3. 评估：基于大模型的准确性、清晰性、简洁性和完整性评估
-----------------------------------------------------------
## Query增强：
### Query转换技术：可以单独使用，也可以结合使用
[query_transformations](https://github.com/HeavyCrown/RAG_Tips/blob/main/query_transformations.py)
1. Rewriting：通过LLM将查询变得更加详细、具体
2. step-back prompting：通过LLM生成更加广泛、通用的query，帮助检索相关的背景信息
3. 子查询分解：通过大模型将复杂问题分解，以便更加全面的查询

### 生成假设性文档
[HypotheticalDocumentsEmbedding](https://github.com/HeavyCrown/RAG_Tips/blob/main/HypotheticalDocumentEmbedding.py)
1. 通过llm将query生成为一个字数为chunk_size的文档
2. 好处：
	1. 扩展成完整文档后，可以捕获更细微和相关的匹配
	2. 对于难以直接匹配的复杂查询特别有效
	3. 可以更好地捕获原始问题背后的上下文和意图
------------------------------------------------------------
## 上下文和内容增强
### 添加分块的header来增强上下文信息
1. 通过LLM为文档生成一个描述性标题
2. 提供更多的上下文内容
3. 有助于检索较复杂的文档内容

### RSE
[RES](https://github.com/HeavyCrown/RAG_Tips/blob/main/RSE.py)
1. 问题：根据问题的不同，我们需要动态的改变召回块的大小
	1. 有些问题需要更大的上下文，需要更大的chunk
	2. 一些简单的查询则最好由小的chunk来处理
2. 解决方案：需要一个更加动态的系统，既可以检索小的chunk，也可以检索到大的chunk
3. 详细步骤：
	1. 计算不同chunk的相似度得分并rerank
	2. 用类似聚类的思想，将相似的chunk合并为一个segment
	3. 选择合适的segment作为召回的内容
	4. 结合llm生成答案

### 根据语义划分chunk（Semantic Chunking）
[semantic_chunking](https://github.com/HeavyCrown/RAG_Tips/blob/main/semantic_chunking.py)
1. 通过尝试保持chunk内部语义的一致性来提高检索信息的质量
2. 对于处理长而复杂的文档（需要维护上下文）很有价值
3. 实现：
	1. 通过LangChain的SementicChunker方法
	2. 提供了三种断点类型：
		1. percentile（百分位）：计算句子之间的所有差异，然后拆分任何大于X百分位的差异
		2. standard_deviation（标准偏差）：任何大于X个标准差的差异都会被拆分
		3. interquartile（四分位间距）

### 文档检索中的上下文压缩（Contextual Compression）
[context_compression](https://github.com/HeavyCrown/RAG_Tips/blob/main/context_compression.py)
1. 传统的文档检索系统通常返回整个块或文档，其中可能包含不相关的信息
2. 上下文压缩通过提取和压缩检索到的最相关的部分来解决这一问题
3. 技术细节：langchain框架
	1. 将基本检索器和压缩器组合成一个ContextualCompressionRetriver
	2. 构建RetrievalQA链，继承压缩检索器
4. 好处：
	1. 提高召回内容的相关性
	2. 提高效率，减少了LLM需要处理的文本量
	3. 减少了成本，减少了输入LLM的token数量
------------------------------------------------------------
## 检索方法增强
### 融合检索（Fusion Retrieval）
[FusionRetrieval]()
1. 通过结合相似度检索和关键字检索（BM25）的方式来优化检索结果
2. 两种检索方法各有优点和缺点，融合后可以增强检索系统的健壮性和准确性，有效地处理更加广泛的查询
3. 过程细节：
	1. 对Query进行基于向量和BM25的两种检索
	2. 将两种方法的得分归一化到一个尺度
	3. 计算这些分数的加权和（alpha参数控制权重）
	4. 根据综合得分进行排名，并返回前k个结果
4. 优点：
	1. 提高检索质量：通过结合两种检索方式，系统可以捕获概念相似和关键字匹配两种维度的结果
	2. 灵活性：alpha参数可以灵活调整
	3. 健壮性：组合方法可以有效处理更大范围的查询，减轻单个方法的弱点